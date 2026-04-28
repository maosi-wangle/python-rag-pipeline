from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import RAGConfig
from .knowledge_base import ChunkStore
from .llm import LLMClient
from .retrievers import GrepRetriever, KeywordRetriever, SemanticRetriever, reciprocal_rank_fusion
from .schemas import RetrievalHit, RetrievalResult
from .state import RetrievalRoundState
from .text_utils import normalize_whitespace, unique_preserve_order
from .tools import (
    AnswerGenerationTool,
    AnswerJudgeTool,
    CohereRerankTool,
    QueryDecomposeTool,
    QueryPlannerTool,
    QueryRewriteTool,
)


class ModularRAGOrchestrator:
    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RAGConfig()
        self.chunk_store = ChunkStore(self.config.data_path)
        self.llm = LLMClient(self.config)
        self.semantic_retriever = SemanticRetriever(self.chunk_store, self.config)
        self.keyword_retriever = KeywordRetriever(self.chunk_store, self.config)
        self.grep_retriever = GrepRetriever(self.chunk_store)
        self.retriever_map = {
            "semantic": self.semantic_retriever,
            "keyword": self.keyword_retriever,
            "grep": self.grep_retriever,
        }
        self.query_rewriter = QueryRewriteTool()
        self.query_decomposer = QueryDecomposeTool()
        self.query_planner = QueryPlannerTool(self.llm)
        self.reranker = CohereRerankTool(self.config)
        self.answer_generator = AnswerGenerationTool(self.llm)
        self.answer_judge = AnswerJudgeTool(self.llm)
        self.initialized = bool(self.chunk_store.chunks)

    def build_ragas_payload(
        self,
        query: str,
        *,
        topk: int | None = None,
        history: list[str] | None = None,
    ) -> dict[str, object]:
        round_state = self.run_round(
            query,
            history=history,
            topk=topk or self.config.default_topk,
            round_index=1,
            max_rounds=1,
        )
        return {
            "user_input": query,
            "retrieved_contexts": [hit.text for hit in round_state.fused_hits],
            "retrieved_context_ids": [
                hit.chunk.legacy_index if hit.chunk.legacy_index is not None else hit.chunk_id
                for hit in round_state.fused_hits
            ],
            "scores": [hit.final_score() for hit in round_state.fused_hits],
            "retrieval_mode": "agentic_modular_rrf",
            "topk": topk or self.config.default_topk,
        }

    def run_round(
        self,
        query: str,
        *,
        history: list[str] | None = None,
        topk: int | None = None,
        round_index: int = 1,
        max_rounds: int | None = None,
    ) -> RetrievalRoundState:
        history = history or []
        topk = topk or self.config.default_topk
        active_query = normalize_whitespace(query)
        plan = self.query_planner.plan(active_query, history)
        subqueries = self._plan_subqueries(active_query, plan)
        query_variants = self._plan_query_variants(active_query, history, subqueries, plan)
        selected_retrievers = plan.get("retrievers") or list(self.retriever_map)

        retrieval_results = self._parallel_retrieve(query_variants, topk, selected_retrievers)
        query_level_hits = self._parallel_rerank_groups(retrieval_results, topk)
        final_hits = reciprocal_rank_fusion(
            [
                RetrievalResult(query=planned_query, retriever="query_group", hits=hits)
                for planned_query, hits in query_level_hits.items()
            ],
            topk=topk,
            k=self.config.fusion_k,
        )
        answer_draft = self.answer_generator.generate(
            query=active_query,
            hits=final_hits,
            history=history,
        )

        judge_result = self.answer_judge.judge(
            query=active_query,
            response=answer_draft,
            hits=final_hits,
            subqueries=subqueries,
            round_index=round_index,
            max_rounds=max_rounds or self.config.max_rounds,
        )

        return RetrievalRoundState(
            round_index=round_index,
            input_query=query,
            active_query=active_query,
            query_variants=query_variants,
            subqueries=subqueries,
            retrieval_results=retrieval_results,
            query_level_hits=query_level_hits,
            fused_hits=final_hits,
            answer_draft=answer_draft,
            completeness=judge_result.completeness,
            relevance_score=judge_result.relevance_score,
            support_score=judge_result.support_score,
            need_followup=judge_result.need_followup,
            next_query=judge_result.next_query,
            missing_aspects=judge_result.missing_aspects,
            judge_reason=judge_result.reason,
        )

    def _plan_subqueries(self, query: str, plan: dict[str, object]) -> list[str]:
        planned = [str(item) for item in plan.get("subqueries", [])]
        if planned:
            return unique_preserve_order(planned)[:4]
        return self.query_decomposer.decompose(query)

    def _plan_query_variants(
        self,
        query: str,
        history: list[str],
        subqueries: list[str],
        plan: dict[str, object],
    ) -> list[str]:
        planned_rewrites = [str(item) for item in plan.get("rewritten_queries", [])]
        rewrite_modes: list[str] = ["chunk_like"]
        if history and len(query) <= 16:
            rewrite_modes.append("specific")
        variants = planned_rewrites + self.query_rewriter.rewrite(query, history=history, modes=rewrite_modes)
        if subqueries:
            variants.extend(subqueries)
        return unique_preserve_order(variants)[:6]

    def _parallel_retrieve(
        self,
        queries: list[str],
        topk: int,
        retriever_names: list[str],
    ) -> list[RetrievalResult]:
        candidate_topk = max(topk * 2, 6)
        future_map = {}
        results: list[RetrievalResult] = []

        with ThreadPoolExecutor(max_workers=self.config.retrieval_pool_size) as executor:
            for query in queries:
                for retriever_name in retriever_names:
                    retriever = self.retriever_map[retriever_name]
                    future = executor.submit(retriever.retrieve, query, candidate_topk)
                    future_map[future] = (query, retriever.name)

            for future in as_completed(future_map):
                result = future.result()
                results.append(result)

        return results

    def _parallel_rerank_groups(
        self,
        retrieval_results: list[RetrievalResult],
        topk: int,
    ) -> dict[str, list[RetrievalHit]]:
        grouped_results: dict[str, list[RetrievalResult]] = {}
        for result in retrieval_results:
            grouped_results.setdefault(result.query, []).append(result)

        query_level_hits: dict[str, list[RetrievalHit]] = {}
        future_map = {}
        candidate_limit = max(topk * 3, 8)

        with ThreadPoolExecutor(max_workers=self.config.rerank_pool_size) as executor:
            for query, results in grouped_results.items():
                fused_hits = reciprocal_rank_fusion(
                    results,
                    topk=candidate_limit,
                    k=self.config.fusion_k,
                )
                future = executor.submit(self.reranker.rerank, query, fused_hits, candidate_limit)
                future_map[future] = query

            for future in as_completed(future_map):
                query = future_map[future]
                query_level_hits[query] = future.result()

        return query_level_hits
