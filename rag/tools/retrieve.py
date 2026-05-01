from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config import RAGConfig
from ..retrievers import reciprocal_rank_fusion
from ..schemas import RetrievalHit, RetrievalResult
from ..text_utils import normalize_whitespace, unique_preserve_order
from .rerank import CohereRerankTool


class RetrievalTool:
    name = "retrieve"

    def __init__(
        self,
        *,
        retriever_map: dict[str, object],
        reranker: CohereRerankTool,
        config: RAGConfig,
    ):
        self.retriever_map = retriever_map
        self.reranker = reranker
        self.config = config

    def run(
        self,
        *,
        query: str,
        plans: list[dict[str, object]] | None = None,
        retrieval_modes: list[str] | None = None,
        topk: int | None = None,
    ) -> dict[str, object]:
        normalized_query = normalize_whitespace(query)
        active_topk = topk or self.config.default_topk
        normalized_plans = self._normalize_plans(
            query=normalized_query,
            plans=plans,
            retrieval_modes=retrieval_modes,
        )
        retrieval_budget = self._build_retrieval_budget(
            requested_topk=active_topk,
            plan_count=len(normalized_plans),
        )

        retrieval_results = self._parallel_retrieve(
            normalized_plans,
            int(retrieval_budget["candidate_topk_per_retriever"]),
        )
        fused_candidate_limit = max(int(retrieval_budget["effective_topk"]) * 3, 8)
        fused_hits = reciprocal_rank_fusion(
            retrieval_results,
            topk=fused_candidate_limit,
            k=self.config.fusion_k,
        )
        final_hits = self.reranker.rerank(
            normalized_query,
            fused_hits,
            topn=int(retrieval_budget["effective_topk"]),
        )
        return {
            "query": normalized_query,
            "plans": normalized_plans,
            "retrieval_budget": retrieval_budget,
            "retrieval_results": retrieval_results,
            "fused_hits": final_hits,
            "used_queries": [str(plan["query"]) for plan in normalized_plans],
        }

    def _build_retrieval_budget(self, *, requested_topk: int, plan_count: int) -> dict[str, int]:
        per_plan_min = 2 if plan_count > 1 else requested_topk
        effective_topk = max(requested_topk, plan_count * per_plan_min)
        return {
            "requested_topk": requested_topk,
            "effective_topk": effective_topk,
            "plan_count": plan_count,
            "per_plan_min": per_plan_min,
            "candidate_topk_per_retriever": max(per_plan_min * 3, requested_topk * 2, 6),
        }

    def _normalize_plans(
        self,
        *,
        query: str,
        plans: list[dict[str, object]] | None,
        retrieval_modes: list[str] | None,
    ) -> list[dict[str, object]]:
        if not plans:
            return [
                {
                    "query": query,
                    "retrieval_modes": self._normalize_modes(retrieval_modes),
                }
            ]

        normalized_plans: list[dict[str, object]] = []
        for item in plans:
            plan_query = normalize_whitespace(str(item.get("query") or query))
            normalized_plans.append(
                {
                    "query": plan_query,
                    "retrieval_modes": self._normalize_modes(item.get("retrieval_modes")),
                }
            )
        return normalized_plans

    def _normalize_modes(self, retrieval_modes: object) -> list[str]:
        if not isinstance(retrieval_modes, list):
            retrieval_modes = list(self.retriever_map)
        modes = [
            str(mode)
            for mode in retrieval_modes
            if str(mode) in self.retriever_map
        ]
        return unique_preserve_order(modes or list(self.retriever_map))

    def _parallel_retrieve(
        self,
        plans: list[dict[str, object]],
        candidate_topk: int,
    ) -> list[RetrievalResult]:
        future_map = {}
        results: list[RetrievalResult] = []

        with ThreadPoolExecutor(max_workers=self.config.retrieval_pool_size) as executor:
            for plan in plans:
                query = str(plan["query"])
                retrieval_modes = [str(mode) for mode in plan.get("retrieval_modes", [])]
                for retriever_name in retrieval_modes:
                    retriever = self.retriever_map[retriever_name]
                    future = executor.submit(retriever.retrieve, query, candidate_topk)
                    future_map[future] = (query, retriever_name)

            for future in as_completed(future_map):
                results.append(future.result())

        return results

    @staticmethod
    def summarize_hits(hits: list[RetrievalHit]) -> list[dict[str, object]]:
        summaries: list[dict[str, object]] = []
        for hit in hits:
            summaries.append(
                {
                    "chunk_id": hit.chunk_id,
                    "source": hit.chunk.source,
                    "title": hit.chunk.title,
                    "section": hit.chunk.section,
                    "score": round(float(hit.score), 4),
                    "rerank_score": (
                        round(float(hit.rerank_score), 4)
                        if hit.rerank_score is not None
                        else None
                    ),
                    "retrievers": list(hit.metadata.get("retrievers", [hit.retriever])),
                    "queries": list(hit.metadata.get("queries", [hit.query])),
                    "content_preview": hit.chunk.context[:280],
                }
            )
        return summaries
