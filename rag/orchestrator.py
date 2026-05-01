from __future__ import annotations

from typing import Any

from .config import RAGConfig
from .knowledge_base import ChunkStore
from .llm import LLMClient
from .retrievers import GrepRetriever, KeywordRetriever, SemanticRetriever
from .text_utils import normalize_whitespace
from .tools.answer import AnswerGenerationTool
from .tools.finish import FinishTool
from .tools.judge import AnswerJudgeTool
from .tools.rerank import CohereRerankTool
from .tools.retrieve import RetrievalTool
from .tools.rewrite import QueryRewriteTool


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
        self.reranker = CohereRerankTool(self.config)
        self.query_transform_tool = QueryRewriteTool(self.llm)
        self.retrieve_tool = RetrievalTool(
            retriever_map=self.retriever_map,
            reranker=self.reranker,
            config=self.config,
        )
        self.answer_generator = AnswerGenerationTool(self.llm)
        self.answer_judge = AnswerJudgeTool(self.llm)
        self.finish_tool = FinishTool()
        self.initialized = bool(self.chunk_store.chunks)

    def default_retrieval_modes(self, query: str) -> list[str]:
        normalized_query = normalize_whitespace(query)
        modes = ["semantic", "keyword"]
        if len(normalized_query) <= 24 or any(
            token in normalized_query for token in ("定义", "命令", "参数", "路径", "配置")
        ):
            modes.append("grep")
        return modes

    def retrieve_once(
        self,
        query: str,
        *,
        plans: list[dict[str, object]] | None = None,
        retrieval_modes: list[str] | None = None,
        topk: int | None = None,
    ) -> dict[str, Any]:
        return self.retrieve_tool.run(
            query=query,
            plans=plans,
            retrieval_modes=retrieval_modes,
            topk=topk,
        )

    def generate_once(
        self,
        *,
        query: str,
        hits: list,
        history: list[str] | None = None,
        instruction: str | None = None,
        source_mode: str = "retrieval",
    ):
        return self.answer_generator.generate(
            query=query,
            hits=hits,
            history=history or [],
            instruction=instruction,
            source_mode=source_mode,
        )

    def build_ragas_payload(
        self,
        query: str,
        *,
        topk: int | None = None,
        history: list[str] | None = None,
    ) -> dict[str, object]:
        retrieval = self.retrieve_once(
            query,
            retrieval_modes=self.default_retrieval_modes(query),
            topk=topk or self.config.default_topk,
        )
        final_hits = list(retrieval["fused_hits"])
        return {
            "user_input": query,
            "retrieved_contexts": [hit.text for hit in final_hits],
            "retrieved_context_ids": [
                hit.chunk.legacy_index if hit.chunk.legacy_index is not None else hit.chunk_id
                for hit in final_hits
            ],
            "scores": [hit.final_score() for hit in final_hits],
            "retrieval_mode": "agentic_retrieve_rrf_rerank",
            "topk": topk or self.config.default_topk,
            "history": history or [],
        }
