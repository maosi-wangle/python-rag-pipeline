from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schemas import RetrievalHit, RetrievalResult


@dataclass(slots=True)
class RetrievalRoundState:
    round_index: int
    input_query: str
    active_query: str
    query_variants: list[str] = field(default_factory=list)
    subqueries: list[str] = field(default_factory=list)
    retrieval_results: list[RetrievalResult] = field(default_factory=list)
    query_level_hits: dict[str, list[RetrievalHit]] = field(default_factory=dict)
    fused_hits: list[RetrievalHit] = field(default_factory=list)
    answer_draft: str = ""
    completeness: str = "no"
    relevance_score: float = 0.0
    support_score: float = 0.0
    need_followup: bool = False
    next_query: str | None = None
    missing_aspects: list[str] = field(default_factory=list)
    judge_reason: str = ""

    def to_trace(self) -> dict[str, Any]:
        return {
            "round": self.round_index,
            "input_query": self.input_query,
            "active_query": self.active_query,
            "query_variants": self.query_variants,
            "subqueries": self.subqueries,
            "retrievers": [
                {
                    "retriever": result.retriever,
                    "query": result.query,
                    "chunk_ids": [hit.chunk_id for hit in result.hits],
                }
                for result in self.retrieval_results
            ],
            "final_chunk_ids": [hit.chunk_id for hit in self.fused_hits],
            "completeness": self.completeness,
            "relevance_score": self.relevance_score,
            "support_score": self.support_score,
            "need_followup": self.need_followup,
            "next_query": self.next_query,
            "missing_aspects": self.missing_aspects,
            "judge_reason": self.judge_reason,
        }


@dataclass(slots=True)
class RAGSessionState:
    user_query: str
    history: list[str] = field(default_factory=list)
    rounds: list[RetrievalRoundState] = field(default_factory=list)
    max_rounds: int = 3
