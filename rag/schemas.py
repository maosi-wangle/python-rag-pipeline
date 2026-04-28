from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    context: str
    keywords: list[str] = field(default_factory=list)
    document_id: str | None = None
    title: str | None = None
    section: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    legacy_index: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any], fallback_index: int) -> "ChunkRecord":
        metadata = dict(payload.get("metadata") or {})
        known_fields = {
            "chunk_id",
            "context",
            "keywords",
            "document_id",
            "title",
            "section",
            "source",
            "metadata",
        }
        for key, value in payload.items():
            if key not in known_fields:
                metadata.setdefault(key, value)

        chunk_id = str(payload.get("chunk_id") or f"legacy_chunk_{fallback_index:05d}")
        keywords = payload.get("keywords") or []
        if not isinstance(keywords, list):
            keywords = [str(keywords)]

        return cls(
            chunk_id=chunk_id,
            context=str(payload.get("context") or ""),
            keywords=[str(item) for item in keywords],
            document_id=payload.get("document_id"),
            title=payload.get("title"),
            section=payload.get("section"),
            source=payload.get("source"),
            metadata=metadata,
            legacy_index=fallback_index,
        )

    def searchable_text(self) -> str:
        keyword_text = " ".join(self.keywords)
        title_bits = " ".join(
            [item for item in (self.title, self.section, self.document_id) if item]
        )
        return " ".join(part for part in (title_bits, keyword_text, self.context) if part)

    def to_ragas_context(self) -> str:
        return self.searchable_text()


@dataclass(slots=True)
class RetrievalHit:
    chunk: ChunkRecord
    score: float
    retriever: str
    query: str
    rank: int | None = None
    rerank_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def text(self) -> str:
        return self.chunk.to_ragas_context()

    def final_score(self) -> float:
        rerank = self.rerank_score if self.rerank_score is not None else 0.0
        return float(self.score) + float(rerank)

    def clone(
        self,
        *,
        score: float | None = None,
        retriever: str | None = None,
        query: str | None = None,
        rank: int | None = None,
        rerank_score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "RetrievalHit":
        return RetrievalHit(
            chunk=self.chunk,
            score=self.score if score is None else score,
            retriever=self.retriever if retriever is None else retriever,
            query=self.query if query is None else query,
            rank=self.rank if rank is None else rank,
            rerank_score=self.rerank_score if rerank_score is None else rerank_score,
            metadata=dict(self.metadata if metadata is None else metadata),
        )


@dataclass(slots=True)
class RetrievalResult:
    query: str
    retriever: str
    hits: list[RetrievalHit] = field(default_factory=list)


@dataclass(slots=True)
class JudgeResult:
    completeness: str
    relevance_score: float
    support_score: float
    need_followup: bool
    next_query: str | None
    missing_aspects: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass(slots=True)
class StructuredRAGResponse:
    response: str
    query: str
    retrieved_chunk_ids: list[str]
    completeness: str
    relevance_score: float
    support_score: float
    need_followup: bool
    next_query: str | None
    round: int
    traces: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "response": self.response,
            "query": self.query,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "completeness": self.completeness,
            "relevance_score": self.relevance_score,
            "support_score": self.support_score,
            "need_followup": self.need_followup,
            "next_query": self.next_query,
            "round": self.round,
            "traces": self.traces,
        }
