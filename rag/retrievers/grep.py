from __future__ import annotations

import re

from ..knowledge_base import ChunkStore
from ..schemas import RetrievalHit, RetrievalResult
from ..text_utils import normalize_whitespace, tokenize
from .base import BaseRetriever


class GrepRetriever(BaseRetriever):
    name = "grep"

    def __init__(self, chunk_store: ChunkStore):
        self.chunk_store = chunk_store

    def retrieve(self, query: str, topk: int) -> RetrievalResult:
        normalized_query = normalize_whitespace(query)
        query_tokens = tokenize(query)
        if not normalized_query:
            return RetrievalResult(query=query, retriever=self.name, hits=[])

        scored_hits: list[tuple[float, RetrievalHit]] = []
        for chunk in self.chunk_store.chunks:
            score = self._score_chunk(normalized_query, query_tokens, chunk.searchable_text())
            if score <= 0:
                continue
            scored_hits.append(
                (
                    score,
                    RetrievalHit(
                        chunk=chunk,
                        score=score,
                        retriever=self.name,
                        query=query,
                    ),
                )
            )

        scored_hits.sort(key=lambda item: item[0], reverse=True)
        hits: list[RetrievalHit] = []
        for rank, (_, hit) in enumerate(scored_hits[:topk], start=1):
            hits.append(hit.clone(rank=rank))
        return RetrievalResult(query=query, retriever=self.name, hits=hits)

    def _score_chunk(self, query: str, query_tokens: list[str], text: str) -> float:
        normalized_text = normalize_whitespace(text).lower()
        normalized_query = query.lower()
        score = 0.0

        if normalized_query and normalized_query in normalized_text:
            score += 5.0

        for token in query_tokens:
            if token in normalized_text:
                score += 1.0 + min(len(re.findall(re.escape(token), normalized_text)), 3) * 0.25

        return score
