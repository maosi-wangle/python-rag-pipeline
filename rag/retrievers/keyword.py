from __future__ import annotations

import json
import os

from ..config import RAGConfig
from ..knowledge_base import ChunkStore
from ..schemas import RetrievalHit, RetrievalResult
from ..text_utils import tokenize
from .base import BaseRetriever


class KeywordRetriever(BaseRetriever):
    name = "keyword"

    def __init__(self, chunk_store: ChunkStore, config: RAGConfig):
        self.chunk_store = chunk_store
        self.config = config
        self.inverted_index: dict[str, list[str]] = {}
        self._load_or_build_index()

    def _load_or_build_index(self) -> None:
        if not os.path.exists(self.config.inverted_index_path):
            self._build_and_save_index()
            return

        try:
            with open(self.config.inverted_index_path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
            self.inverted_index = self._normalize_loaded_index(raw)
        except Exception:
            self._build_and_save_index()

    def _normalize_loaded_index(self, raw: dict[str, list[str]]) -> dict[str, list[str]]:
        normalized: dict[str, list[str]] = {}
        for token, values in raw.items():
            chunk_ids: list[str] = []
            for value in values:
                if isinstance(value, str):
                    chunk_ids.append(value)
                    continue
                if isinstance(value, int):
                    chunk = self.chunk_store.by_legacy_index.get(value)
                    if chunk is not None:
                        chunk_ids.append(chunk.chunk_id)
            if chunk_ids:
                normalized[token] = chunk_ids
        if normalized:
            return normalized
        self._build_and_save_index()
        return self.inverted_index

    def _build_and_save_index(self) -> None:
        inverted_index: dict[str, list[str]] = {}
        for chunk in self.chunk_store.chunks:
            for token in set(tokenize(chunk.searchable_text())):
                inverted_index.setdefault(token, []).append(chunk.chunk_id)
        self.inverted_index = inverted_index
        with open(self.config.inverted_index_path, "w", encoding="utf-8") as handle:
            json.dump(self.inverted_index, handle, ensure_ascii=False)

    def retrieve(self, query: str, topk: int) -> RetrievalResult:
        query_tokens = tokenize(query)
        if not query_tokens:
            return RetrievalResult(query=query, retriever=self.name, hits=[])

        scores: dict[str, float] = {}
        for token in query_tokens:
            for chunk_id in self.inverted_index.get(token, []):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:topk]
        hits: list[RetrievalHit] = []
        for rank, (chunk_id, score) in enumerate(ranked, start=1):
            chunk = self.chunk_store.by_chunk_id.get(chunk_id)
            if chunk is None:
                continue
            hits.append(
                RetrievalHit(
                    chunk=chunk,
                    score=float(score),
                    retriever=self.name,
                    query=query,
                    rank=rank,
                )
            )
        return RetrievalResult(query=query, retriever=self.name, hits=hits)
