from __future__ import annotations

import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import RAGConfig
from ..knowledge_base import ChunkStore
from ..schemas import RetrievalHit, RetrievalResult
from .base import BaseRetriever


class SemanticRetriever(BaseRetriever):
    name = "semantic"

    def __init__(self, chunk_store: ChunkStore, config: RAGConfig):
        self.chunk_store = chunk_store
        self.config = config
        self.model: SentenceTransformer | None = None
        self.index: faiss.Index | None = None
        self.embeddings: np.ndarray | None = None
        self.initialized = self._initialize()

    def _initialize(self) -> bool:
        try:
            self.model = SentenceTransformer(self.config.embedding_model_name)
        except Exception:
            return False

        loaded = self._try_load_index()
        if loaded:
            return True

        return self._build_and_save_index()

    def _try_load_index(self) -> bool:
        if not (
            os.path.exists(self.config.index_path)
            and os.path.exists(self.config.embeddings_path)
        ):
            return False

        try:
            self.index = faiss.read_index(self.config.index_path)
            self.embeddings = np.load(self.config.embeddings_path)
        except Exception:
            self.index = None
            self.embeddings = None
            return False

        if self.index.ntotal != len(self.chunk_store):
            self.index = None
            self.embeddings = None
            return False

        if self.embeddings.shape[0] != len(self.chunk_store):
            self.index = None
            self.embeddings = None
            return False

        return True

    def _build_and_save_index(self) -> bool:
        if self.model is None:
            return False

        texts = [chunk.searchable_text() for chunk in self.chunk_store.chunks]
        try:
            self.embeddings = self.model.encode(
                texts,
                batch_size=64,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.index.add(self.embeddings.astype(np.float32))
            faiss.write_index(self.index, self.config.index_path)
            np.save(self.config.embeddings_path, self.embeddings)
            return True
        except Exception:
            self.index = None
            self.embeddings = None
            return False

    def retrieve(self, query: str, topk: int) -> RetrievalResult:
        if not self.initialized or self.model is None or self.index is None:
            return RetrievalResult(query=query, retriever=self.name, hits=[])

        q_vec = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.index.search(q_vec, topk)

        hits: list[RetrievalHit] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:
                continue
            chunk = self.chunk_store.chunks[int(idx)]
            hits.append(
                RetrievalHit(
                    chunk=chunk,
                    score=float(score),
                    retriever=self.name,
                    query=query,
                    rank=rank,
                    metadata={"legacy_index": chunk.legacy_index},
                )
            )
        return RetrievalResult(query=query, retriever=self.name, hits=hits)
