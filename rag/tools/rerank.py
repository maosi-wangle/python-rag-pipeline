from __future__ import annotations

import json
from urllib import request

from ..config import RAGConfig
from ..schemas import RetrievalHit
from ..text_utils import lexical_overlap_score


class CohereRerankTool:
    name = "cohere_rerank"

    def __init__(self, config: RAGConfig):
        self.config = config

    def rerank(self, query: str, hits: list[RetrievalHit], topn: int | None = None) -> list[RetrievalHit]:
        if not hits:
            return []

        if not self.config.cohere_api_key:
            return self._fallback_rerank(query, hits, topn)

        try:
            return self._cohere_rerank(query, hits, topn)
        except Exception:
            return self._fallback_rerank(query, hits, topn)

    def _cohere_rerank(
        self,
        query: str,
        hits: list[RetrievalHit],
        topn: int | None,
    ) -> list[RetrievalHit]:
        documents = [hit.text for hit in hits]
        payload = json.dumps(
            {
                "model": self.config.cohere_model,
                "query": query,
                "documents": documents,
                "top_n": topn or len(hits),
            }
        ).encode("utf-8")

        req = request.Request(
            self.config.cohere_api_base,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.config.cohere_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=30) as response:
            response_payload = json.loads(response.read().decode("utf-8"))

        reranked_hits: list[RetrievalHit] = []
        for rank, item in enumerate(response_payload.get("results", []), start=1):
            idx = int(item["index"])
            base_hit = hits[idx]
            rerank_score = float(item.get("relevance_score", 0.0))
            reranked_hits.append(
                base_hit.clone(
                    retriever=f"{base_hit.retriever}+rerank",
                    rank=rank,
                    rerank_score=rerank_score,
                )
            )

        return reranked_hits[: topn or len(reranked_hits)]

    def _fallback_rerank(
        self,
        query: str,
        hits: list[RetrievalHit],
        topn: int | None,
    ) -> list[RetrievalHit]:
        rescored_hits: list[RetrievalHit] = []
        for hit in hits:
            lexical_score = lexical_overlap_score(query, hit.text)
            rescored_hits.append(
                hit.clone(
                    rerank_score=lexical_score,
                    retriever=f"{hit.retriever}+fallback_rerank",
                )
            )

        rescored_hits.sort(key=lambda item: item.final_score(), reverse=True)
        limit = topn or len(rescored_hits)
        return [hit.clone(rank=rank) for rank, hit in enumerate(rescored_hits[:limit], start=1)]
