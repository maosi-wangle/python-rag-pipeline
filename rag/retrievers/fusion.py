from __future__ import annotations

from ..schemas import RetrievalHit, RetrievalResult


def reciprocal_rank_fusion(
    results: list[RetrievalResult],
    *,
    topk: int,
    k: int = 60,
) -> list[RetrievalHit]:
    aggregated: dict[str, RetrievalHit] = {}
    details: dict[str, list[dict[str, object]]] = {}

    for result in results:
        for rank, hit in enumerate(result.hits, start=1):
            fused_score = 1.0 / (k + rank)
            if hit.chunk_id not in aggregated:
                aggregated[hit.chunk_id] = hit.clone(
                    score=fused_score,
                    retriever="fusion",
                    query=result.query,
                    rank=rank,
                    metadata={
                        "retrievers": [hit.retriever],
                        "queries": [result.query],
                    },
                )
                details[hit.chunk_id] = []
            else:
                current = aggregated[hit.chunk_id]
                retrievers = list(current.metadata.get("retrievers", []))
                queries = list(current.metadata.get("queries", []))
                if hit.retriever not in retrievers:
                    retrievers.append(hit.retriever)
                if result.query not in queries:
                    queries.append(result.query)
                current.metadata["retrievers"] = retrievers
                current.metadata["queries"] = queries
                current.score += fused_score
                current.rerank_score = max(
                    current.rerank_score or 0.0,
                    hit.rerank_score or 0.0,
                )

            details[hit.chunk_id].append(
                {
                    "retriever": hit.retriever,
                    "query": result.query,
                    "rank": rank,
                    "score": hit.score,
                    "rerank_score": hit.rerank_score,
                }
            )

    fused_hits = list(aggregated.values())
    for fused_hit in fused_hits:
        fused_hit.metadata["contributors"] = details.get(fused_hit.chunk_id, [])

    fused_hits.sort(key=lambda hit: (hit.final_score(), hit.score), reverse=True)
    return fused_hits[:topk]
