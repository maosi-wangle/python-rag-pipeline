# Agentic / Modular / Self-RAG Notes

This repository now contains a modular RAG package under `rag/` with the following layers:

- `knowledge_base.py`
  - Loads future-ready chunk records with `chunk_id`, `document_id`, `title`, `section`, `context`, `keywords`, `source`, and `metadata`.
  - Legacy datasets still load through fallback IDs so current scripts do not break.
- `retrievers/`
  - `semantic.py`: FAISS + sentence-transformers
  - `keyword.py`: current keyword inverted index approach
  - `grep.py`: lexical / exact-match style retrieval
  - `fusion.py`: reciprocal rank fusion
- `tools/`
  - `rewrite.py`: contextual query rewrite variants
  - `decompose.py`: subquery decomposition
  - `rerank.py`: Cohere rerank with fallback lexical rerank
  - `judge.py`: structured completeness / relevance / support judgement
- `orchestrator.py`
  - Plans query variants
  - Runs multi-route retrieval in parallel
  - Runs rerank per query group in parallel
  - Fuses results and produces a grounded answer draft
- `self_rag.py`
  - Runs multi-round retrieval based on `completeness`

## New Chunk Schema

Use this format for future knowledge bases:

```json
{
  "chunk_id": "doc_001_chunk_0001",
  "document_id": "doc_001",
  "title": "Document title",
  "section": "Section heading",
  "context": "Chunk body",
  "keywords": ["kw1", "kw2"],
  "source": "file-or-url",
  "metadata": {
    "lang": "zh"
  }
}
```

## Parallelism

- Retrieval runs in parallel across:
  - semantic
  - keyword
  - grep
  - each rewritten query / subquery
- Rerank runs in parallel per query group before global fusion.

## Structured Output

`SelfRAGPipeline.run(...)` returns:

```json
{
  "response": "...",
  "query": "...",
  "retrieved_chunk_ids": ["..."],
  "completeness": "yes|no",
  "relevance_score": 0.0,
  "support_score": 0.0,
  "need_followup": false,
  "next_query": null,
  "round": 1,
  "traces": []
}
```

## Cohere Rerank

Set:

- `COHERE_API_KEY`
- optional `COHERE_RERANK_MODEL`
- optional `COHERE_RERANK_URL`

If not set, the rerank tool falls back to lexical overlap scoring so the pipeline still runs.
