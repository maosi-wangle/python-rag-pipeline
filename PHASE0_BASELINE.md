# Phase 0 Baseline

Phase 0 的目标是把当前 agentic RAG 内核收口成稳定 baseline，为后续平台化做地基。

## Active Runtime Path

当前唯一主链路是：

```text
FaceAiSystem / agentic_rag_cli
-> ToolCallingRAGAgent
-> query_transform / retrieve / generate / finish
-> StructuredRAGResponse
```

对应文件：

- [faceaiRAG.py](/c:/project/python-rag-pipeline/faceaiRAG.py)
- [agentic_rag_cli.py](/c:/project/python-rag-pipeline/agentic_rag_cli.py)
- [rag/agent.py](/c:/project/python-rag-pipeline/rag/agent.py)
- [rag/orchestrator.py](/c:/project/python-rag-pipeline/rag/orchestrator.py)
- [rag/tools/rewrite.py](/c:/project/python-rag-pipeline/rag/tools/rewrite.py)
- [rag/tools/retrieve.py](/c:/project/python-rag-pipeline/rag/tools/retrieve.py)
- [rag/tools/answer.py](/c:/project/python-rag-pipeline/rag/tools/answer.py)
- [rag/tools/finish.py](/c:/project/python-rag-pipeline/rag/tools/finish.py)

## Tool Contract

### query_transform

Input:

- `query`
- `rewrite_mode`
- `should_decompose`

Output:

- `queries`
- `plans`
- `transform_type`
- `rationale`

### retrieve

Input:

- `query`
- optional `plans`
- optional `retrieval_modes`
- optional `topk`

Output to agent:

- `retrieval_id`
- `used_queries`
- `retrieved_chunk_ids`
- chunk previews
- retriever run summaries

Internal fixed flow:

```text
parallel retrieval
-> wait all
-> RRF
-> final rerank
```

### generate

Input:

- `query`
- `retrieval_id`
- optional `instruction`

Output:

- answer draft
- `retrieval_id`
- `retrieved_chunk_ids`

### finish

Input:

- `response`
- `query`
- `grounded`
- `completeness`
- `if_multi_turn`
- `rationale`
- optional `next_focus`
- optional `retrieval_id`

Output:

- final structured response

## Removed From Active Path

These old single-round/outer-loop modules were removed from the active code tree:

- `rag/tools/retrieve_generate.py`
- `rag/tools/planner.py`
- `rag/tools/decompose.py`
- `rag/state.py`

`rag/tools/judge.py` remains as a fallback evaluator for:

- no tool-calling LLM configured
- auto-finish when the agent does not call `finish`
- max-rounds auto-finish

It does not drive the main self-RAG loop.

## Current Boundaries

Phase 0 intentionally does not implement:

- parser layer
- multi-knowledge-base registry
- user model registry
- long-term memory
- system-level tracing
- service/API layer

Those belong to the next platformization phases.

## Verification

Compile:

```powershell
python -m compileall rag faceaiRAG.py agentic_rag_cli.py phase0_smoke.py
```

Smoke:

```powershell
python phase0_smoke.py --query "防晒需要注意什么" --topk 2 --max-rounds 2
```

Expected:

- command exits with code 0
- JSON contains `response`
- JSON contains `retrieved_chunk_ids`
- `completeness` is `yes` or `no`
- `tool_rounds` is positive
