# Agentic Modular RAG

这个项目现在的主架构已经切到“单主 agent + tool-calling loop”模式。

核心思想不是再包一层显式的 `self_rag for` 去驱动每一轮，而是：

1. 主 agent 自己判断要不要先改写 / 拆分 query。
2. `retrieve` 工具固定执行并行多路检索。
3. `retrieve` 内部固定执行 `wait-all -> RRF -> final rerank`。
4. `generate` 工具固定只做基于证据的生成或重生成。
5. 主 agent 在拿到答案后自己决定下一步是：
   - 再检索
   - 重新生成
   - `finish`

## Canonical Flow

```text
user query
-> main agent tool loop
   -> optional query_transform
   -> retrieve
   -> generate
   -> agent self-evaluate
      -> retrieve again
      -> or generate again
      -> or finish
-> structured response
```

## Current Architecture

### 1. Main Agent

- 文件：[rag/agent.py](/c:/project/python-rag-pipeline/rag/agent.py)
- 职责：维护 message history、注册 tool schema、执行 tool loop、在 `finish` 时返回最终结构化结果。

### 2. Deterministic Retrieval / Generation Layer

- 文件：[rag/tools/retrieve.py](/c:/project/python-rag-pipeline/rag/tools/retrieve.py)
- 文件：[rag/tools/answer.py](/c:/project/python-rag-pipeline/rag/tools/answer.py)

`retrieve` 内部流程固定：

```text
multi-route retrieval in parallel
-> wait until all retrieval tasks complete
-> merge / dedup
-> RRF
-> final rerank
-> return retrieval_id + evidence preview
```

`generate` 内部流程固定：

```text
query + retrieval_id + optional instruction
-> load evidence from retrieval memory
-> grounded answer generation
-> return answer draft
```

### 3. Query Transformation Layer

- 文件：[rag/tools/rewrite.py](/c:/project/python-rag-pipeline/rag/tools/rewrite.py)

支持：

- contextualize
- specific rewrite
- general rewrite
- chunk-like rewrite
- decomposition

输出统一为：

```json
{
  "queries": ["..."],
  "plans": [
    {
      "query": "...",
      "retrieval_modes": ["semantic", "keyword", "grep"]
    }
  ]
}
```

### 4. Finish Layer

- 文件：[rag/tools/finish.py](/c:/project/python-rag-pipeline/rag/tools/finish.py)

`finish` 是唯一真正的结束信号。

它接收：

- `response`
- `query`
- `grounded`
- `completeness`
- `if_multi_turn`
- `rationale`
- `next_focus`
- `retrieval_id`

注意：`if_multi_turn` 现在只是最终诊断字段，不再驱动主循环控制。

### 5. Component Container

- 文件：[rag/orchestrator.py](/c:/project/python-rag-pipeline/rag/orchestrator.py)

现在的 `orchestrator` 不再负责旧式多轮控制，而是负责组装：

- knowledge base
- retrievers
- reranker
- query transform tool
- retrieve tool
- generate tool
- finish tool

## Retrieval Strategy

当前默认支持三路：

- `semantic`
- `keyword`
- `grep`

说明：

- `keyword` 沿用现有关键词检索，不额外改成 BM25。
- 多路检索是并行发起的。
- 最终不是“先重排再 RRF”，而是“先 RRF，再 final rerank”。

## Structured Output

`run_agentic_query()` 当前返回：

```json
{
  "response": "最终回答",
  "query": "最终回答对应的问题",
  "grounded": true,
  "retrieved_chunk_ids": ["chunk_001", "chunk_019"],
  "completeness": "yes",
  "rationale": "为什么现在可以结束",
  "next_focus": null,
  "if_multi_turn": false,
  "tool_rounds": 3,
  "used_queries": ["原始 query", "补充 query"],
  "traces": []
}
```

字段含义：

- `grounded`：答案是否被召回证据支撑。
- `completeness`：答案是否已经覆盖用户问题。
- `if_multi_turn`：如果此时被迫停止，是否仍然建议继续一轮。
- `rationale`：为什么结束。
- `next_focus`：若还值得继续，下一轮最应该补什么。

## Entry Points

- [faceaiRAG.py](/c:/project/python-rag-pipeline/faceaiRAG.py)
  - 兼容入口
- [agentic_rag_cli.py](/c:/project/python-rag-pipeline/agentic_rag_cli.py)
  - CLI 入口

## Run

### 1. Install

```powershell
python -m pip install -r requirements.txt
```

### 2. Set Env

LLM:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- optional `OPENAI_BASE_URL`

Rerank:

- `COHERE_API_KEY`
- optional `COHERE_RERANK_MODEL`
- optional `COHERE_RERANK_URL`

### 3. CLI

```powershell
python agentic_rag_cli.py --query "防晒需要注意什么" --topk 5 --max-rounds 4 --print-traces
```

### 4. Interactive

```powershell
python agentic_rag_cli.py --interactive --topk 5 --max-rounds 4 --print-traces
```

## Knowledge Format

后续知识库建议统一使用：

```json
{
  "chunk_id": "doc_001_chunk_0001",
  "document_id": "doc_001",
  "title": "文档标题",
  "section": "章节名",
  "context": "chunk 正文",
  "keywords": ["关键词1", "关键词2"],
  "source": "source path or url",
  "metadata": {
    "lang": "zh"
  }
}
```

旧格式仍然兼容读取，但已经不再是推荐主格式。

## Detailed Explanation

更详细的架构、模块逻辑、代码逻辑说明见：

- [PHASE0_BASELINE.md](/c:/project/python-rag-pipeline/PHASE0_BASELINE.md)
- [PROJECT_EXPLANATION.md](/c:/project/python-rag-pipeline/PROJECT_EXPLANATION.md)
- [AGENTIC_RAG_PLAN.md](/c:/project/python-rag-pipeline/AGENTIC_RAG_PLAN.md)
- [PLATFORMIZATION_PLAN.md](/c:/project/python-rag-pipeline/PLATFORMIZATION_PLAN.md)
