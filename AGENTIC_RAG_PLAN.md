# Agentic RAG Plan

这份文档只保留当前生效的架构口径。

## 1. Current Canonical Design

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
-> structured result
```

## 2. What Is Agentic

真正 agentic 的部分只有一件事：

- 主 agent 决定下一步做什么

也就是：

- 要不要改写 query
- 要不要拆成 subquery
- 要不要补检索
- 要不要只重生成
- 要不要结束

## 3. What Is Deterministic

下面这些不再让 agent 手工编排细节，而是固定成工具内部工作流：

### retrieve

```text
parallel retrieval
-> wait all
-> merge / dedup
-> RRF
-> final rerank
```

### generate

```text
retrieval_id
-> load evidence
-> grounded generation
```

### finish

```text
normalize final fields
-> emit structured result
-> stop
```

## 4. Self-RAG Interpretation

现在的 self-RAG 不是外层固定 loop 再跑一遍完整流水线，而是：

- 证据不够时再检索
- 证据够但答案不够好时再生成
- 直到 agent 主动 finish

## 5. Key Design Rules

### Rule 1

`finish` 是唯一正常结束信号。

### Rule 2

`if_multi_turn` 现在只是最终诊断字段，不再控制主循环。

### Rule 3

`next_focus` 只是告诉外部“如果继续，最该补什么”，不是强制下一步动作。

### Rule 4

多路检索统一先 RRF，再 final rerank。

### Rule 5

`keyword` 继续沿用现有关键词检索，不强改 BM25。

## 6. Expected Tool Sequences

最常见的几条：

### Sequence A

```text
query_transform -> retrieve -> generate -> finish
```

### Sequence B

```text
retrieve -> generate -> generate -> finish
```

### Sequence C

```text
retrieve -> generate -> retrieve -> generate -> finish
```

### Sequence D

```text
query_transform -> retrieve -> generate -> retrieve -> generate -> finish
```

## 7. Final Output Contract

最终结构化输出至少包含：

- `response`
- `query`
- `grounded`
- `retrieved_chunk_ids`
- `completeness`
- `if_multi_turn`
- `rationale`
- `next_focus`
- `used_queries`
- `tool_rounds`
