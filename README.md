# Agentic Modular Self-RAG

一个面向中文知识库的 Python RAG 项目，现已支持：

- modular RAG 组件化检索
- agentic query planning
- self-RAG 多轮检索闭环
- 并行多路检索：`semantic + keyword + grep`
- 并行 rerank：`Cohere rerank`，无 key 时自动 fallback
- 结构化输出：`response / query / retrieved_chunk_ids / completeness / next_query / traces`
- RAGAS 检索评测兼容

## 目录

- [faceaiRAG.py](/c:/project/python-rag-pipeline/faceaiRAG.py)
  - 兼容入口，保留 `FaceAiSystem`
- [agentic_rag_cli.py](/c:/project/python-rag-pipeline/agentic_rag_cli.py)
  - 命令行入口
- [rag](/c:/project/python-rag-pipeline/rag)
  - 新的 modular / agentic / self-RAG 核心包
- [markdown_chunk_processor.py](/c:/project/python-rag-pipeline/markdown_chunk_processor.py)
  - 新知识库构建脚本
- [eval_ragas.py](/c:/project/python-rag-pipeline/eval_ragas.py)
  - RAGAS 检索评测脚本
- [AGENTIC_RAG_PLAN.md](/c:/project/python-rag-pipeline/AGENTIC_RAG_PLAN.md)
  - 设计说明

## 检索架构

### 1. Modular Retrieval

- `SemanticRetriever`
  - 使用 `sentence-transformers + faiss`
- `KeywordRetriever`
  - 保留现有关键词倒排检索思路
- `GrepRetriever`
  - 做短语/实体/精确文本补充召回
- `Fusion`
  - 使用 RRF 合并结果

### 2. Agentic Planning

- `QueryPlannerTool`
  - 决定是否改写 query
  - 决定是否拆 subquery
  - 决定要调用哪些 retriever
- `QueryRewriteTool`
  - 做 chunk-like / contextual rewrite
- `QueryDecomposeTool`
  - 做多意图问题拆解

### 3. Self-RAG Loop

流程如下：

1. 基于上下文记忆规划 query
2. 并行执行多路检索
3. 按 query group 并行 rerank
4. 生成 grounded answer
5. 自评 `relevance / support / completeness`
6. 若 `completeness = no`，生成下一轮 query 继续检索

## 新知识库格式

后续知识库请统一使用以下 schema：

```json
{
  "chunk_id": "doc_001_chunk_0001",
  "document_id": "doc_001",
  "title": "文档标题",
  "section": "章节名",
  "context": "chunk 正文",
  "keywords": ["关键词1", "关键词2"],
  "source": "来源路径或 URL",
  "metadata": {
    "lang": "zh"
  }
}
```

旧 `knowledgeBase.json` 仍可兼容读取，但建议只作为过渡数据。

## 安装

```powershell
python -m pip install -r requirements.txt
```

## 环境变量

### OpenAI-compatible LLM

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- 可选 `OPENAI_BASE_URL`
- 可选 `OPENAI_TIMEOUT`
- 可选 `OPENAI_TEMPERATURE`

用途：

- query planning
- grounded answer generation
- self-judge

未配置时，系统会回退到启发式 planner / generator / judge。

### Cohere Rerank

- `COHERE_API_KEY`
- 可选 `COHERE_RERANK_MODEL`
- 可选 `COHERE_RERANK_URL`

未配置时，系统会回退到 lexical overlap rerank。

## 运行

### 1. 交互式 Agentic RAG

```powershell
python agentic_rag_cli.py --interactive --topk 5 --max-rounds 3 --print-traces
```

### 2. 单次查询

```powershell
python agentic_rag_cli.py --query "防晒需要注意什么" --topk 5 --max-rounds 3 --print-traces
```

### 3. 兼容旧入口

```powershell
python faceaiRAG.py
```

### 4. 构建新知识库

```powershell
python markdown_chunk_processor.py docs/example.md --document-id doc_001 --output knowledgeBase.new.json
```

### 5. RAGAS 检索评测

```powershell
python eval_ragas.py --dataset ragas_eval_dataset.formal.json --topk 5 --output-dir ragas_outputs_formal
```

## 结构化输出

`run_agentic_query()` 输出格式：

```json
{
  "response": "最终回答",
  "query": "本轮实际检索 query",
  "retrieved_chunk_ids": ["doc_001_chunk_0001", "doc_001_chunk_0007"],
  "completeness": "yes",
  "relevance_score": 0.88,
  "support_score": 0.81,
  "need_followup": false,
  "next_query": null,
  "round": 1,
  "traces": []
}
```

## 当前状态

这版项目已经具备完整的 agentic / modular / self-RAG 基础链路

- 如果没有配置 LLM 和 Cohere key，系统仍然可跑，但 planning / answer / judge / rerank 会使用 fallback 逻辑，效果会弱于完整线上配置。
- 回答生成目前是“grounded first”的工程取向，优先保证可追踪和可扩展
