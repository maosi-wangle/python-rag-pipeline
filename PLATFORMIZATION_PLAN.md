# Agentic RAG 平台化规划方案

本文档目标：在当前项目已有的 agentic RAG 能力上，参考 `ragflow/` 的成熟平台分层，把系统从“静态查一个知识库的 RAG agent”演进为“可注册用户、可切换知识库、可接入解析层、可配置模型、可追踪调用链”的轻量平台。

## 1. 当前项目内容

当前项目已经具备一条可用的 agentic RAG 主链路。

现有核心能力：

- 单主 agent tool-calling loop：[rag/agent.py](/c:/project/python-rag-pipeline/rag/agent.py)
- query transformation tool：[rag/tools/rewrite.py](/c:/project/python-rag-pipeline/rag/tools/rewrite.py)
- 并行多路检索 + RRF + rerank：[rag/tools/retrieve.py](/c:/project/python-rag-pipeline/rag/tools/retrieve.py)
- grounded generation：[rag/tools/answer.py](/c:/project/python-rag-pipeline/rag/tools/answer.py)
- finish 结构化收口：[rag/tools/finish.py](/c:/project/python-rag-pipeline/rag/tools/finish.py)
- 静态 knowledgeBase.json 加载：[rag/knowledge_base.py](/c:/project/python-rag-pipeline/rag/knowledge_base.py)
- 兼容入口：[faceaiRAG.py](/c:/project/python-rag-pipeline/faceaiRAG.py)

当前主要限制：

- 只有一个静态知识库文件。
- 知识库不是对象，检索方式、embedding 模型、parser 配置没有归属于知识库。
- 用户和模型配置主要依赖环境变量，没有用户级注册。
- 没有正式 document ingest / parser / indexing pipeline。
- 对话历史只是传入参数，没有长短期记忆层。
- tracing 只在返回结果里有简单 traces，没有系统级 trace/span/event。
- 没有平台 API/service/repository 分层。

## 2. RAGFlow 可借鉴的设计

`ragflow/` 很大，不建议照搬。但它有几个对象边界非常值得借鉴。

## 2.1 Knowledgebase

RAGFlow 中 `Knowledgebase` 是一等对象。

它包含：

- `id`
- `tenant_id`
- `name`
- `description`
- `embd_id`
- `parser_id`
- `parser_config`
- `doc_num`
- `chunk_num`
- `similarity_threshold`
- `vector_similarity_weight`
- `permission`

对当前项目的启发：

知识库不应该只是 `knowledgeBase.json`，而应该是一个具备配置和统计信息的对象。

## 2.2 Document / File / Task

RAGFlow 把文件、文档、解析任务拆开。

可借鉴对象：

- `File`：原始上传文件。
- `Document`：属于某个 knowledge base 的可解析文档。
- `Task`：解析、切分、embedding、索引等异步任务。

对当前项目的启发：

parser 层应该接入 ingest pipeline，而不是直接把文件变成 chunk。

## 2.3 TenantLLM / LLMFactories

RAGFlow 把模型供应商和租户模型配置分开。

可借鉴对象：

- `LLMFactory`：OpenAI、DeepSeek、Moonshot、Ollama 等 provider。
- `TenantLLM`：某个用户/租户注册的模型实例，包含 `api_key`、`api_base`、`model_type`、`llm_name`。

对当前项目的启发：

LLM 不应该只来自环境变量。用户应该能注册自己的 chat model、embedding model、rerank model、ocr model。

## 2.4 Dialog / Conversation

RAGFlow 把应用配置和会话分开。

可借鉴对象：

- `Dialog`：一个 RAG 应用配置，绑定 LLM、prompt、kb_ids、top_n、rerank 等。
- `Conversation`：一次用户会话，保存 message 和 reference。

对当前项目的启发：

当前的 agent 应该被包装成一个 `RAGApplication` 或 `DialogAgent`，对话历史应该持久化。

## 2.5 Memory

RAGFlow 有独立 memory 模块。

可借鉴点：

- memory 有类型。
- memory 有 embedding 模型。
- memory 有忘记策略。
- memory 可以独立查询。

对当前项目的启发：

需要拆出短期记忆和长期记忆：

- 短期记忆：当前 conversation 的 message window 和 summary。
- 长期记忆：用户画像、偏好、长期事实、历史任务摘要，可检索。

## 2.6 Langfuse / Tracing

RAGFlow 支持 tenant 级 Langfuse 配置。

对当前项目的启发：

先做本地 tracing 抽象，再预留 Langfuse / OpenTelemetry adapter。

## 3. 平台化后的目标对象模型

建议把当前系统抽象成以下对象。

## 3.1 UserProfile

表示用户。

字段建议：

- `user_id`
- `display_name`
- `default_llm_profile_id`
- `default_embedding_profile_id`
- `default_rerank_profile_id`
- `permissions`
- `created_at`
- `updated_at`

职责：

- 管理用户身份。
- 关联用户自己的模型配置。
- 关联用户可访问的知识库。

## 3.2 LLMProfile

表示用户注册的模型配置。

字段建议：

- `profile_id`
- `user_id`
- `provider`
- `model_type`
- `model_name`
- `api_key`
- `base_url`
- `timeout`
- `temperature`
- `max_tokens`
- `extra`

`model_type` 建议枚举：

- `chat`
- `embedding`
- `rerank`
- `vision`
- `ocr`

职责：

- 替代当前环境变量式模型配置。
- 给 agent、parser、embedding、rerank 提供可切换模型。

## 3.3 KnowledgeBaseProfile

表示一个知识库对象，例如“皮肤知识库”。

字段建议：

- `kb_id`
- `owner_user_id`
- `name`
- `description`
- `language`
- `embedding_profile_id`
- `rerank_profile_id`
- `retrieval_modes`
- `parser_id`
- `parser_config`
- `chunking_config`
- `index_config`
- `storage_config`
- `doc_count`
- `chunk_count`
- `status`

职责：

- 管理知识库配置。
- 决定使用什么 embedding 模型。
- 决定支持哪些检索方式。
- 决定 parser 和 chunking 策略。
- 支持知识库切换。

示例：

```json
{
  "kb_id": "skin_kb",
  "name": "皮肤知识库",
  "embedding_profile_id": "emb_text2vec_zh",
  "rerank_profile_id": "cohere_rerank",
  "retrieval_modes": ["semantic", "keyword", "grep"],
  "parser_id": "pdf",
  "parser_config": {
    "ocr": true,
    "page_ranges": [[1, 1000000]]
  },
  "chunking_config": {
    "chunk_size": 800,
    "overlap": 120
  },
  "index_config": {
    "vector_index": "faiss",
    "metric": "ip"
  }
}
```

## 3.4 DocumentAsset

表示进入平台的原始文档。

字段建议：

- `doc_id`
- `kb_id`
- `file_name`
- `file_type`
- `source_path`
- `content_hash`
- `parser_id`
- `parser_config`
- `status`
- `progress`
- `chunk_count`
- `error`

职责：

- 记录 PDF、图片、文本的解析状态。
- 作为 parser pipeline 的输入。

## 3.5 ParsedDocument

表示 parser 输出的标准中间结构。

字段建议：

- `doc_id`
- `elements`
- `metadata`

`elements` 建议统一格式：

```json
{
  "element_id": "doc_001_p1_e03",
  "type": "text|image|table|ocr_text",
  "page": 1,
  "content": "...",
  "bbox": [0, 0, 100, 100],
  "metadata": {}
}
```

职责：

- 屏蔽 PDF、图片、文本解析差异。
- 给 chunker 提供统一输入。

## 3.6 ChunkRecord

当前已有 `ChunkRecord`，建议扩展为平台级 chunk。

字段建议：

- `chunk_id`
- `kb_id`
- `doc_id`
- `content`
- `keywords`
- `title`
- `section`
- `page`
- `source`
- `embedding_model`
- `metadata`

职责：

- 支持多知识库隔离。
- 支持按 doc/source/page 追踪引用。

## 3.7 ConversationSession

表示一次对话。

字段建议：

- `conversation_id`
- `user_id`
- `active_kb_ids`
- `llm_profile_id`
- `messages`
- `short_memory`
- `summary`
- `created_at`
- `updated_at`

职责：

- 保存短期对话历史。
- 保存本轮 RAG references。
- 控制 context window。

## 3.8 MemoryStore

表示长期记忆库。

字段建议：

- `memory_id`
- `user_id`
- `memory_type`
- `content`
- `embedding`
- `importance`
- `last_accessed_at`
- `expires_at`
- `metadata`

`memory_type` 建议：

- `semantic`：长期事实，例如用户偏好。
- `episodic`：历史交互摘要。
- `procedural`：用户习惯、工作流偏好。

职责：

- 在 query_transform 或 agent planning 前召回用户相关长期记忆。
- 在回答结束后异步沉淀重要信息。

## 3.9 TraceRecord

表示一次端到端调用链。

字段建议：

- `trace_id`
- `user_id`
- `conversation_id`
- `kb_ids`
- `input`
- `output`
- `started_at`
- `ended_at`
- `status`

## 3.10 TraceSpan

表示 trace 内的一个阶段。

字段建议：

- `span_id`
- `trace_id`
- `parent_span_id`
- `name`
- `type`
- `input`
- `output`
- `latency_ms`
- `error`
- `metadata`

span 类型建议：

- `agent_loop`
- `query_transform`
- `retrieve`
- `semantic_retrieve`
- `keyword_retrieve`
- `grep_retrieve`
- `rerank`
- `generate`
- `finish`
- `parse`
- `chunk`
- `embed`
- `index`

## 4. 目标系统分层

建议从当前 `rag/` 单包，演进为轻量平台结构。

```text
rag/
  platform/
    schemas.py
    registry.py
    repositories.py
    services.py
  parsers/
    base.py
    router.py
    pdf_parser.py
    image_parser.py
    text_parser.py
  ingestion/
    pipeline.py
    chunker.py
    indexer.py
  memory/
    short_term.py
    long_term.py
    summarizer.py
  models/
    profiles.py
    factory.py
  tracing/
    tracer.py
    exporters.py
  agent.py
  orchestrator.py
  tools/
  retrievers/
```

## 5. Parser 层规划

## 5.1 Parser 抽象

新增 `DocumentParser` 基类：

```python
class DocumentParser:
    parser_id: str
    supported_types: set[str]

    def parse(self, asset: DocumentAsset, config: dict) -> ParsedDocument:
        raise NotImplementedError
```

## 5.2 ParserRouter

根据文件类型和知识库 parser 配置路由：

```text
DocumentAsset
-> ParserRouter
   -> PdfParser
   -> ImageParser
   -> TextParser
-> ParsedDocument
```

## 5.3 暂定 parser 实现

### PdfParser

优先级：

1. 文本型 PDF：直接提取文本。
2. 扫描型 PDF：走 OCR。
3. 保留 page/source metadata。

可选依赖：

- `pypdf`
- `pymupdf`
- 后续可接 PaddleOCR / MinerU。

### ImageParser

优先级：

1. OCR 提取文本。
2. 可选 vision model 生成图片描述。
3. 保留图片路径和 OCR 置信度。

### TextParser

支持：

- `.txt`
- `.md`
- 纯文本输入

## 6. 知识库切换方案

当前检索器默认读取一个 `ChunkStore`。

平台化后改成：

```text
kb_id
-> KnowledgeBaseRegistry.get(kb_id)
-> build ChunkStore / IndexStore / RetrieverSet
-> retrieve
```

推荐新增：

- `KnowledgeBaseRegistry`：管理可用知识库对象。
- `KnowledgeBaseRuntime`：某个知识库运行态，持有 chunk store、index、retrievers。
- `RetrieverFactory`：根据 `KnowledgeBaseProfile.retrieval_modes` 构建检索器。

agent 输入应支持：

```python
agent.run(
    query,
    user_id="u_001",
    conversation_id="c_001",
    kb_ids=["skin_kb"]
)
```

多知识库检索时：

```text
kb_ids × query_plans × retrieval_modes
```

所有结果统一进入：

```text
RRF -> rerank -> evidence set
```

## 7. LLM 用户对象化方案

当前 `RAGConfig` 从环境变量读取模型信息。

平台化后保留环境变量作为默认配置，但新增用户级 `LLMProfile`。

调用路径：

```text
user_id
-> LLMProfileRegistry
-> LLMProfile
-> LLMClientFactory
-> Chat / Embedding / Rerank / OCR client
```

建议分四类 profile：

- `chat_profile`
- `embedding_profile`
- `rerank_profile`
- `ocr_or_vision_profile`

知识库可以绑定 embedding/rerank profile。

对话可以绑定 chat profile。

parser 可以绑定 OCR/vision profile。

## 8. 记忆层规划

## 8.1 短期记忆

短期记忆属于 `ConversationSession`。

职责：

- 保存最近 N 轮 messages。
- 超过窗口后压缩成 summary。
- query_transform 时提供上下文。

建议模块：

- `ShortTermMemory`
- `ConversationStore`
- `ContextCompressor`

## 8.2 长期记忆

长期记忆属于用户。

职责：

- 保存用户稳定偏好。
- 保存跨会话事实。
- 保存历史任务摘要。

调用位置：

```text
user query
-> retrieve long-term memory
-> inject into agent messages
-> agent tool loop
-> finish
-> maybe write long-term memory
```

## 8.3 记忆和知识库的区别

知识库是领域知识，例如“皮肤医学知识”。

记忆是用户相关知识，例如：

- 用户是敏感肌。
- 用户偏好简洁回答。
- 用户经常关注儿童用药。

两者都可以检索，但归属和更新策略不同。

## 9. Tracing 规划

先做内置 tracer，再适配外部系统。

## 9.1 本地 tracing

新增：

- `TraceManager`
- `TraceRecord`
- `TraceSpan`
- `JsonTraceExporter`

最小实现：

```python
with tracer.span("retrieve", input=payload) as span:
    result = retrieve(...)
    span.set_output(result)
```

## 9.2 Trace 覆盖范围

必须覆盖：

- agent loop 每轮
- query_transform
- retrieve 总耗时
- 每一路 retriever 耗时
- rerank 耗时
- generate 耗时
- finish payload
- parser pipeline
- embedding/indexing

## 9.3 后续扩展

预留：

- Langfuse exporter
- OpenTelemetry exporter
- SQLite trace viewer

## 10. 推荐实施步骤

## Phase 0：稳定现有主链路

目标：

- 保持当前 CLI 和 `FaceAiSystem.run_agentic_query()` 可用。
- 不破坏现有静态知识库 fallback。
- 形成可复用的 agentic RAG baseline，后续平台化都围绕它扩展。

任务：

- 清理旧的 `retrieve_generate.py`、`planner.py`、`decompose.py`、`state.py` active path 残留。
- 保留 `judge.py` 作为 fallback evaluator，不作为主流程控制器。
- 确认 `agent.py` 的 tool loop 对真实 LLM tool calling 可用。
- 增加最小单元测试或 smoke test。

产出：

- 当前 agentic RAG baseline 稳定。
- [PHASE0_BASELINE.md](/c:/project/python-rag-pipeline/PHASE0_BASELINE.md) 记录当前边界和验收方式。
- [phase0_smoke.py](/c:/project/python-rag-pipeline/phase0_smoke.py) 提供最小可执行验收入口。

## Phase 1：平台 schemas 和 registry

目标：

先把对象立起来，不急着引入数据库。

新增：

- `rag/platform/schemas.py`
- `rag/platform/registry.py`
- `rag/platform/repositories.py`

对象：

- `UserProfile`
- `LLMProfile`
- `KnowledgeBaseProfile`
- `DocumentAsset`
- `ParsedDocument`
- `ConversationSession`
- `TraceRecord`
- `TraceSpan`

存储方式：

- 初期用 JSON repository。
- 后续再换 SQLite / Postgres。

产出：

- 支持注册用户。
- 支持注册模型配置。
- 支持注册多个知识库 profile。

## Phase 2：知识库对象化和切换

目标：

从单 `knowledgeBase.json` 切到多知识库。

新增：

- `KnowledgeBaseRegistry`
- `KnowledgeBaseRuntime`
- `RetrieverFactory`

改造：

- `ChunkStore` 支持从 `KnowledgeBaseProfile.storage_config` 加载。
- `RetrievalTool` 支持 `kb_ids`。
- `retrieval result` 增加 `kb_id`。

产出：

- 可以创建“皮肤知识库”。
- 可以切换知识库。
- 可以多知识库并行检索。

## Phase 3：Parser 层和 ingest pipeline

目标：

接入 PDF、图片、文本解析。

新增：

- `rag/parsers/base.py`
- `rag/parsers/router.py`
- `rag/parsers/pdf_parser.py`
- `rag/parsers/image_parser.py`
- `rag/parsers/text_parser.py`
- `rag/ingestion/pipeline.py`
- `rag/ingestion/chunker.py`
- `rag/ingestion/indexer.py`

流程：

```text
DocumentAsset
-> ParserRouter
-> ParsedDocument
-> Chunker
-> Embedding
-> Indexer
-> KnowledgeBaseRuntime refresh
```

产出：

- 可以把 PDF 加入指定知识库。
- 可以把图片 OCR 后加入指定知识库。
- 可以把文本/Markdown 加入指定知识库。

## Phase 4：LLMProfile 和用户模型配置

目标：

模型不再只来自环境变量。

新增：

- `LLMProfileRegistry`
- `LLMClientFactory`
- `EmbeddingClientFactory`
- `RerankClientFactory`
- `VisionOcrClientFactory`

改造：

- `LLMClient` 从 `LLMProfile` 构建。
- `SemanticRetriever` 从知识库绑定的 embedding profile 构建。
- `CohereRerankTool` 从知识库或用户绑定的 rerank profile 构建。

产出：

- 用户可注册 OpenAI-compatible 模型。
- 用户可为不同知识库配置不同 embedding 模型。
- 用户可切换 chat model。

## Phase 5：长短期记忆

目标：

让 agent 在对话中具备上下文治理和跨会话记忆。

新增：

- `ConversationStore`
- `ShortTermMemory`
- `ContextCompressor`
- `LongTermMemoryStore`
- `MemoryRetriever`
- `MemoryWriter`

改造 agent 流程：

```text
load conversation
-> load short memory
-> retrieve long memory
-> run agent loop
-> append messages
-> maybe summarize
-> maybe write long memory
```

产出：

- 支持 conversation_id。
- 支持历史对话压缩。
- 支持用户长期偏好召回。

## Phase 6：Tracing

目标：

让每次问答、检索、生成、解析都有可观测记录。

新增：

- `rag/tracing/tracer.py`
- `rag/tracing/exporters.py`

改造：

- agent loop 包 trace。
- retrieval 内部每个 retriever 包 span。
- parser pipeline 每步包 span。
- LLM 调用记录 latency、model、tokens。

产出：

- 本地 JSON trace。
- 后续可接 Langfuse。

## Phase 7：Service/API 层

目标：

把平台能力暴露成接口。

建议先做 Python service，不急着 Web UI。

新增：

- `RAGPlatformService`
- `KnowledgeBaseService`
- `DocumentIngestionService`
- `ConversationService`
- `UserModelService`
- `TraceService`

后续可选：

- FastAPI REST API
- 简单 web 管理界面
- OpenAI-compatible chat endpoint

## 11. 推荐优先级

建议按这个顺序做：

1. `schemas + registry`
2. `KnowledgeBaseProfile + kb switching`
3. `parser router + text/pdf/image parser`
4. `ingestion pipeline`
5. `LLMProfile`
6. `ConversationSession + short memory`
7. `long-term memory`
8. `tracing`
9. `service/api`

原因：

- 知识库对象化是所有后续能力的地基。
- parser 层必须先知道要写入哪个知识库。
- LLMProfile 需要被知识库、parser、agent 共同引用。
- memory 和 tracing 都依赖 conversation/user 对象。

## 12. 最小可行版本

MVP 不需要一上来做完整数据库和 Web UI。

建议 MVP 目标：

- JSON 文件保存用户、模型、知识库、会话。
- 支持两个知识库 profile。
- 支持 PDF / image / text 三种 parser 路由。
- 支持 `agent.run(query, user_id, conversation_id, kb_ids)`。
- 支持本地 JSON trace。
- 保留 CLI。

MVP 完成后的项目形态：

```text
用户选择模型
-> 用户选择知识库
-> 上传文档
-> parser 路由解析
-> chunk / embed / index
-> agent 基于指定知识库回答
-> conversation 记忆更新
-> trace 记录全过程
```

## 13. 一句话总结

平台化的关键不是把 RAGFlow 全量搬过来，而是借它的对象边界：

> UserProfile 管用户，LLMProfile 管模型，KnowledgeBaseProfile 管知识库，DocumentAsset 管入库文档，ConversationSession 管对话，MemoryStore 管长期记忆，TraceRecord 管观测；当前 agentic RAG 作为平台的核心回答引擎，被这些对象驱动和约束。
