# Phase 1 Platform Foundation

Phase 1 的目标是把当前已经跑通的 agentic RAG 链路，升级成一个可以继续平台化的 RAG 基座。

Phase 0 已经解决了“一个 agentic RAG 如何自主检索、生成、评估、结束”的问题。Phase 1 不再继续堆 agent 智能，而是开始解决“这个系统如何管理多个用户、多个模型、多个知识库、多个文件、多个会话、可追踪运行过程”的问题。

一句话概括：

```text
Phase 0: 让 RAG agent 跑起来。
Phase 1: 让 RAG agent 有平台对象、有配置、有边界、有可管理入口。
```

## 1. 当前状态

当前项目已经具备一个较完整的 agentic RAG 核心链路：

```text
用户 query
-> ToolCallingRAGAgent
-> retrieve / generate / finish 工具调用
-> RuntimeStore 保存 retrieval 和 answer artifact
-> trace 记录工具调用、agent_decision、LLM 上下文、raw output
-> StructuredRAGResponse 输出结构化结果
```

核心能力包括：

- 单 agent tool-calling loop。
- 多轮检索与生成。
- retrieve 支持多路检索、RRF、rerank。
- generate 只基于检索 chunks 生成答案。
- finish 只通过 `answer_id` 结束，防止主 agent 绕过 generate 直接写答案。
- RuntimeStore 管理 `retrieval_001`、`answer_001` 等运行期 artifact。
- trace 中记录 `agent_decision`，形成 ReAct-lite 可观测链路。

但当前系统仍然偏 demo：

- 知识库基本还是静态加载。
- 用户没有对象化。
- 模型配置主要依赖环境变量。
- parser / ingest 没有统一入口。
- conversation / memory 只是轻量 history，不是正式模块。
- trace 只随响应返回，没有稳定持久化。
- 不能自然表达“皮肤知识库”“产品知识库”“医学知识库”等多个知识库对象。

## 2. Phase 1 的意义

Phase 1 的意义是建立平台骨架，让后续功能不是继续写死在 agent 里，而是能挂到统一对象上。

例如，Phase 0 里我们问：

```text
怎么让 agent 自己决定检索、生成、结束？
```

Phase 1 要问：

```text
这个 agent 使用哪个用户的模型配置？
这个 query 查哪个知识库？
这个知识库用什么 embedding、rerank、parser、chunk 配置？
这次运行的 trace 存在哪里？
这段对话属于哪个 conversation？
上传文件后如何进入知识库？
```

如果不做 Phase 1，后面新增 parser、memory、tracing、知识库切换都会变成“临时参数 + if else”。做了 Phase 1，后续扩展会变成：

```text
注册对象 -> 选择对象 -> 运行服务 -> 保存状态
```

## 3. 目标架构

Phase 1 后，项目应该逐步形成下面这组对象：

```text
RAGPlatformService
├── UserProfile
├── LLMProfile
├── KnowledgeBaseProfile
├── DocumentAsset
├── ConversationSession
├── RuntimeStore
├── TraceRecord / TraceSpan
└── AgenticRAGService
```

对应的运行链路：

```text
user_id + conversation_id + kb_ids + query
-> 加载 UserProfile
-> 加载 LLMProfile
-> 加载 KnowledgeBaseProfile
-> 构建 KnowledgeBaseRuntime / RetrieverSet
-> 加载 conversation memory
-> 运行 ToolCallingRAGAgent
-> 保存 response / trace / conversation
```

## 4. 核心对象设计

### 4.1 UserProfile

用户对象表示“谁在使用系统”。

建议字段：

```python
@dataclass
class UserProfile:
    user_id: str
    display_name: str
    default_chat_profile_id: str | None = None
    default_embedding_profile_id: str | None = None
    default_rerank_profile_id: str | None = None
    permissions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
```

意义：

- 支持不同用户使用不同模型。
- 支持不同用户默认知识库、权限、偏好。
- 后续可以挂长期 memory、私有知识库、用量统计。

### 4.2 LLMProfile

模型配置对象表示“如何调用某个模型”。

建议字段：

```python
@dataclass
class LLMProfile:
    profile_id: str
    owner_user_id: str | None
    provider: str
    model_type: str
    model_name: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    timeout: int = 60
    max_tokens: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)
```

`model_type` 建议枚举：

```text
chat
embedding
rerank
vision
ocr
```

意义：

- 不再完全依赖环境变量。
- 同一平台可以注册 OpenAI-compatible、DashScope、DeepSeek、Cohere rerank 等不同模型。
- 知识库可以指定 embedding / rerank 模型。
- parser 可以指定 OCR / vision 模型。

### 4.3 KnowledgeBaseProfile

知识库对象表示“一个可检索知识集合”。

建议字段：

```python
@dataclass
class KnowledgeBaseProfile:
    kb_id: str
    name: str
    description: str = ""
    owner_user_id: str | None = None
    language: str = "zh"
    chunk_store_path: str | None = None
    embedding_profile_id: str | None = None
    rerank_profile_id: str | None = None
    retrieval_modes: list[str] = field(default_factory=lambda: ["semantic", "keyword"])
    parser_id: str = "text"
    parser_config: dict[str, Any] = field(default_factory=dict)
    chunking_config: dict[str, Any] = field(default_factory=dict)
    index_config: dict[str, Any] = field(default_factory=dict)
    status: str = "ready"
    metadata: dict[str, Any] = field(default_factory=dict)
```

示例：

```json
{
  "kb_id": "skin_kb",
  "name": "皮肤护理知识库",
  "description": "皮肤护理、防晒、美白、敏感肌相关知识",
  "chunk_store_path": "data/knowledge/skin/chunks.json",
  "embedding_profile_id": "emb_default_zh",
  "rerank_profile_id": "cohere_rerank_default",
  "retrieval_modes": ["semantic", "keyword", "grep"],
  "parser_id": "pdf",
  "parser_config": {
    "ocr": true
  },
  "chunking_config": {
    "chunk_size": 800,
    "overlap": 120
  }
}
```

意义：

- 把“查哪个知识库”变成显式对象。
- 每个知识库可以有自己的检索方式、embedding、rerank、parser、chunk 策略。
- 后续支持 `--kb skin_kb` 或 `kb_ids=["skin_kb", "product_kb"]`。

### 4.4 DocumentAsset

文档对象表示“一个进入知识库的原始文件或文本资源”。

建议字段：

```python
@dataclass
class DocumentAsset:
    doc_id: str
    kb_id: str
    file_name: str
    file_type: str
    source_path: str
    content_hash: str | None = None
    parser_id: str | None = None
    parser_config: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    chunk_count: int = 0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

意义：

- parser / ingest 有了统一输入。
- 后续可以追踪文件处理状态。
- 一个知识库可以管理多个文档。

### 4.5 ConversationSession

会话对象表示“一段用户与 agent 的持续对话”。

建议字段：

```python
@dataclass
class ConversationSession:
    conversation_id: str
    user_id: str
    active_kb_ids: list[str]
    chat_profile_id: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    short_memory: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
```

意义：

- 支持多轮对话上下文。
- 支持短期记忆、摘要压缩。
- 支持一个会话绑定多个知识库。

### 4.6 TraceRecord / TraceSpan

Trace 对象表示“一次 RAG 运行的可观测记录”。

建议字段：

```python
@dataclass
class TraceRecord:
    trace_id: str
    user_id: str | None
    conversation_id: str | None
    kb_ids: list[str]
    query: str
    response: str | None = None
    status: str = "running"
    started_at: str | None = None
    ended_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

```python
@dataclass
class TraceSpan:
    span_id: str
    trace_id: str
    parent_span_id: str | None
    name: str
    span_type: str
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    latency_ms: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

span 类型建议：

```text
agent_loop
agent_decision
retrieve
semantic_retrieve
keyword_retrieve
grep_retrieve
rrf
rerank
generate
finish
parse
chunk
embed
index
```

意义：

- 后续可以回放一次 RAG 调用。
- 可以定位慢在哪里、错在哪里。
- 可以统计检索命中、生成失败、fallback、模型空输出等问题。

## 5. Phase 1 推荐目录结构

建议新增：

```text
rag/
  platform/
    __init__.py
    schemas.py
    registry.py
    repositories.py
    service.py
  models/
    __init__.py
    profiles.py
    factory.py
  tracing/
    __init__.py
    tracer.py
    exporters.py
```

稍后 Phase 2 / Phase 3 再新增：

```text
rag/
  parsers/
    base.py
    router.py
    text_parser.py
    pdf_parser.py
    image_parser.py
  ingestion/
    pipeline.py
    chunker.py
    indexer.py
  memory/
    short_term.py
    long_term.py
    compressor.py
```

## 6. Phase 1 分解步骤

### Step 1: 定义平台 schemas

新增 `rag/platform/schemas.py`。

包含：

- `UserProfile`
- `LLMProfile`
- `KnowledgeBaseProfile`
- `DocumentAsset`
- `ConversationSession`
- `TraceRecord`
- `TraceSpan`

目标：

```text
先把对象边界定下来，不急着做复杂存储。
```

### Step 2: 实现 JSON Repository

新增 `rag/platform/repositories.py`。

先用 JSON 文件存储：

```text
data/platform/users.json
data/platform/llm_profiles.json
data/platform/knowledge_bases.json
data/platform/conversations.json
data/platform/traces/
```

目标：

```text
先本地可运行，后续再换 SQLite / Postgres。
```

### Step 3: 实现 Registry

新增 `rag/platform/registry.py`。

职责：

- `UserRegistry.get(user_id)`
- `LLMProfileRegistry.get(profile_id)`
- `KnowledgeBaseRegistry.get(kb_id)`
- `ConversationRegistry.get(conversation_id)`

目标：

```text
把“按 id 找对象”集中管理。
```

### Step 4: 知识库切换接入现有 RAG

让 CLI 支持：

```powershell
python agentic_rag_cli.py --query "防晒需要注意什么" --kb skin_kb --user default
```

内部流程：

```text
--kb skin_kb
-> KnowledgeBaseRegistry.get("skin_kb")
-> 读取 chunk_store_path
-> 构建当前 KB 的 retrievers
-> agent run
```

目标：

```text
先实现单知识库切换，再考虑多知识库融合。
```

### Step 5: LLMProfile 接入

让 CLI 支持：

```powershell
python agentic_rag_cli.py --query "..." --user default --chat-profile glm_5_1
```

内部流程：

```text
LLMProfile
-> RAGConfig
-> LLMClient
```

目标：

```text
环境变量仍可作为 fallback，但正式运行优先用 LLMProfile。
```

### Step 6: ConversationSession 接入

让 CLI 支持：

```powershell
python agentic_rag_cli.py --query "继续说" --conversation c_001
```

内部流程：

```text
load conversation messages
-> pass history into agent
-> append user / assistant messages
-> save conversation
```

目标：

```text
先做短期记忆，不急着做长期 memory embedding。
```

### Step 7: Trace 持久化

每次 agent run 生成一个 `trace_id`。

保存：

```text
data/platform/traces/{trace_id}.json
```

内容包括：

- query
- response
- kb_ids
- user_id
- conversation_id
- traces
- agent_decision
- retrieval ids
- answer ids
- used queries
- errors

目标：

```text
从“打印 trace”升级为“可回放 trace”。
```

## 7. 与当前 agentic 链路的关系

Phase 1 不应该推翻当前 `ToolCallingRAGAgent`。

建议做法是包一层 service：

```python
class AgenticRAGService:
    def answer(
        self,
        query: str,
        user_id: str = "default",
        conversation_id: str | None = None,
        kb_ids: list[str] | None = None,
        chat_profile_id: str | None = None,
    ) -> StructuredRAGResponse:
        ...
```

这层 service 负责：

```text
加载用户
加载模型配置
加载知识库
加载会话历史
调用 ToolCallingRAGAgent
保存会话
保存 trace
```

而 `ToolCallingRAGAgent` 继续只负责：

```text
工具调用循环
self-RAG 决策
retrieve / generate / finish
RuntimeStore
```

这样边界更清楚。

## 8. 与 RAGFlow 的参考关系

RAGFlow 值得参考的不是某一段具体代码，而是它把 RAG 系统拆成了几类稳定对象：运行时上下文、知识库配置、文档任务、模型配置、对话应用、会话记录、可观测 trace。Phase 1 可以借这个对象拆分方式，但不需要照搬它的数据库、Redis、Celery、前端 Canvas 或企业权限体系。

### 8.1 Canvas 运行时上下文

RAGFlow 的 `ragflow/agent/canvas.py` 里有一套 Canvas 运行时模型：

```text
globals: sys.query / sys.history / sys.user_id / sys.files
component output: component_id@output
get_variable_value()
set_variable_value()
add_reference()
add_memory()
tool_use_callback()
```

它的价值是把“节点之间传递什么状态”显式化，而不是让每个函数私下传递散乱参数。我们当前的 `RuntimeStore` 已经有类似雏形：

```text
sys.last_retrieval_id
sys.last_answer_id
sys.last_valid_answer_id
retrieval_001
answer_001
```

Phase 1 可以继续借这个方向，把运行上下文扩展成平台级上下文：

```text
sys.user_id
sys.conversation_id
sys.kb_ids
sys.trace_id
sys.chat_profile_id
sys.embedding_profile_id
```

建议：保留我们现在轻量的 `RuntimeStore`，不要引入完整 Canvas DAG。我们目前是单 agent tool-calling 链路，不需要 RAGFlow 那种可视化组件编排。

### 8.2 知识库对象

RAGFlow 的 `Knowledgebase` 模型把知识库配置和检索参数放在一起，关键字段包括：

```text
id
tenant_id
name
language
description
embd_id
tenant_embd_id
doc_num
token_num
chunk_num
similarity_threshold
vector_similarity_weight
parser_id
parser_config
status
```

这说明“知识库”不只是一个 chunks 文件路径，而是检索策略、embedding 模型、parser 策略、统计状态的集合。我们的 `KnowledgeBaseProfile` 应该至少覆盖：

```text
kb_id
name
description
language
chunk_store_path
embedding_profile_id
rerank_profile_id
retrieval_modes
similarity_threshold
vector_similarity_weight
parser_id
parser_config
chunking_config
status
metadata
```

建议：Phase 1 先支持 `--kb skin_kb` 加载不同知识库配置。`doc_num`、`token_num`、`chunk_num` 可以先作为统计字段，不必马上做完整管理后台。

### 8.3 文件、文档、任务三层拆分

RAGFlow 把数据进入知识库拆成三层：

```text
File: 原始文件在哪里，谁上传的，文件类型和大小是什么
Document: 这个文件进入哪个 KB，用什么 parser，处理进度如何
Task: 一次解析/切片/embedding/index 的执行任务
```

相关位置：

```text
ragflow/api/db/db_models.py: File / Document / Task
ragflow/rag/svr/task_executor.py: build_chunks() / embedding() / insert_chunks()
ragflow/rag/flow/pipeline.py: callback() 记录进度
```

这个拆分对我们下一阶段很重要，因为现在项目里的 `knowledgeBase.json`、`knowledge.index`、`knowledge_embeddings.npy` 更像已经加工好的结果，缺少“原始文档如何变成索引”的生命周期。

建议：Phase 1 只定义 `DocumentAsset` 和 `IngestionTask` 的 schema，暂不实现完整异步任务队列。Phase 2 再做 Parser + Ingestion。

### 8.4 ParserRouter 和入库流水线

RAGFlow 在 `ragflow/rag/svr/task_executor.py` 里用 `FACTORY` 把不同 parser 类型映射到不同解析器：

```text
naive
paper
book
presentation
manual
laws
qa
table
resume
picture
audio
email
```

它的入库主线是：

```text
file/document
-> parser factory 选择解析器
-> build_chunks()
-> 可选 auto_keywords / auto_questions / metadata / toc
-> embedding()
-> insert_chunks()
-> progress_callback()
```

这对我们的参考意义是：不要把 parser 写死在某个脚本里，而是提前留出 `parser_id + parser_config`。不过 Phase 1 不应该急着实现所有 parser。

建议：Phase 1 只把 `parser_id`、`parser_config`、`chunking_config` 放进 `KnowledgeBaseProfile` 和 `DocumentAsset`。Phase 2 再建：

```text
rag/parsers/router.py
rag/ingestion/pipeline.py
rag/ingestion/chunker.py
rag/ingestion/indexer.py
```

### 8.5 LLMProfile / TenantLLM

RAGFlow 的 `TenantLLM` 把模型配置对象化：

```text
tenant_id
llm_factory
model_type
llm_name
api_key
api_base
max_tokens
used_tokens
status
```

这点非常值得借。我们现在靠环境变量：

```text
OPENAI_API_KEY
OPENAI_MODEL
OPENAI_BASE_URL
```

这对 demo 足够，但平台化之后会遇到几个问题：

```text
chat 模型、embedding 模型、rerank 模型无法独立配置
不同用户/知识库无法使用不同模型
无法记录模型来源、上下文长度、超时、温度、fallback
API key 暴露风险更高
```

建议：Phase 1 做 `LLMProfile`，先支持 OpenAI-compatible provider，字段覆盖 `provider`、`model_type`、`model_name`、`base_url`、`api_key_env`、`temperature`、`timeout`、`max_tokens`。注意配置里最好存 `api_key_env`，不要直接把 key 写进 JSON。

### 8.6 Dialog 和 Conversation

RAGFlow 区分了 `Dialog` 和 `Conversation`：

```text
Dialog: 一个可复用的聊天应用配置，包含 llm_setting、prompt_config、kb_ids、top_n、top_k、rerank_id
Conversation: 某一次用户会话，挂在某个 dialog 下，保存消息和状态
```

这能解释我们 Phase 1 为什么不能只做一个 `history` 列表。平台化以后至少有两层：

```text
ChatProfile: 默认 prompt、模型参数、topk、是否引用、默认 KB
ConversationSession: 当前用户的一次对话，保存 messages、summary、active_kb_ids
```

建议：Phase 1 可以先不叫 `Dialog`，避免引入过重概念；用 `ChatProfile` 更贴近我们当前项目。CLI 可以演进为：

```powershell
python agentic_rag_cli.py --query "防晒需要注意什么" --user default --kb skin_kb --conversation demo_001 --chat-profile default
```

### 8.7 Agent reasoning / thoughts / trace

RAGFlow 有几类可观测机制：

```text
agent_with_tools.py: AgentParam 包含 reasoning、context、user_prompt
canvas.py: tool_use_callback() 记录 tool_name、arguments、result、elapsed_time
base.py / canvas.py: thoughts() 给 UI 展示节点状态
dialog_service.py: 可接 Langfuse trace
pipeline.py: callback() 记录组件进度和耗时
```

这说明它不是把完整思维链暴露给用户，而是保存“可解释的动作摘要”和“可观测执行过程”。这和我们现在的 ReAct-lite 是一致的：

```text
decision.stage
decision.rationale
decision.expected_gain
decision.confidence
tool arguments
tool result
elapsed time
error
```

建议：继续保留 ReAct-lite，不做完整 chain-of-thought 展示。Phase 1 应该把这些 decision 和 tool trace 落盘到 `TraceRecord / TraceSpan`，后续再考虑 Langfuse 或 UI。

### 8.8 Prompt 资产

RAGFlow 的 `ragflow/rag/prompts/` 里有很多独立 prompt 文件，例如：

```text
sufficiency_check.md
multi_queries_gen.md
next_step.md
reflect.md
citation_prompt.md
citation_plus.md
summary4memory.md
rank_memory.md
tool_call_summary.md
```

它的参考意义是 prompt 应该资产化、按能力拆分，而不是全部堆在一个 Python 字符串里。我们现在已经有 `rag/prompts.py`，Phase 1 可以先保持简单；等 prompt 继续增多后，再迁移成：

```text
rag/prompts/
  agent_system.md
  answer_system.md
  judge_system.md
  sufficiency_check.md
  multi_query.md
```

建议：短期只保持中文 prompt 清晰可测，不急着拆文件。等 Phase 2/Phase 3 prompt 数量增加再拆。

### 8.9 现在应该借什么，不借什么

Phase 1 应该借：

```text
Canvas 的显式运行时变量思想
Knowledgebase 的配置对象思想
TenantLLM 的模型配置对象思想
Dialog/Conversation 的配置与会话分离思想
File/Document/Task 的入库生命周期思想
tool callback / pipeline callback 的 trace 思想
parser FACTORY 的可插拔 parser 思想
```

Phase 1 暂时不借：

```text
完整 Canvas DAG 编排
Redis 日志缓存
数据库 ORM 全量模型
异步任务队列和取消机制
Web UI
MCP 插件系统
Langfuse 深度集成
GraphRAG / RAPTOR / mindmap
复杂权限和团队协作
```

最合适的落点是：先把我们的系统对象边界搭起来，让 CLI 能通过 `user_id + kb_id + conversation_id + chat_profile_id` 启动现有 agentic RAG。这样不会把当前已跑通的链路推倒重来，但会为后续 parser、memory、API、UI 留出清楚接口。

## 9. MVP 验收标准

Phase 1 完成后，至少应该满足：

### 9.1 配置对象可加载

可以从 JSON 加载：

- default user
- default chat profile
- skin knowledge base

### 9.2 CLI 支持对象参数

示例：

```powershell
python agentic_rag_cli.py --query "防晒需要注意什么" --user default --kb skin_kb --conversation demo_001 --print-traces
```

### 9.3 知识库可以切换

不同 `--kb` 对应不同 chunk 文件。

### 9.4 会话可以保存

同一个 `conversation_id` 下，下一轮 query 能拿到上一轮 history。

### 9.5 trace 可以落盘

运行后生成：

```text
data/platform/traces/{trace_id}.json
```

里面能看到：

- agent_decision
- retrieve 参数与结果
- generate 上下文与 raw output
- finish 结果
- fallback / error

## 10. 建议实施顺序

推荐顺序：

```text
1. schemas.py
2. JSON repositories
3. registries
4. default platform config files
5. AgenticRAGService
6. CLI 增加 --user / --kb / --conversation / --chat-profile
7. conversation 保存
8. trace 保存
```

不要先做 parser。

原因是 parser 属于 ingest pipeline，它依赖：

- 知识库对象
- 文档对象
- chunking config
- storage config

先把这些对象做出来，parser 才不会变成孤立脚本。

## 11. Phase 1 之后

Phase 1 完成后，下一阶段可以自然进入：

```text
Phase 2: Parser + Ingestion
Phase 3: Multi-KB Retrieval Runtime
Phase 4: Memory
Phase 5: Tracing UI / Langfuse / API
```

其中最推荐的下一步是：

```text
Phase 2: Parser + Ingestion
```

因为有了 KnowledgeBaseProfile 和 DocumentAsset 后，就可以自然实现：

```text
PDF / image / text
-> ParserRouter
-> ParsedDocument
-> Chunker
-> Indexer
-> KnowledgeBaseRuntime
```

## 12. 最小实现范围

如果只做 Phase 1 MVP，不做过度设计，建议只落地：

- `UserProfile`
- `LLMProfile`
- `KnowledgeBaseProfile`
- `ConversationSession`
- `TraceRecord`
- JSON repository
- registry
- `AgenticRAGService`
- CLI 参数接入

暂时不做：

- 数据库
- Web UI
- 权限系统
- 长期 memory embedding
- parser 真实复杂实现
- 多租户隔离
- trace 可视化

这样能最快把项目从“单链路 demo”推进到“平台雏形”。
