# AstraRAG

AstraRAG 是一个面向中文皮肤知识问答场景的 modular agentic RAG 引擎实验项目。项目从知识库构建、query 改写、混合检索、检索记忆、答案生成到 RAGAS 评测形成完整链路，并在平台层引入用户配置、多知识库选择、多 LLM profile、附件解析、MCP 工具扩展和会话 trace 持久化能力。

内部 Python 包名仍为 `rag/`，项目名为 `AstraRAG`。

## 产出概述

本项目搭建了一条可落地的 modular RAG 检索链路，覆盖以下阶段：

```text
原始资料 / Markdown
-> 知识库 chunk 构建
-> FAISS 语义索引 + 关键词倒排索引
-> query 改写 / 拆分模块
-> semantic / keyword / grep 混合检索
-> RRF 融合 + rerank
-> agent 工具调用与多轮检索记忆
-> grounded generation
-> structured response / trace
-> RAGAS retrieval evaluation
```

核心产出包括：

- 知识库构建：`markdown_chunk_processor.py` 将 Markdown 文档切分为 `knowledgeBase.json` 所需的 chunk 结构；`SemanticRetriever` 可加载或重建 FAISS 索引，`KeywordRetriever` 可加载或重建倒排索引。
- 检索链路：`RetrievalTool` 并行调用 semantic、keyword、grep 检索器，使用 reciprocal rank fusion 融合召回结果，并通过 Cohere rerank 或本地 lexical fallback rerank 得到最终证据。
- Agentic RAG：`ToolCallingRAGAgent` 将知识库选择、检索、生成、finish、MCP 等能力组织为 agent 可调用工具，支持多轮 tool-call、检索结果记忆和结构化输出。
- 平台化运行：`AgenticRAGService` 通过 JSON repository 管理用户、聊天配置、LLM profile、知识库 profile、会话、artifact 和 trace，形成多用户、多知识库、多模型的 RAG 平台雏形。
- 评测闭环：`eval_ragas.py` 可基于正式评测集生成 retrieval enriched dataset，并运行 RAGAS 检索指标，为知识库和检索策略迭代提供量化反馈。

## Modular 架构

项目针对传统 RAG 链路耦合度较高、拓展性不强的问题，将 RAG 系统拆成低耦合模块，并通过统一 schema 和 service 层组织起来。

主要模块：

```text
rag/
├── config.py              # RAGConfig，集中管理路径、模型、topk、超时等参数
├── schemas.py             # ChunkRecord / RetrievalHit / StructuredRAGResponse
├── knowledge_base.py      # ChunkStore，负责加载本地 chunk store
├── retrievers/            # semantic / keyword / grep / fusion
├── tools/                 # retrieve / rewrite / answer / judge / rerank / finish
├── runtime.py             # 单次 agent run 内的 retrieval / answer 记忆
├── agent.py               # ToolCallingRAGAgent
├── orchestrator.py        # 组装检索器、工具和 LLM client
├── input/                 # 多模态输入解析与 artifact 存储
├── mcp/                   # MCP-style tool discovery / session / runtime
└── platform/              # 用户、LLM、知识库、会话、trace 的平台化管理
```

解耦方式：

- 用户配置模块化：`rag/platform/schemas.py` 定义 `UserProfile`、`ChatProfile`、`LLMProfile`、`KnowledgeBaseProfile`、`ConversationSession`、`TraceRecord`。
- 知识库模块化：每个 `KnowledgeBaseProfile` 可以指定独立的 chunk store、FAISS index、embedding 文件、倒排索引和检索模式。
- LLM 模块化：`LLMProfile` 支持 OpenAI-compatible provider，可通过 `api_key_env`、`base_url_env` 和 `model_name` 切换不同模型服务。
- Agent 链路模块化：`ToolCallingRAGAgent` 暴露 `retrieve`、`generate`、`finish` 以及 MCP 工具；query 改写由 `QueryRewriteTool` 独立实现，可接入 agent 检索规划。
- Memory 模块化：`RuntimeStore` 负责单次运行内的 retrieval memory 和 answer memory；`ConversationSession` 负责跨轮对话历史、活跃知识库、附件和短期记忆字段。
- MCP 扩展模块化：`rag/mcp/` 将 provider、tool cache、runtime call 和 retrieval evidence 转换拆开，当前实现 Firecrawl web search provider。

平台模式入口在 `rag/platform/service.py`。调用 `AgenticRAGService.answer()` 时，系统会先解析用户、聊天配置、知识库列表和 LLM profile，再为每个候选知识库构建 `ModularRAGOrchestrator`，最后交给 agent 自主选择知识库和召回策略。

## Input Parser

项目为平台增加了多模态输入识别和解析路由，核心代码位于 `rag/input/parser.py`、`rag/input/storage.py` 和 `rag/input/schemas.py`。

支持格式：

```text
.txt
.md / .markdown
.html / .htm
.png / .jpg / .jpeg
```

解析流程：

```text
--file path
-> LocalArtifactStorage.store_file / to_parsed_artifact
-> MultimodalInputParser.parse_artifact
-> 按后缀路由到 text / markdown / html / image parser
-> 生成 ParsedMessage 和 InputArtifact
-> build_query_with_artifacts
-> 附件内容进入本次 query context
-> 若提供 conversation_id，则 artifact 写入平台记忆
```

各类输入处理方式：

- 文本：直接读取 UTF-8 文本，忽略无法解码字节。
- Markdown：去除代码块、链接、标题、列表符号等标记，提取纯文本。
- HTML：通过 BeautifulSoup 移除 `script`、`style`、`noscript`，提取正文和标题。
- 图片：通过 EasyOCR 识别中英文文字，保留 OCR blocks、置信度、坐标框、图片尺寸和简单图片类型判断。

附件并不会自动写入知识库。它的定位是“平台输入上下文”：在普通模式下作为本轮 query context；在平台会话模式下保存为 artifact，并通过 `ConversationSession.metadata.artifact_ids` / `active_artifact_ids` 进入后续上下文记忆。

## Agentic Self-RAG

AstraRAG 的 agentic-self-RAG 目标是解决单轮检索回答不足的问题：让 agent 根据上下文、召回结果和生成质量，自主决定是否继续检索、换知识库、重新生成或结束回答。

当前代码中的关键实现：

- Agent 工具循环：`ToolCallingRAGAgent.run()` 维护 tool-call loop，在每轮中由 LLM 选择调用 `retrieve`、`generate`、`finish` 或 MCP 工具。
- 自主知识库选择：平台模式下 agent 会收到可用知识库 catalog，`retrieve` 工具支持传入 `kb_ids`，并通过 `_effective_kb_ids()` 在多知识库运行时中选择目标 KB。
- 多轮检索记忆：`RuntimeStore.put_retrieval()` 会把每次检索结果保存为 `retrieval_001`、`retrieval_002` 等 ID，后续 `generate` 可引用单个 retrieval、多个 retrieval 或 `all`。
- 多轮答案记忆：`RuntimeStore.put_answer()` 保存生成结果，`finish` 可通过 `answer_id` 使用最近有效答案。
- 检索计划：`retrieve` 支持 `plans` 参数，每个 plan 可以指定 query 和 retrieval modes，实现多 query、多检索模式召回。
- Query 改写模块：`QueryRewriteTool` 支持 specific、general、chunk_like、hybrid、decompose 等模式，并能在无 LLM 时使用启发式 fallback。
- 回答评估：`AnswerJudgeTool` 可在有 LLM 时做 groundedness / completeness 判断；无 LLM 时使用 lexical overlap 和支持度得分 fallback。
- 会话上下文：平台模式会从 `ConversationSession.messages` 中读取最近历史，并把用户 query、assistant response、trace_id、active_kb_ids 持久化。

当前已落地的是“单次运行内 retrieval / answer memory + 平台会话历史 + artifact context”。`ConversationSession` 中已经预留 `summary` 和 `short_memory` 字段，适合继续实现定期 session 摘要、长期记忆沉淀和 memory ranking；这部分属于下一阶段增强方向。

典型 agentic flow：

```text
user query
-> 读取 conversation history / artifacts
-> agent 判断是否 retrieve
-> retrieve 选择 KB、query plans、retrieval modes
-> RuntimeStore 保存 retrieval_id
-> generate 基于 retrieval evidence / memory / mixed source 生成答案
-> agent 判断是否继续 retrieve / regenerate / finish
-> finish 输出 structured response
-> platform 保存 conversation 和 trace
```

## RAG 链路评测

评测脚本位于 `eval_ragas.py`，用于验证检索链路在中文皮肤知识问答场景下的有效性。

评测输入：

- 自主构建的皮肤知识库：`knowledgeBase.json`、`knowledge.index`、`knowledge_embeddings.npy`、`inverted_index.json`
- 正式评测集：`ragas_eval_dataset.formal.json`
- Top-K 设置：`topk=5`
- 输出目录：`ragas_outputs_formal/`

运行方式：

```powershell
python eval_ragas.py `
  --dataset ragas_eval_dataset.formal.json `
  --topk 5 `
  --output-dir ragas_outputs_formal
```

`eval_ragas.py` 的流程：

```text
load dataset
-> validate user_input
-> FaceAiSystem.retrieve_for_ragas
-> ModularRAGOrchestrator.build_ragas_payload
-> 写出 retrieval_enriched.json
-> 构建 RAGAS EvaluationDataset
-> 根据字段选择可运行指标
-> 输出 ragas_result.csv
```

正式评测结果记录：

| 指标 | 分数 |
| --- | ---: |
| Faithfulness | 0.89 |
| Context Recall | 0.87 |
| Answer Relevancy | 0.96 |
| Context Precision | 0.83 |

该结果说明，在 300 条正式评测集、Top-5 检索设置和皮肤知识库场景下，系统对中文皮肤知识问答具有较好的检索有效性和答案相关性。

说明：当前仓库中的 `eval_ragas.py` 主要覆盖 retrieval enriched dataset 和检索侧 RAGAS 指标；Faithfulness 与 Answer Relevancy 属于带 response 的生成侧评测口径，复现实验时需要确保评测集包含 response / reference，并配置可用 LLM judge。

## 运行方式

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

可选环境变量：

```powershell
$env:OPENAI_API_KEY="..."
$env:OPENAI_MODEL="..."
$env:OPENAI_BASE_URL="..."
$env:COHERE_API_KEY="..."
$env:FIRECRAWL_API_KEY="..."
```

普通本地模式：

```powershell
python agentic_rag_cli.py --query "你的问题" --topk 5 --max-rounds 3
```

平台模式：

```powershell
python agentic_rag_cli.py `
  --platform-root data/platform `
  --user default `
  --kb default `
  --conversation demo `
  --query "你的问题"
```

附加文件：

```powershell
python agentic_rag_cli.py `
  --user default `
  --conversation demo `
  --file README.md `
  --query "总结这个附件"
```

管理平台配置：

```powershell
python platform_admin_cli.py users list
python platform_admin_cli.py llms list
python platform_admin_cli.py chats list
python platform_admin_cli.py kbs list
python platform_admin_cli.py mcps tools --server firecrawl
```

Smoke check：

```powershell
python phase0_smoke.py --topk 2 --max-rounds 2
```

## 代码索引

- `agentic_rag_cli.py`：命令行入口，负责普通模式 / 平台模式分流。
- `faceaiRAG.py`：兼容入口，封装 `FaceAiSystem`。
- `rag/orchestrator.py`：组装知识库、检索器、reranker、answer generator、judge、finish。
- `rag/agent.py`：tool-calling agent 主循环，多轮工具调用和结构化响应。
- `rag/tools/retrieve.py`：并行检索、RRF、rerank、检索预算。
- `rag/tools/rewrite.py`：query 改写、泛化、chunk-like rewrite、拆分。
- `rag/input/parser.py`：多模态输入解析路由。
- `rag/platform/service.py`：平台服务入口，解析用户、会话、知识库、LLM、MCP 和 artifact。
- `rag/mcp/session.py`：Firecrawl provider 和 MCP-style session。
- `eval_ragas.py`：RAGAS 检索评测脚本。

