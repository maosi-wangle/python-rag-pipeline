# 项目详细讲解

这份文档专门回答你前面一直在追问的三个层面：

1. 架构逻辑
2. 模块逻辑
3. 代码逻辑

同时把 `rationale`、`next_focus`、self-RAG 如何判断要不要继续、多路检索与重生成之间的关系，一次讲清楚。

## 1. 架构逻辑

### 1.1 现在到底是不是两个 agent

不是。

现在是一个主 agent。

这个主 agent 自己维护对话消息和工具调用历史，然后自己决定下一步：

- 是否先改写 query
- 是否直接检索
- 是否基于已有证据重生成
- 是否结束

也就是说，现在没有“一个 agent 检索生成，另一个 agent 判断是否多轮”的拆分。

## 1.2 现在的 self-RAG 是怎么实现的

现在的 self-RAG 不是一个单独的外层控制器去机械地重复固定流程，而是内化到主 agent 的决策里：

```text
retrieve -> generate -> self-evaluate
```

然后主 agent 决定：

- 证据不够：再 `retrieve`
- 证据够但答案写得不好：再 `generate`
- 已经足够：`finish`

所以“多轮”的本质不再是固定 for 循环，而是 agent 根据当前状态决定要不要继续调用工具。

## 1.3 为什么这样更适合你的需求

因为你要的是：

- modular RAG
- agentic decision
- self-RAG

这三者的最自然组合不是把所有环节都做成智能决策，而是：

- 检索和生成本身固定成稳定工作流
- 是否继续、补检索还是重生成，交给 agent 决策

这样既保留了可控性，又把“下一步做什么”的自由度交给 LLM。

## 1.4 现在的主流程

```text
user query
-> main agent
   -> optional query_transform
   -> retrieve
   -> generate
   -> self-evaluate in agent
      -> retrieve again
      -> or generate again
      -> or finish
-> structured result
```

## 1.5 这套架构里 finish 是干嘛的

`finish` 是显式的结束动作。

不是“默认没有 tool call 就结束”，而是要求 agent 明确调用 `finish`，提交最终结构化结果。

这有三个好处：

1. 终止条件清晰。
2. 输出结构固定。
3. traces 和调试信息更容易收口。

## 2. 模块逻辑

## 2.1 主 agent

- 文件：[rag/agent.py](/c:/project/python-rag-pipeline/rag/agent.py)

主 agent 负责：

- 维护 messages
- 向 LLM 注册 tool schema
- 解析 tool call
- 执行工具
- 保存 `retrieval_id -> evidence` 映射
- 在 `finish` 后输出结构化结果

这层是整个项目真正的“agentic”核心。

## 2.2 query_transform 模块

- 文件：[rag/tools/rewrite.py](/c:/project/python-rag-pipeline/rag/tools/rewrite.py)

它是 modular 的第一部分。

职责：

- 让 query 更具体
- 让 query 更泛化
- 让 query 更像 chunk 文档语言
- 拆成 subquery
- 给每个 query 绑定 retrieval modes

标准输出：

```json
{
  "transform_applied": true,
  "transform_type": "rewrite|decompose|hybrid|none",
  "queries": ["..."],
  "plans": [
    {
      "query": "...",
      "retrieval_modes": ["semantic", "keyword", "grep"]
    }
  ],
  "rationale": "..."
}
```

这里 modular 的关键点在于：

- query rewrite 是一个独立工具
- decomposition 是一个独立能力
- retrieval mode binding 也是独立能力
- 但它们统一收口成 `plans`

所以后面的 `retrieve` 根本不关心这些 query 是原始的、改写的还是拆出来的。

## 2.3 retrieve 模块

- 文件：[rag/tools/retrieve.py](/c:/project/python-rag-pipeline/rag/tools/retrieve.py)

这是 modular 的第二部分。

它只负责检索链路，不负责回答。

内部流程固定为：

```text
plans
-> multi-route retrieval in parallel
-> wait all
-> merge / dedup
-> RRF
-> final rerank
-> return evidence
```

### 为什么是先 RRF 再 rerank

因为你后来明确要求改成：

- 多路检索召回内容先直接 RRF
- 最后再重排

这也是现在代码里的实现。

### 并行点在哪里

并行粒度是：

```text
每个 plan × 每种 retrieval mode
```

所以例如：

```text
3 个 subquery
×
3 种检索方式
= 9 个并行 retrieval task
```

然后等待全部完成后，再做 RRF 和 final rerank。

## 2.4 generate 模块

- 文件：[rag/tools/answer.py](/c:/project/python-rag-pipeline/rag/tools/answer.py)

这是 modular 的第三部分。

它只负责：

- 根据 `retrieval_id` 找回证据
- 基于证据生成答案
- 接收可选 `instruction`

这个 `instruction` 很重要，因为它让“重生成”变得很干净。

比如 agent 发现：

- 证据已经够了
- 但答案不够完整

那就不用再检索，而是直接：

```text
generate(query, retrieval_id, instruction="补充注意事项与禁忌，并严格引用 chunk id")
```

这就是你要的“不是每次多轮都要从头走改写和检索”。

## 2.5 finish 模块

- 文件：[rag/tools/finish.py](/c:/project/python-rag-pipeline/rag/tools/finish.py)

这是 modular 的第四部分。

它只做收口，不做决策。

输入：

- `response`
- `query`
- `grounded`
- `completeness`
- `if_multi_turn`
- `rationale`
- `next_focus`
- `retrieval_id`

输出：

- 固定结构化 payload

## 2.6 orchestrator 模块

- 文件：[rag/orchestrator.py](/c:/project/python-rag-pipeline/rag/orchestrator.py)

现在的 `orchestrator` 已经不是旧版那种“一轮 agent 执行器”。

现在它更像一个组件装配器，统一初始化：

- knowledge base
- semantic retriever
- keyword retriever
- grep retriever
- reranker
- query transform tool
- retrieve tool
- generate tool
- finish tool

所以它的逻辑从“控制流中心”变成了“能力中心”。

## 3. 代码逻辑

## 3.1 入口怎么走

CLI 入口：

- [agentic_rag_cli.py](/c:/project/python-rag-pipeline/agentic_rag_cli.py)

兼容入口：

- [faceaiRAG.py](/c:/project/python-rag-pipeline/faceaiRAG.py)

`FaceAiSystem.run_agentic_query()` 现在会直接调用：

```python
ToolCallingRAGAgent.run(...)
```

## 3.2 agent.run() 做了什么

在 [rag/agent.py](/c:/project/python-rag-pipeline/rag/agent.py) 里，`run()` 的逻辑可以概括为：

```python
reset runtime state
build initial messages

for round in max_rounds:
    llm.chat_with_tools(...)
    if no tool call:
        auto finish
    else:
        execute each tool call
        append tool result into messages
        if finish:
            return final structured response

if max rounds reached:
    auto finish
```

这就是你之前写的那个 `chat()` 伪代码，在项目里的落地版本。

## 3.3 retrieval_id 是怎么工作的

这是现在代码里很关键的点。

`retrieve` 执行后，不是只把文本扔回给 LLM，而是会在 agent 内部保存：

```text
retrieval_id -> full retrieval artifact
```

artifact 里有：

- retrieval results
- fused hits
- used queries

然后 `generate` 只需要带上 `retrieval_id`，就能从 agent 内存取回完整证据。

这样做的好处是：

1. 不需要把长上下文再次作为 tool 参数塞回去。
2. 重生成非常轻量。
3. 多轮补检索和重生成可以清楚地区分。

## 3.4 self-RAG 是怎么判断要不要多轮的

不是单独的 judge tool 在外部裁决，而是主 agent 在看完：

- query
- retrieval output
- generation output

之后自己判断：

### 情况 1：证据不够

表现：

- chunk 没覆盖用户问题的关键方面
- grounded 风险高
- `retrieve` 返回内容偏题

动作：

- 再次 `retrieve`
- 必要时先 `query_transform`

### 情况 2：证据够，但答案不够好

表现：

- 遗漏关键信息
- 引用不充分
- 组织不清楚

动作：

- 直接再 `generate`
- 给更精确的 `instruction`

### 情况 3：已经足够

动作：

- 调 `finish`

所以现在“self-RAG 判断是否继续”的本质是：

```text
agent 对当前证据覆盖度 + 答案质量的联合判断
```

## 3.5 rationale 和 next_focus 是干嘛的

### rationale

`rationale` 是“为什么现在结束”的说明。

它回答的是：

- 为什么现在可以停
- 或者为什么虽然不完美但也要停

示例：

- “现有证据已经覆盖主要问题，答案可结束。”
- “已达到最大轮数，建议后续补充更精确检索。”

### next_focus

`next_focus` 是“如果还要继续，下一轮最该补什么”。

它不是流程控制字段。

它只是诊断字段。

示例：

- “补充防晒霜补涂频率”
- “补充儿童和敏感肌的注意事项”

所以：

- `rationale` 解释为什么停
- `next_focus` 提醒如果继续该补什么

## 4. 现在到底对什么做了 modular

可以分五层来看。

### 4.1 Query Modular

- contextualization
- rewrite mode
- decomposition
- retrieval plan binding

### 4.2 Retrieval Modular

- semantic retrieval
- keyword retrieval
- grep retrieval
- parallel execution
- RRF fusion
- rerank

### 4.3 Generation Modular

- grounded answer generation
- answer regeneration with extra instruction

### 4.4 Finish Modular

- structured result commit
- stop signal

### 4.5 Runtime Modular

- LLM tool-calling layer
- retrieval memory layer
- entrypoint compatibility layer

## 5. 五个要点，用 STAR 方式介绍

这里按你要的 STAR 风格来讲这次改造。

## 5.1 从固定 self-RAG 外层循环改成主 agent 自主决策

### Situation

旧版是外层 `for` 循环决定是否继续，主流程比较死。

### Task

把“是否继续、补检索还是重生成”交给主 agent 自己判断。

### Action

新增 [rag/agent.py](/c:/project/python-rag-pipeline/rag/agent.py)，实现真正的 tool-calling loop，并把 `finish` 设为唯一结束信号。

### Result

现在系统不是机械多轮，而是 agent 根据当前证据和答案状态决定下一步。

## 5.2 从 retrieve_generate 一体化改成 retrieve / generate 解耦

### Situation

旧版把检索、融合、重排、生成绑在一个工具里，无法优雅支持“只重生成”。

### Task

支持“证据够但答案不够好”的场景。

### Action

把旧的 `retrieve_generate` 拆成：

- `retrieve`
- `generate`

并引入 `retrieval_id` 做证据缓存。

### Result

现在可以自然支持：

- `retrieve -> generate -> finish`
- `retrieve -> generate -> generate -> finish`
- `retrieve -> generate -> retrieve -> generate -> finish`

## 5.3 把多路检索统一收口成固定检索工作流

### Situation

你要求多路并行检索，最后统一 RRF，再 final rerank。

### Task

把这条链做成稳定、可复用、可并行的确定性工具。

### Action

在 [rag/tools/retrieve.py](/c:/project/python-rag-pipeline/rag/tools/retrieve.py) 里实现：

- parallel retrieval
- wait all
- RRF
- final rerank

### Result

主 agent 不再关心屏障同步和融合细节，只关心“要不要再检索”。

## 5.4 把 query 改写和拆分做成独立模块

### Situation

你需要具体化、泛化、chunk-like、拆分 subquery 等 transformation 能力。

### Task

把 transformation 做成独立 tool，而不是散落在主流程里。

### Action

重写 [rag/tools/rewrite.py](/c:/project/python-rag-pipeline/rag/tools/rewrite.py)，统一输出 `plans`。

### Result

主 agent 可以自由选择：

- 先 transform 再 retrieve
- 或者直接 retrieve

## 5.5 把最终结束状态结构化

### Situation

你需要的不只是答案，还要知道 grounded、completeness、是否还值得继续。

### Task

把结束状态标准化，便于外部系统消费。

### Action

重写 [rag/tools/finish.py](/c:/project/python-rag-pipeline/rag/tools/finish.py)，统一返回：

- `response`
- `query`
- `grounded`
- `completeness`
- `if_multi_turn`
- `rationale`
- `next_focus`

### Result

外部拿到的不再只是“一个回答”，而是带自评状态的结构化结果。

## 6. 现在这套架构最关键的一句话

现在这套系统最准确的描述是：

> 一个主 agent 通过 tool-calling loop 驱动 modular RAG；检索与生成是固定工作流，是否补检索、重生成或结束由 agent 自主判断，最终通过 finish 输出带 grounded / completeness / if_multi_turn 的结构化结果。
