# Python RAG Pipeline

## 项目简介

本项目是一个面向中文皮肤护理知识库的本地 RAG 检索原型，包含：

- 文档切块与关键词抽取
- 向量索引构建与语义检索
- 倒排索引与关键词检索
- RRF 混合检索
- 基于 RAGAS 的检索量化评估

当前项目评估范围只覆盖检索阶段，不包含 LLM 答案生成阶段。

## 正式评估结果

正式评测集文件：

- `ragas_eval_dataset.formal.json`

样本规模：

- 30 条

执行命令：

```powershell
python eval_ragas.py --dataset ragas_eval_dataset.formal.json --topk 5 --output-dir ragas_outputs_formal
```

正式结果如下：

| Metric | Score |
| --- | --- |
| non_llm_context_precision_with_reference | 0.6750 |
| non_llm_context_recall | 0.8667 |
| id_based_context_precision | 0.1733 |
| id_based_context_recall | 0.8667 |

## 结果解读

- `non_llm_context_recall = 0.8667`
  - 大部分标准上下文能够被召回，说明当前链路的覆盖能力较好。
- `non_llm_context_precision_with_reference = 0.6750`
  - 召回结果整体相关性较好，但前几条结果里仍有一定比例的非目标 chunk。
- `id_based_context_recall = 0.8667`
  - 按 chunk id 来看，标准目标块大多数情况下也能进入 top-k，因此这一项同样是较高的。
- `id_based_context_precision = 0.1733`
  - id 级别的精确率较低，这说明 top-k 中虽然常常能包含正确块，但也混入了较多其它相关或近邻块。

综合来看，当前链路的特点是：

- 优势：召回覆盖较好，混合检索设计有效
- 局限：top-k 排序精度仍有提升空间，尤其容易混入相邻主题或语义相关但非目标的 chunk

## 项目结构

核心代码：

- `faceaiRAG.py`
  - 主检索链路
  - 包含向量检索、关键词检索、混合检索
  - 新增了 `retrieve_for_ragas()` 供 RAGAS 输出结构化检索结果
- `markdown_chunk_processor.py`
  - 原始文本切块
  - 关键词提取
- `eval_ragas.py`
  - RAGAS 检索评估脚本

核心数据：

- `knowledgeBase.json`
  - 当前知识库切块结果
- `ragas_eval_dataset.formal.json`
  - 正式检索评测集

依赖文件：

- `requirements-ragas.txt`
  - RAGAS 评估依赖

## RAG 链路说明

### 1. 切块与关键词

知识库数据由切块脚本生成，每个 chunk 至少包含：

- `context`
- `keywords`

关键词来自 `jieba.analyse.extract_tags` 等方法的抽取结果，用于增强后续检索。

### 2. 语义检索

语义检索流程如下：

1. 使用 `shibing624/text2vec-base-chinese` 对 `keywords + context` 编码
2. 使用 `Faiss IndexFlatIP` 构建向量索引
3. 查询时对问题编码后做向量相似度检索

当前索引类型是精确向量检索，不是近似 ANN 索引。

### 3. 关键词检索

项目中还实现了基于倒排索引的关键词检索：

1. 对 `keywords + context` 分词
2. 构建 `term -> doc_ids` 倒排表
3. 查询时做关键词命中计数排序

这里的关键词检索不是 BM25，而是简单的命中计数检索。

### 4. 混合检索

最终检索结果由以下两路融合得到：

- 语义检索
- 关键词检索

融合方式为 `RRF (Reciprocal Rank Fusion)`。

## RAGAS 评估说明

### 评估目标

本次评估只针对检索模块，目的是量化当前 RAG 检索链路的召回能力和精度。

### 评估方式

本项目采用的是 RAGAS 中的非 LLM 检索指标，因此本次评估不依赖额外评估模型。

使用指标如下：

- `non_llm_context_precision_with_reference`
- `non_llm_context_recall`
- `id_based_context_precision`
- `id_based_context_recall`

其中：

- 非 LLM 指标通过 `retrieved_contexts` 与 `reference_contexts` 比较计算
- ID 指标通过 `retrieved_context_ids` 与 `reference_context_ids` 比较计算

### 正式评测集覆盖主题

- 美白祛斑
- 防晒
- 儿童防晒
- 补水保湿
- 眼部护理
- 基础皮肤知识

## 如何运行

### 1. 检索主程序

```powershell
python faceaiRAG.py
```

### 2. 安装 RAGAS 评估依赖

```powershell
python -m pip install -r requirements-ragas.txt
```

### 3. 执行正式评估

```powershell
python eval_ragas.py --dataset ragas_eval_dataset.formal.json --topk 5 --output-dir ragas_outputs_formal
```

## 项目结论

本项目已经完成一条可运行的中文 RAG 检索链路，并通过 RAGAS 完成了检索阶段的量化评估。

从正式评测结果看：

- 当前链路具备较好的召回能力
- 混合检索设计有效
- 但在 top-k 排序精度方面仍有进一步优化空间
