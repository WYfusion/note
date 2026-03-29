## 一句话概括
MonoT5 是一种基于 [[02_去噪自编码_Denoising_T5#^60fffe|T5]]（Text-to-Text Transfer Transformer）的生成式精排模型，将排序问题转化为序列生成任务——输入 query + doc，让模型生成 "true" 或 "false" 来判断相关性。

## 核心原理
### 关键思想：排序即生成
传统 Cross-Encoder 把排序当分类任务（输出分数）。MonoT5 换了个思路：**把排序当文本生成任务**。

### 具体流程
**输入格式**
`Query: {query 文本} Document: {document 文本} Relevant:`

**模型输出**
`"true"` 或 `"false"`

**相关性分数**
取模型生成 "true" 的概率作为相关性分数：
$$score = P(\text{"true"} | \text{query, document})$$
### 为什么用生成方式？

1. **T5 的统一框架**：T5 将所有 NLP 任务都转化为 text-to-text 格式，排序任务也不例外
2. **利用预训练知识**：T5 在大规模文本上预训练的语言理解能力可以直接用于判断相关性
3. **灵活性**：输入格式灵活，可以加入更多上下文信息（如指令、领域描述）

### 与 Cross-Encoder 的区别

|对比项|Cross-Encoder|MonoT5|基础架构|BERT（仅编码器）|T5（编码器-解码器）|
|---|---|---|---|---|---|
|任务类型|分类（输出分数）|生成（输出 "true"/"false"）|评分方式|[CLS] → 线性层 → 分数|生成 "true" 的概率|
|参数量|110M~335M|220M~11B（T5 系列）|预训练任务|MLM（掩码语言模型）|Span Corruption（跨度填充）|

### 训练方式

在标注数据上微调 T5：

- 正样本：query + 相关 doc → 目标输出 "true"
- 负样本：query + 不相关 doc → 目标输出 "false"
- 损失函数：标准的序列生成交叉熵损失

## 模型版本

|版本|参数量|速度|精度|MonoT5-base|220M|快|中|
|---|---|---|---|---|---|---|---|
|MonoT5-large|770M|中|高|MonoT5-3B|3B|慢|很高|

## DuoT5 扩展

MonoT5 的进阶版本 **DuoT5** 做成对比较：

- 输入：query + doc_A + doc_B
- 输出："A" 或 "B"（哪个更相关）
- 通过成对比较得到更准确的排序
- 但计算量是 MonoT5 的 $O(n^2)$

## 优势

- **利用 T5 的强大语言理解**：尤其是大参数版本效果出色
- **灵活的输入格式**：可以轻松加入指令或额外上下文
- **统一框架**：与其他 T5 任务共享架构，迁移方便
- **可解释性**："true"/"false" 的概率分布提供了置信度信息

## 局限

- **编码器-解码器架构开销**：比纯编码器的 Cross-Encoder 慢
- **参数量大**：T5-3B 需要较多 GPU 显存
- **不如 Cross-Encoder 普及**：社区工具和优化不如 Cross-Encoder 丰富

## 适用场景

- 对精排精度要求极高的场景
- 已有 T5 基础设施的团队
- 需要灵活输入格式的精排任务
- 学术研究和基准测试

## 工程实现

- Hugging Face：`castorini/monot5-base-msmarco`、`castorini/monot5-3b-msmarco` 等
- Pyserini / Pygaggle：学术框架，原生支持
- 可通过 Sentence-Transformers 的 `CrossEncoder` 适配

---

**相关页面**：[Reranker（精排器）](https://www.notion.so/Reranker-c676e27a75fe4518bde38787ee24006e?pvs=21) · [Rerank-first Pipeline（先精排流水线）](https://www.notion.so/Rerank-first-Pipeline-9173ce479e5c4e0b8dcb1fa1dd337e63?pvs=21) · [Two-Stage RAG](https://www.notion.so/Two-Stage-RAG-5e5cd2078fa14b5d8e6bc4bcfafc7404?pvs=21)