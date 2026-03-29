## 是什么
对检索召回的候选文档按相关性重新打分排序，筛选出最相关的 top-K 结果。类比：初筛后的面试官精选。
## 为什么重要
- 召回阶段追求**高召回率**（宁可多不可少），结果中噪声多
- 精排阶段追求**高精度**（只留最好的），是质量提升的关键步骤
- 只召回不精排是最常见的 RAG 失败原因之一
## 核心原理
`(query, doc) → 模型 → 相关性分数`
与检索模型的区别：检索模型独立编码 query 和 doc，而 Reranker 将两者一起输入做交叉注意力，精度更高但速度更慢。

## 代表模型

### [[Cross-Encoder]]
最经典的精排方式。将 query 和 doc 拼接后输入 BERT，输出一个相关性分数。
将 query 和 document 拼接后一起输入模型打分。精度最高，但速度慢（每对都要计算一次）。
- **精度**：最高（全交叉注意力）
- **速度**：最慢（每对都要完整前向传播）
- **代表**：ms-marco-MiniLM、bge-reranker-base

### [[BGE-Reranker]]
智源研究院出品的轻量级精排模型。中英文效果好，速度比传统 Cross-Encoder 快。
基于 BGE 系列的轻量级精排模型。精度接近 Cross-Encoder，速度更快。
- **BGE-Reranker-v2**：性价比最高的选择
- **BGE-Reranker-v2-m3**：多语言支持

### [[MonoT5]]
基于 T5 的生成式精排模型。将排序问题转化为文本生成：输入 query+doc，输出 "true" 或 "false"。

### [[LLM-as-Reranker]]
直接用 GPT 等大语言模型判断相关性。最灵活，可以理解复杂查询意图，但成本最高。

- **方式**：给 LLM 候选列表，让它排序或打分
- **适用**：其他 Reranker 效果不足时的兜底方案

## 选型对比
- **性价比首选** → BGE-Reranker-v2
- **最高精度** → Cross-Encoder（大模型）
- **多语言** → BGE-Reranker-v2-m3
- **最灵活** → LLM-as-Reranker
- **API 服务** → Cohere Rerank

## 使用要点
- 召回 50~200 → 精排保留 5~20
- GPU 密集型，注意延迟预算
- 可做 batch scoring 加速

## 典型流程
`召回 top-50~200 → Reranker 精排 → 取 top-5~20 → 送入生成`

## 与其他模块的关系
- **输入** ← 接收 Retrieval 的召回结果
- **输出** → 将精排结果传递给 Context Compression