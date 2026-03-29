RAG（Retrieval-Augmented Generation，检索增强生成）是一种将**外部知识检索**与**大语言模型生成**相结合的技术范式。其核心思想：不依赖模型记住所有知识，而是在生成时动态检索相关信息，让回答更准确、可溯源、可控。
## 本质
`参数知识 + 外部知识 + 推理控制`
- **参数知识**：LLM 预训练时学到的知识（可能过时或不完整）
- **外部知识**：通过检索从文档库、数据库、知识图谱等实时获取的信息
- **推理控制**：决定何时检索、如何融合、是否拒答的策略逻辑
## 核心流程

`检索 → 压缩/重排 → 生成 → 校验/引用`

---

## 📚 内容导航

本指南按以下结构分层组织，点击子页面逐层深入：

---
### 🧩 一、核心概念与模块
RAG 系统的 7 大核心模块：
- [[查询重构（Query Reformulation）]]、[[检索召回（Retrieval）]]、[[精排重排（Reranking）]]、[[4上下文压缩（Context Compression）]]
- [[5生成（Generation）]]、[[6校验与引用（Verification、Citation、Guardrails）]]、[[7记忆与反馈（Memory 、Cache、Feedback Loop）]]
## 模块总览
`Query Reformulation → Retrieval → Reranking → Context Compression → Generation → Verification → Memory`

| 模块                      | 职责    | 一句话说明                |
| ----------------------- | ----- | -------------------- |
| Query Reformulation     | 查询优化  | 把用户的模糊问题变成检索友好的精确查询  |
| Retrieval               | 检索召回  | 从知识库中找出与查询相关的候选文档片段  |
| Reranking               | 精排重排  | 对召回结果按相关性重新排序，筛选最优片段 |
| Context Compression     | 上下文压缩 | 去冗余、提纯、压缩，让模型聚焦关键信息  |
| Generation              | 生成回答  | 基于检索到的证据生成最终回答       |
| Verification / Citation | 校验与引用 | 验证回答的忠实性，绑定证据来源      |
| Memory / Cache          | 记忆与缓存 | 利用历史交互和缓存提升效率和连贯性    |
### 📖 二、经典 RAG 方案
7 种经典架构：
- [[1Naive初级 RAG]] · [[2Multi-Query RAG]] · [[3Hybrid RAG]] · [[4Two-Stage RAG]]
- [[5Step-back抽象、Rewrite重写 RAG]] · [[6Parent-Child 父子 、Hierarchical Chunk层级块 RAG]] · [[7Conversational对话 RAG]]
## 方案速览

| 方案                      | 核心流程                             | 特点                 |
| ----------------------- | -------------------------------- | ------------------ |
| Naive RAG               | query → 检索 top-k → 拼接 → LLM      | 最简单的基线方案           |
| Hybrid RAG              | BM25 + Dense + RRF 融合            | 稀疏+稠密互补，工程标配       |
| Two-Stage RAG           | 召回 → 精排 → 生成                     | 加入 Reranker 大幅提升质量 |
| Multi-Query RAG         | 扩展为多子查询并行召回                      | 覆盖面广，解决歧义          |
| Step-back / Rewrite RAG | 问题抽象化/改写后再检索                     | 提升检索召回率            |
| Parent-Child RAG        | 小块召回，大块喂给模型                      | 兼顾精确匹配和上下文完整       |
| Conversational RAG      | 历史记忆 + query rewrite + retrieval | 多轮对话场景             |
### 🔥 三、当前流行方案

这些方案是当前工程实践中最常见、最实用的技术，已被广泛验证并大规模部署。
## 方案速览

|方案|核心思路|地位|
|---|---|---|
|Hybrid Retrieval|BM25 + Dense Embedding|基本标配|
|Rerank-first Pipeline|top50~200 召回 → top5~20 精排|质量提升关键|
|Contextual Chunking|按标题/段落/语义边界切块|数据预处理核心|
|Metadata Filtering|时间/权限/标签/文档类型过滤|企业场景必备|
|Context Compression|摘要压缩/去重/片段提纯|大规模知识库必备|
|Citation RAG|答案强绑定证据片段|可信度保障|
|Agentic RAG|规划子任务 → 多轮检索 → 合成|复杂任务首选|
|Long-context RAG|大上下文窗口 + 检索协同|新兴趋势|
|Cache / Semantic Cache|相似问题复用结果|性能优化|
|Feedback RAG|点击/人工反馈驱动召回优化|持续改进闭环|
10 种工程实践最常见的方案：

- [[3Hybrid RAG]] · [Rerank-first Pipeline（先精排流水线）](https://www.notion.so/Rerank-first-Pipeline-9173ce479e5c4e0b8dcb1fa1dd337e63?pvs=21) · [Contextual Chunking（语境切块）](https://www.notion.so/Contextual-Chunking-0e81134c575a4582bc736801cf7aafb3?pvs=21) · [Metadata Filtering（元数据过滤）](https://www.notion.so/Metadata-Filtering-4148ca32955f4d91938d10a5b8ae6759?pvs=21) · [Context Compression（上下文压缩）](https://www.notion.so/Context-Compression-809ec413369a4cb4bc8858cf70b26103?pvs=21)
- [Citation RAG（引用型 RAG）](https://www.notion.so/Citation-RAG-RAG-51dc1c71be0a47238c47121e6de1bcef?pvs=21) · [Agentic RAG（智能体 RAG）](https://www.notion.so/Agentic-RAG-RAG-7e493573e2044635a7d64d6a4b16f682?pvs=21) · [Long-context RAG（长上下文 RAG）](https://www.notion.so/Long-context-RAG-RAG-075cad25b106479ab1f2e2075887fd85?pvs=21) · [Cache / Semantic Cache（语义缓存）](https://www.notion.so/Cache-Semantic-Cache-060c1476cac44263b60cce1cb269e71a?pvs=21) · [Feedback RAG（反馈驱动 RAG）](https://www.notion.so/Feedback-RAG-RAG-8cd51efe757a43e789331d0bfd3070ec?pvs=21)

### ⚙️ 四、检索组件详解
[[检索召回（Retrieval）]]
5 大类检索组件：
- [[稀疏检索]] · [[稠密检索]] · [[延迟交互（Late Interaction）|延迟交互]] · [[精排重排（Reranking）]] · [[混合检索（Hybrid Retrieval）融合策略（fusion）|融合策略]]
## 组件分类

| 类别               | 原理            | 代表模型/算法                                                 | 特点                            |
| ---------------- | ------------- | ------------------------------------------------------- | ----------------------------- |
| Sparse（稀疏检索）     | 基于词频统计的精确匹配   | BM25 / TF-IDF / SPLADE                                  | 关键词匹配强，无需 GPU                 |
| Dense（稠密检索）      | 基于语义向量的相似度匹配  | DPR / Contriever / E5 / BGE / GTE                       | 语义理解强，需向量数据库                  |
| Late Interaction | token 级延迟交互匹配 | ColBERT                                                 | 精度高于 Dense，速度快于 Cross-Encoder |
| Reranker（精排器）    | 对候选对做精细相关性打分  | Cross-Encoder / BGE-Reranker / MonoT5 / LLM-as-reranker | 精度最高，但速度慢，用于二阶段               |
| Fusion（融合策略）     | 合并多路召回结果      | RRF / Weighted Sum / Learning-to-Rank                   | 综合多路优势                        |
### 🚀 五、前沿研究方向

[前沿研究方向](https://www.notion.so/d62c4b635083442ebc1538abb6753c39?pvs=21)

11 个热门研究方向：

- [Self-RAG](https://www.notion.so/Self-RAG-cc7a2cc8ad1a4082bd393329ba9c65ee?pvs=21) · [Corrective / Adaptive RAG](https://www.notion.so/Corrective-Adaptive-RAG-23eda4759d284759b2feac52e26b720c?pvs=21) · [GraphRAG](https://www.notion.so/GraphRAG-eb9e830ebe1947eb8a539347fb561910?pvs=21) · [LightRAG](https://www.notion.so/LightRAG-df23a369561341599068e7c4402ed451?pvs=21)
- [Multi-hop RAG](https://www.notion.so/Multi-hop-RAG-2d92f4d59ac4485da80a4ccaceccc090?pvs=21) · [Agentic / Planner RAG](https://www.notion.so/Agentic-Planner-RAG-d19c59728df0427081913718b3205931?pvs=21) · [Multimodal RAG](https://www.notion.so/Multimodal-RAG-4b144f473092461f8d51049ee547b8c9?pvs=21) · [Structured RAG](https://www.notion.so/Structured-RAG-0767940275b7493c821c04ad9f65c91d?pvs=21)
- [Streaming / Real-time RAG](https://www.notion.so/Streaming-Real-time-RAG-ab592e8baad545d099a3e70398c87495?pvs=21) · [Personalized RAG](https://www.notion.so/Personalized-RAG-0615afa96ece46b7a0ab42f7af7e2077?pvs=21) · [Evaluation-centric RAG](https://www.notion.so/Evaluation-centric-RAG-2d2a0d06321842adafe9dc499135d634?pvs=21)

### 🎯 六、场景选型指南

[场景选型指南](https://www.notion.so/d74c54be84aa4d0986aac60ce230eae9?pvs=21)

不同业务场景的最佳方案推荐与一句话选型速查

### 🛠️ 七、工程实践与避坑

[工程实践与避坑](https://www.notion.so/805ac96294de409095e7486da60194fa?pvs=21)
主流综合流程、默认工程组合、常见失败点

---
## 🗺️ 一句话选型速查

|场景|推荐方案|
|---|---|
|简单问答|Hybrid RAG|
|企业文档|Hybrid + Reranker + Metadata|
|长文档复杂推理|GraphRAG / LightRAG|
|多步任务|Agentic RAG|
|高可靠输出|Citation + Verification + Refusal|
