## 一句话概括

SPLADE（SParse Lexical AnD Expansion model,稀疏词汇和扩展模型）是一种学习型稀疏检索模型，用预训练语言模型（如 BERT）为每个词预测权重，同时能自动扩展出语义相关的词，兼具 BM25 的速度和 Dense 的语义理解。

## 核心原理

### 问题背景

BM25 只做精确词面匹配，不理解语义。Dense Retrieval 理解语义但速度慢且需要向量数据库。SPLADE 的目标是在稀疏检索的框架内引入语义能力。

### 关键思想

1. **用 BERT 预测词权重**：不再用简单的词频统计，而是让 BERT 学习每个词在当前上下文中的重要性权重
2. **词项扩展（Term Expansion）**：模型还能为文档/查询「想到」原文中没有出现但语义相关的词。比如文档提到「汽车」，SPLADE 可以自动激活「轿车」「车辆」等词

### 技术流程

`文本 → BERT 编码 → MLM Head → 词汇表维度的 logits → ReLU + log(1+x) 激活 → 取 max pooling → 稀疏向量`

逐步拆解：

1. 输入文本经过 BERT，得到每个 token 在词汇表上的 logits（MLM 预测头）
2. 对 logits 做 `log(1 + ReLU(x))` 激活，得到非负权重
3. 对所有 token 位置取 max pooling，得到一个词汇表大小的稀疏向量
4. 大部分维度接近 0（稀疏），非零维度对应「重要的词」

### 为什么能扩展词项？

BERT 的 MLM Head 本来就是预测「这个位置可能是什么词」。如果文档提到「苹果手机」，MLM Head 在对应位置可能给「iPhone」「智能手机」等词也打出较高分数 → 这些词就被自动激活到稀疏向量中。

## 稀疏性控制

SPLADE 通过正则化项控制向量的稀疏度：

$L = L_{rank} + \lambda \cdot L_{sparsity}$

- $L_{rank}$：排序损失（让相关文档排在前面）
- $L_{sparsity}$：稀疏正则化（鼓励向量中更多维度为 0）
- $\lambda$ 越大 → 越稀疏 → 越快但可能丢信息

## 与其他方法对比

|特性|BM25|SPLADE|Dense|
|---|---|---|---|
|词项扩展|❌ 无|✅ 自动扩展|N/A|
|推理速度|最快|快（倒排索引）|中（ANN 检索）|
|是否需要 GPU|❌|编码时需要|编码时需要|

## 版本演进

- **SPLADE v1**：基础版本
- **SPLADE v2 / SPLADEv2**：改进训练策略，效果更好
- **SPLADE++**：引入蒸馏和更好的负采样
- **Efficient SPLADE**：更稀疏的版本，适合大规模部署

## 适用场景

- 需要语义理解但又想用倒排索引的场景
- 已有 Elasticsearch 基础设施，想加入语义能力
- 对可解释性有要求的检索系统

## 工程实现

- Hugging Face 上有预训练模型（`naver/splade-cocondenser-ensembledistil` 等）
- 可以配合 Elasticsearch / Vespa 使用（支持稀疏向量）
- PISA、Anserini 等学术检索引擎支持

---

**相关页面**：[Sparse Retrieval（稀疏检索）](https://www.notion.so/Sparse-Retrieval-dcc60a4e987541589511862507f5e2a7?pvs=21) · [Dense Retrieval（稠密检索）](https://www.notion.so/Dense-Retrieval-8263fa030b7740a98bc72321e43033cf?pvs=21) · [Hybrid Retrieval（混合检索）](https://www.notion.so/Hybrid-Retrieval-27d82bb8f71a49fc909733f2f2a4c761?pvs=21)