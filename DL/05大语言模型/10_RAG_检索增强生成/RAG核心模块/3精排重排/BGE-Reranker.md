## 一句话概括

BGE-Reranker 是智源研究院出品的轻量级精排模型系列，基于 Cross-Encoder 架构，中英文效果优秀，与 BGE 嵌入模型配套使用，是当前性价比最高的精排选择。

## 核心原理

### 架构

本质上仍是 Cross-Encoder 架构：

`[CLS] query [SEP] document [SEP] → Transformer → [CLS] 向量 → 线性层 → 相关性分数`

但在训练数据、训练策略和模型选型上做了大量优化，使其在多个基准上超越通用 Cross-Encoder。

### 训练策略

1. **高质量训练数据**：使用大规模、多领域的中英文标注数据
2. **Hard Negative Mining**：精心挖掘难负样本，提升模型区分能力
3. **知识蒸馏**：用更大模型的分数指导小模型训练
4. **多语言数据混合训练**：提升跨语言精排能力

## 模型版本

|版本|基础架构|参数量|特点|bge-reranker-base|BERT-base|110M|基础版，速度与精度平衡|
|---|---|---|---|---|---|---|---|
|bge-reranker-large|BERT-large|335M|高精度版|bge-reranker-v2-m3|多语言架构|568M|多语言版，100+ 语言|
|bge-reranker-v2-gemma|Gemma|2B|基于 LLM，效果顶尖|bge-reranker-v2-minicpm|MiniCPM|2.4B|LLM 版，中英文强|

### v2 版本的改进

**BGE-Reranker-v2** 引入了两个重要版本：

- **v2-m3**：与 BGE-M3 嵌入模型配套，支持多语言
- **v2-gemma / v2-minicpm**：使用 LLM 作为基础架构，精度更高但成本也更高

## 与 BGE 嵌入模型的配合

BGE 生态提供了**一站式检索方案**：

`BGE-M3（召回：Dense + Sparse + ColBERT）→ BGE-Reranker（精排）→ LLM 生成`

优势：同一生态的模型往往在数据分布和训练策略上互补，配合效果好于混搭不同来源的模型。

## 选型建议

- **性价比首选** → bge-reranker-v2-m3（多语言、效果好、速度合理）
- **中文场景** → bge-reranker-large 或 v2-minicpm
- **多语言场景** → bge-reranker-v2-m3
- **追求极致精度** → bge-reranker-v2-gemma
- **低延迟场景** → bge-reranker-base

## 使用要点

- 典型用法：召回 top-50~200 → BGE-Reranker 精排 → 保留 top-5~20
- GPU 推理更快，但 CPU 推理也可用（速度较慢）
- 支持 batch 推理加速
- 输出的分数可以通过 Sigmoid 转化为 0~1 的概率

## 工程实现

- **Hugging Face**：`BAAI/bge-reranker-v2-m3` 等
- **FlagEmbedding**：智源官方库，`FlagReranker` 类一键使用
- **LangChain / LlamaIndex**：原生集成
- **Sentence-Transformers**：`CrossEncoder` 类兼容加载

---

**相关页面**：[Reranker（精排器）](https://www.notion.so/Reranker-c676e27a75fe4518bde38787ee24006e?pvs=21) · [BGE（BAAI General Embedding）](https://www.notion.so/BGE-BAAI-General-Embedding-f552f7a45a8f475e976129af37f2e1ba?pvs=21) · [Rerank-first Pipeline（先精排流水线）](https://www.notion.so/Rerank-first-Pipeline-9173ce479e5c4e0b8dcb1fa1dd337e63?pvs=21)