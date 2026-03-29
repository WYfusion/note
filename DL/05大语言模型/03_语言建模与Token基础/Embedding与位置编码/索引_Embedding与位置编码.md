---
tags:
  - LLM/多模态
aliases:
  - 向量表示
  - Embedding
  - 文本嵌入
  - 向量检索
created: 2025-01-01
updated: 2026-03-28
---

# Embedding 与位置编码：索引

> [!abstract] 模块概览
> Embedding 是将非结构化数据映射到向量空间的核心技术。本模块涵盖从传统词向量到现代上下文化嵌入的完整体系，包括文本、视觉、音频等多种模态的表示方法，以及向量检索和位置编码技术。

## 知识地图

```
Embedding 与位置编码
├── 1文本嵌入
│   ├── 01_静态词向量_Word2Vec_GloVe_FastText.md
│   ├── 02_Dense_Bi-Encoder_句向量_SBERT_E5_BGE.md
│   ├── 03_长上下文文本嵌入_Nomic_Jina.md
│   ├── 04_查询文档区分嵌入_DPR_E5_Voyage.md
│   ├── 05_稀疏学习嵌入_SPLADE.md
│   ├── 06_多向量_Late_Interaction_ColBERT.md
│   └── 07_上下文化_Chunk_嵌入_Late_Chunking.md
├── 2多模态嵌入
│   ├── 01_视觉嵌入_Image_Embeddings.md
│   ├── 02_语音音频嵌入_Audio_Speech_Embeddings.md
│   ├── 03_视频嵌入_Video_Embeddings.md
│   └── 04_通用多模态嵌入_Multimodal.md
├── 3向量嵌入技术核心
│   ├── 01_表示层与距离度量.md
│   ├── 02_训练范式详解.md
│   ├── 03_检索视角的五类表示.md
│   └── 04_评测体系.md
└── 3向量数据库与检索引擎
    ├── 1 ANN 索引算法详解.md
    ├── 2 专用向量数据库实战.md
    ├── 3 搜索引擎向量扩展.md
    ├── 4 pgvector 关系数据库向量扩展.md
    └── 5 Faiss 算法库实战.md
```

## 核心概念

### [[索引_文本嵌入|文本嵌入]]
将文本转化为向量表示的技术，从传统词向量到现代上下文化嵌入，支持从词级到文档级的语义表示。

> [!summary] 文本表示进化史
> 从静态词向量到动态上下文化的完整技术栈。
- **静态词向量** — Word2Vec、GloVe、FastText
- **稠密句向量** — SBERT、E5、BGE、OpenAI
- **长上下文嵌入** — Nomic、Jina v2-v4
- **专业嵌入** — SPLADE、ColBERT、Late Chunking

### [[索引_多模态嵌入]] #LLM/多模态
跨越文本、图像、音频等不同模态的统一嵌入表示，实现跨模态的理解和检索。
#LLM/多模态
> [!summary] 跨模态统一表示
> 不同模态数据的嵌入技术和对齐方法。
- **视觉嵌入** — CLIP、DINO、ViT Embeddings
- **音频嵌入** — Wav2Vec、HuBERT、Whisper Embeddings
- **视频嵌入** — VideoMAE、TimeSformer
- **通用多模态** — OpenCLIP、CLIP

### [[索引_向量嵌入技术核心|向量嵌入技术核心]]
深入探讨嵌入表示的数学基础、训练方法和评估标准，是理解嵌入技术的理论基础。
> [!summary] 技术理论基础
- **表示层与距离度量** — 余弦相似度、欧氏距离
- **训练范式** — 对比学习、掩码自编码
- **五类检索表示** — Dense、Sparse、Late Interaction 等
- **评测体系** — MTEB、BEIR、Embedding Benchmark


###[[索引_向量数据库与检索引擎|向量检索]]]
高效存储和检索大规模向量数据的技术栈，包括 ANN 算法、专用数据库和传统搜索引擎扩展。
> [!summary] 工程实现方案
- **ANN 算法** — HNSW、IVF、PQ
- **专用向量数据库** — Milvus、Qdrant、Pinecone、Weaviate
- **搜索引擎扩展** — Elasticsearch、OpenSearch、Vespa
- **关系数据库扩展** — PostgreSQL + pgvector


## 前置与延伸

**前置知识**：
- [[01_BPE_WordPiece_Unigram|BPE算法，子词分词]] — Token 化基础
- [[01_绝对位置编码_Sinusoidal_Learnable|位置编码]] — 位置信息编码

**相关主题**：
-[[索引_多模态Token化|多模态 Token 化]]] — Token 级别的表示
- [[01_缩放点积注意力_为什么是点积_为什么要除以根号dk|注意力机制]] — 自注意力在嵌入中的应用

**延伸阅读**：
- [[../../11_多模态与跨模态/索引_多模态与跨模态|多模态大模型]] — 端到端多模态理解
- [[index_RAG|]] [[检索召回（Retrieval）|检索增强生成 RAG]]— 嵌入在 RAG 中的应用




