---
tags: [LLM/推理]
aliases: [查询文档区分嵌入, Query-Distinct Embeddings, Input Type Optimization]
created: 2025-01-01
updated: 2026-03-28
---

# 查询文档区分嵌入：DPR、E5、Voyage

> [!abstract] 摘要
> 查询文档区分嵌入通过识别 query 和 document 的语言分布差异，使用不同的编码策略优化检索效果。这种方法认识到用户查询通常是短问句，而文档是长段落，需要区别对待以获得更好的语义匹配。

## 0. 统一概念：为什么需要区分查询和文档？

> [!important] 传统方法的局限性
> 传统 Bi-Encoder 对 query 和 document 使用完全相同的编码方式，但存在两个关键问题：

1. **语言分布不对称**：
   - Query：短小精悍，信息密度高，意图明确
   - Document：冗长详细，信息分散，背景丰富

2. **语义理解差异**：
   - Query 需要准确把握用户意图
   - Document 需要全面捕获主题信息

**核心思想**：**让模型知道当前输入是"问题"还是"文档"**

> [!warning] 混合编码的问题
> 如果将长文档和短查询用相同方式编码，模型可能会：
> - 过分关注文档的局部细节
> - 忽略查询的核心意图
> - 产生不相关的匹配结果

---

## 1. 三种实现范式

### 1.1 独立双编码器（DPR 风格）

> [!important] DPR：开创性的双塔架构

**DPR**（Dense Passage Retrieval, Facebook 2020）使用两个独立的 BERT 编码器：

```
Query: "What is HNSW?"        Doc: "HNSW is a graph-based ANN index..."
       ↓                              ↓
  Question Encoder (BERT)       Passage Encoder (BERT)
       ↓                              ↓
    q ∈ R^768                      d ∈ R^768
       ↓                              ↓
              sim(q, d) = q · d
```

**关键特性**：
- **参数隔离**：两个编码器有独立参数
- **空间分离**：查询和文档在不同子空间
- **训练优化**：专门针对检索任务设计

### 1.2 共享编码器 + 前缀区分（E5 / BGE 风格）

> [!tip] 更高效的实现方式

使用同一个模型，但通过不同前缀告知角色：

| 模型 | Query 前缀 | Document 前缀 | 效果提升 |
|------|------------|---------------|----------|
| **E5** | `"query: "` | `"passage: "` | +3-5% Recall |
| **BGE** | `"Represent...: "` | 无前缀 | +2-4% NDCG |
| **Nomic** | `"search_query: "` | `"search_document: "` | +4-6% MRR |

### 1.3 API 参数区分（OpenAI / Voyage 风格）

> [!note] 商业 API 的优化策略 #LLM/推理

通过 API 参数 `input_type` 指定：

```python
# Voyage API
vo.embed(texts, model="voyage-3", input_type="query")  # 查询
vo.embed(texts, model="voyage-3", input_type="document")  # 文档

# OpenAI 内部优化（虽无显式参数）
# 会自动识别 query/document 并调整编码策略
```

---

## 2. DPR 详细实现

### 2.1 DPR 架构设计

> [!important] 双塔架构的数学基础

**Question Encoder**：
$$q = \text{BERT}_{\text{question}}(\text{input}) \in \mathbb{R}^d$$

**Passage Encoder**：
$$d = \text{BERT}_{\text{passage}}(\text{input}) \in \mathbb{R}^d$$

**相似度计算**：
$$\text{sim}(q, d) = q \cdot d \quad \text{（点积相似度）}$$

### 2.2 训练损失函数

> [!important] InfoNCE 损失的实现

$$\mathcal{L} = -\log \frac{\exp(q_i \cdot d_i^+)}{\exp(q_i \cdot d_i^+) + \sum_{j} \exp(q_i \cdot d_j^-)}$$

其中：
- $(q_i, d_i^+)$ 是正样本对
- $\{d_j^-\}$ 是负样本集合
- 关键：**Hard Negative Mining**

### 2.3 Hard Negative Mining

> [!tip] 难负例的挖掘策略

**难负例的定义**：
- BM25 检索出的高分但不相关段落
- 语义相近但不匹配的文档
- 硬负例的重要性

```python
# 难负例挖掘示例
from rank_bm25 import BM25Okapi
import numpy as np

def mine_hard_negatives(query, corpus, k=10):
    """使用 BM25 挖掘难负例"""
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query)
    top_indices = np.argsort(scores)[-k:]
    return [corpus[i] for i in top_indices]

# 训练数据构建
positive_pairs = [(query, relevant_doc)]
hard_negatives = mine_hard_negatives(query, all_corpus)
negative_samples = hard_negatives  # 难负例作为负样本
```

---

## 3. Python 实战指南

### 3.1 使用 DPR 原始模型（Hugging Face）

> [!warning] 注意模型版本兼容性 #LLM/推理

```python
from transformers import DPRQuestionEncoder, DPRContextEncoder
from transformers import DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
import torch
import numpy as np

# 加载 question 和 passage 编码器
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base"
)
q_model = DPRQuestionEncoder.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base"
)

p_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base"
)
p_model = DPRContextEncoder.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base"
)

# 编码 query
q_input = q_tokenizer("What is HNSW index?", return_tensors="pt")
with torch.no_grad():
    q_emb = q_model(**q_input).pooler_output.numpy()  # (1, 768)

# 编码 passages
passages = [
    "HNSW is a graph-based approximate nearest neighbor algorithm used in vector databases.",
    "Python is a popular programming language for data science.",
    "HNSW builds a hierarchical graph with multiple layers for efficient search.",
]
p_embs = []
for p in passages:
    p_input = p_tokenizer(p, return_tensors="pt")
    with torch.no_grad():
        emb = p_model(**p_input).pooler_output.numpy()
    p_embs.append(emb)
p_embs = np.vstack(p_embs)  # (3, 768)

# 计算相似度（DPR 用点积）
scores = (q_emb @ p_embs.T)[0]
for i, s in enumerate(scores):
    print(f"  Passage {i}: {s:.4f} — {passages[i][:50]}")
```

### 3.2 E5 query/passage 前缀

> [!important] E5 的前缀规范

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("intfloat/e5-base-v2")

# 关键：query 和 passage 必须使用不同前缀
queries = ["query: How does vector search work?"]
passages = [
    "passage: Vector search uses approximate nearest neighbor algorithms to find similar embeddings.",
    "passage: Machine learning models can be trained on GPU clusters.",
]

q_emb = model.encode(queries, normalize_embeddings=True)
p_emb = model.encode(passages, normalize_embeddings=True)

scores = q_emb @ p_emb.T
print(f"Query vs Passage1: {scores[0][0]:.4f}")  # 高
print(f"Query vs Passage2: {scores[0][1]:.4f}")  # 低

# ⚠️ 错误用法：不加前缀或前缀混用会导致检索效果大幅下降！
wrong_queries = ["How does vector search work?"]
wrong_scores = model.encode(wrong_queries) @ p_emb.T
print(f"不加前缀的得分: {wrong_scores[0][0]:.4f}")  # 会显著降低
```

### 3.3 Voyage API input_type

> [!note] Voyage 的最新优化（2024） #LLM/推理

```python
import voyageai
import numpy as np

vo = voyageai.Client()  # 需设置 VOYAGE_API_KEY

# 编码 query（设置 input_type）
q_result = vo.embed(
    ["What are the best vector databases?"],
    model="voyage-3",
    input_type="query",
)
q_emb = np.array(q_result.embeddings)

# 编码 documents
d_result = vo.embed(
    [
        "Qdrant is an open-source vector search engine with rich filtering.",
        "Pinecone is a managed vector database for production workloads.",
        "PostgreSQL is a relational database management system.",
    ],
    model="voyage-3",
    input_type="document",
)
d_emb = np.array(d_result.embeddings)

scores = q_emb @ d_emb.T
for i, s in enumerate(scores[0]):
    print(f"  Doc {i}: {s:.4f}")
```

---

## 4. 为什么区分有效？

### 4.1 统计差异分析

| 统计特征 | Query | Document | 差异程度 |
|----------|-------|----------|----------|
| **平均长度** | 10-20 tokens | 100-500 tokens | 10-50x |
| **词汇密度** | 高 (信息密集) | 中等 (冗余较多) | 显著 |
| **句式结构** | 疑问句、祈使句 | 陈述句、说明句 | 不同 |
| **主题范围** | 狭窄 (具体问题) | 宽泛 (主题覆盖) | 不同 |

### 4.2 注意力机制差异

> [!tip] 不同角色的注意力策略

**Query 编码的注意力**：
- 关注问题的核心关键词
- 识别问题的意图类别（是什么、为什么、怎么做）
- 问题的前置和后缀信息同样重要

**Document 编码的注意力**：
- 关注文档的主题句
- 重要的段落标题和首尾句
- 术语定义和核心概念

### 4.3 实测效果提升

> [!important] 区分嵌入的性能提升

| 模型 | 不区分前缀 | 使用前缀 | 提升 |
|------|------------|----------|------|
| **E5-base** | 68.2% | 73.5% | +5.3% |
| **BGE-base** | 70.1% | 74.8% | +4.7% |
| **Voyage-2** | 75.3% | 78.9% | +3.6% |

> [!note] 提升分析
> - **E5**：前缀策略最有效，因为设计了专门的 query/passage 任务
> - **BGE**：instruction 前缀提供了更好的意图识别
> - **Voyage**：商业 API 优化，效果稳定

---

## 5. 实践要点与常见错误

### 5.1 最常见错误

> [!warning] 一定要避免的错误

```python
# ❌ 错误示例 1：忘记加前缀
queries = ["How to use HNSW?"]  # 没有前缀
documents = ["HNSW is an algorithm..."]  # 没有前缀

# ✅ 正确做法
queries = ["query: How to use HNSW?"]  # 必须加前缀
documents = ["passage: HNSW is an algorithm..."]  # 必须加对应前缀

# ❌ 错误示例 2：前缀混用
# query 用 query:, document 用 search_document: (错误)
```

### 5.2 最佳实践

> [!tip] 生产环境配置建议

1. **离线索引阶段**：
   ```python
   # 索引文档时
   corpus_embeddings = model.encode(
       [f"passage: {text}" for text in corpus],
       normalize_embeddings=True,
       batch_size=256,
   )
   ```

2. **在线检索阶段**：
   ```python
   # 检索时
   query_embedding = model.encode(
       [f"query: {user_query}"],
       normalize_embeddings=True
   )
   ```

3. **配置管理**：
   ```python
   class EmbeddingConfig:
       QUERY_PREFIX = "query: "
       DOCUMENT_PREFIX = "passage: "

       def encode_query(self, text):
           return model.encode([self.QUERY_PREFIX + text])

       def encode_document(self, text):
           return model.encode([self.DOCUMENT_PREFIX + text])
   ```

---

## 6. 性能优化策略

### 6.1 多模型融合

> [!important] 混合多种嵌入方法

```python
def hybrid_query_embedding(query, models):
    """融合多个模型的查询嵌入"""
    embeddings = []
    for model in models:
        if "e5" in model.name:
            emb = model.encode([f"query: {query}"])
        elif "bge" in model.name:
            emb = model.encode([f"Represent this sentence for searching relevant passages: {query}"])
        else:
            emb = model.encode([query])
        embeddings.append(emb)
    return np.mean(embeddings, axis=0)
```

### 6.2 缓存优化

> [!tip] 智能缓存策略

```python
from functools import lru_cache
import hashlib

def get_embedding_cache_key(text, prefix=""):
    """生成缓存键"""
    content = prefix + text
    return hashlib.md5(content.encode()).hexdigest()

@lru_cache(maxsize=10000)
def cached_encode_with_prefix(cache_key):
    """带前缀的编码缓存"""
    # 实际编码逻辑
    return embedding

def smart_encode(text, prefix="", cache=True):
    """智能编码，自动使用缓存"""
    if cache:
        cache_key = get_embedding_cache_key(text, prefix)
        return cached_encode_with_prefix(cache_key)
    else:
        if prefix:
            return model.encode([prefix + text])
        return model.encode([text])
```

---

## 相关链接

**所属模块**：[[索引_Embedding与位置编码]]

**前置知识**：
- [[2 Dense Bi-Encoder 句向量 (SBERT E5 BGE OpenAI)|稠密句向量]] — 理解基础嵌入方法
- [[../../04_Transformer核心结构/02_注意力机制/01_缩放点积注意力_为什么是点积_为什么要除以根号dk|注意力机制]] — 自注意力原理

**相关主题**：
- [[3 长上下文文本嵌入 (Nomic Jina v2-v4)|长上下文嵌入]] — 处理长文档
- [[../../4向量数据库与检索引擎/索引_向量数据库|向量数据库]] — 存储和检索优化
-[[02_Transformer与注意力革命|Transformer 革命]]] — 架构演进

**延伸阅读**：
- [[../../3向量嵌入技术核心/02_训练范式详解.md|训练范式详解]] — 对比学习原理
- [[../../3向量嵌入技术核心/03_检索视角的五类表示.md|五类检索表示]] — 更多嵌入方法

