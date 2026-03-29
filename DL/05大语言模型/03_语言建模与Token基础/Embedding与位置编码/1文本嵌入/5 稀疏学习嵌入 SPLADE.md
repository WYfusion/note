---
tags: [LLM/推理]
aliases: [SPLADE, 稀疏嵌入, 学习型稀疏检索, Lexical Expansion]
created: 2025-01-01
updated: 2026-03-28
---

# 稀疏学习嵌入：SPLADE

> [!abstract] 摘要
> SPLADE（SParse Lexical And Expansion model）是一种创新的学习型稀疏嵌入方法，结合了传统稀疏检索的可解释性和神经网络的语义理解能力。它通过 BERT 为每个文档生成高维稀疏向量，实现词项自动扩展，让稀疏检索也能捕捉语义关联。

## 0. 统一概念：SPLADE 的革命性创新

> [!important] 从传统 BM25 到学习型稀疏检索

**传统稀疏检索的局限性**：
- **精确词匹配**：只能匹配完全相同的词
- **无语义理解**：无法处理同义词和语义相关
- **词干合并限制**：基于规则的词干提取不够智能

**SPLADE 的核心思想**：
> 用 BERT 为每个文档/查询生成一个**高维稀疏向量**，维度对应词表中的词项（term），非零值表示该词项的重要性权重。模型会自动进行**词项扩展**——即使原文没有出现某个词，模型也可能为语义相关的词赋予非零权重。

**词项扩展示例**：
```
输入: "deep learning"
原文词项: deep(2.1), learning(1.8)
扩展词项: neural(1.2), network(0.9), ai(0.7), machine(0.6)
```

---

## 1. 与其他嵌入方法的对比

> [!important] 三种表示方式的权衡

| 方法 | 向量类型 | 维度 | 可解释性 | 语义能力 | 计算效率 |
|------|----------|------|----------|----------|----------|
| **BM25** | 稀疏（词频统计） | 词表大小(~30K) | ✅ 高 | ❌ 无 | 极高 |
| **SPLADE** | 稀疏（学习型） | 词表大小(~30K) | ✅ 高 | ✅ 有（词项扩展） | 高 |
| **Dense Embedding** | 稠密 | 768-1024 | ❌ 低 | ✅ 强 | 中等 |

> [!note] 选择建议
> - **追求可解释性**：选择 SPLADE 或 BM25
> - **追求语义精度**：选择 Dense Embedding
> - **最佳实践**：SPLADE + Dense 混合检索

---

## 2. SPLADE 核心原理

### 2.1 架构设计

> [!important] SPLADE 的处理流程

```
输入文本: "What is HNSW index?"
         ↓
    BERT Encoder
         ↓
  每个 token 的 MLM logits → (seq_len, vocab_size)
         ↓
  对 seq_len 维度取 max pooling
         ↓
  ReLU + log(1+x) 激活
         ↓
  稀疏向量 ∈ R^vocab_size  (大部分为 0)
```

### 2.2 关键数学公式

> [!important] SPLADE 的生成公式

给定输入 token 序列 $t_1, \ldots, t_n$，SPLADE 输出稀疏向量 $s$：

$$s_j = \max_{i=1}^{n} \log(1 + \text{ReLU}(w_j^i))$$

其中：
- $w_j^i$ 是第 $i$ 个 token 在词表第 $j$ 个位置的 MLM logit
- $\text{ReLU}(x) = \max(0, x)$ 是激活函数
- $\log(1 + x)$ 是对数压缩函数

### 2.3 稀疏度控制

> [!tip] FLOPS 正则化的作用

SPLADE 使用 **FLOPS 正则化**控制稀疏度：

$$\mathcal{L}_{\text{reg}} = \lambda \sum_{j} \bar{a}_j^2$$

其中：
- $\bar{a}_j$ 是词项 $j$ 在 batch 内的平均激活值
- $\lambda$ 是正则化系数（默认 0.1）
- $\lambda$ 越大，向量越稀疏（检索越快但精度可能下降）

---

## 3. Python 实战指南

### 3.1 使用 Hugging Face 推理

> [!tip] SPLADE 模型的选择 #LLM/推理

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np

# 选择不同的 SPLADE 模型
model_name = "naver/splade-cocondenser-ensembledistil"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.eval()

def splade_encode(text: str) -> dict[str, float]:
    """将文本编码为 SPLADE 稀疏向量（词项 → 权重）"""
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**tokens)

    # SPLADE: max pooling over sequence + log(1 + ReLU(x))
    logits = output.logits  # (1, seq_len, vocab_size)
    sparse_vec = torch.max(
        torch.log(1 + torch.relu(logits)) * tokens["attention_mask"].unsqueeze(-1),
        dim=1
    ).values.squeeze()  # (vocab_size,)

    # 提取非零词项
    nonzero_idx = sparse_vec.nonzero().squeeze().tolist()
    if isinstance(nonzero_idx, int):
        nonzero_idx = [nonzero_idx]

    result = {}
    for idx in nonzero_idx:
        token = tokenizer.decode([idx])
        weight = sparse_vec[idx].item()
        result[token] = round(weight, 4)

    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

# 编码示例
query_sparse = splade_encode("What is vector database?")
print("Top-10 query 词项:")
for term, weight in list(query_sparse.items())[:10]:
    print(f"  {term}: {weight}")

doc_sparse = splade_encode(
    "A vector database stores high-dimensional embeddings and "
    "supports approximate nearest neighbor search for retrieval."
)
print(f"\n文档非零词项数: {len(doc_sparse)}")
```

### 3.2 稀疏向量相似度计算

> [!important] 稀疏向量的高效计算

```python
def sparse_dot_product(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """两个稀疏向量的点积（只计算非零项）"""
    score = 0.0
    for term, weight_a in vec_a.items():
        if term in vec_b:
            score += weight_a * vec_b[term]
    return score

def cosine_similarity_sparse(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """稀疏向量的余弦相似度"""
    dot_product = sparse_dot_product(vec_a, vec_b)

    # 计算模长
    norm_a = np.sqrt(sum(w**2 for w in vec_a.values()))
    norm_b = np.sqrt(sum(w**2 for w in vec_b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

# 测试示例
q = splade_encode("How does HNSW work?")
d1 = splade_encode("HNSW is a graph-based approximate nearest neighbor algorithm.")
d2 = splade_encode("Python is a popular programming language.")

print(f"Query vs Doc1 (相关): {sparse_dot_product(q, d1):.4f}")
print(f"Query vs Doc2 (无关): {sparse_dot_product(q, d2):.4f}")
print(f"Query vs Doc1 (余弦): {cosine_similarity_sparse(q, d1):.4f}")
```

### 3.3 混合检索策略

> [!tip] SPLADE + Dense 的最佳实践 #LLM/推理

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# 加载 Dense 模型
dense_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

def hybrid_search(
    query: str,
    corpus: list[str],
    alpha: float = 0.7,  # dense 权重
    normalize: bool = True,
) -> list[tuple[int, float]]:
    """Hybrid = alpha * dense_score + (1-alpha) * sparse_score"""

    # Dense scores
    q_dense = dense_model.encode([query], normalize_embeddings=normalize)
    d_dense = dense_model.encode(corpus, normalize_embeddings=normalize)
    dense_scores = (q_dense @ d_dense.T)[0]

    # Sparse scores (SPLADE)
    q_sparse = splade_encode(query)
    sparse_scores = []
    for doc in corpus:
        d_sparse = splade_encode(doc)
        sparse_scores.append(sparse_dot_product(q_sparse, d_sparse))
    sparse_scores = np.array(sparse_scores)

    # 归一化到 [0, 1] 区间
    if normalize:
        if dense_scores.max() > 0:
            dense_scores = dense_scores / dense_scores.max()
        if sparse_scores.max() > 0:
            sparse_scores = sparse_scores / sparse_scores.max()

    # 混合策略
    hybrid_scores = alpha * dense_scores + (1 - alpha) * sparse_scores

    # 排序并返回结果
    ranked = sorted(enumerate(hybrid_scores), key=lambda x: x[1], reverse=True)
    return ranked

# 示例使用
corpus = [
    "HNSW builds a hierarchical graph for fast nearest neighbor search.",
    "IVF partitions vectors into clusters for efficient retrieval.",
    "Python is great for machine learning development.",
]

# 纯 SPLADE 检索
print("=== 纯 SPLADE 检索 ===")
results = hybrid_search("How does approximate nearest neighbor work?", corpus, alpha=0.0)
for idx, score in results[:3]:
    print(f"  [{score:.4f}] {corpus[idx][:60]}")

# 混合检索（70% Dense + 30% SPLADE）
print("\n=== 混合检索 (70% Dense + 30% SPLADE) ===")
results = hybrid_search("How does approximate nearest neighbor work?", corpus, alpha=0.7)
for idx, score in results[:3]:
    print(f"  [{score:.4f}] {corpus[idx][:60]}")
```

---

## 4. SPLADE 的变体与演进

### 4.1 SPLADE 系列

> [!important] SPLADE 的主要版本

| 版本 | 特点 | 改进点 |
|------|------|--------|
| **SPLADE** | 基础版本 | 原始实现 |
| **SPLADE++** | 多任务预训练 | 增强了词项扩展能力 |
| **CoCondenser** | 集成蒸馏 | 更快、更小、更准确 |
| **Ensemble** | 多模型集成 | 提升稳定性 |

### 4.2 其他稀疏嵌入方法

| 方法 | 核心思想 | 特点 |
|------|----------|------|
| **BERT sparse** | 直接使用 BERT 输出 | 简单但效果一般 |
| **DeepCT** | 对数变换的词频 | 结合统计与深度学习 |
| **Word2Vec sparse** | 词向量稀疏化 | 语义扩展有限 |

---

## 5. 实际应用场景

### 5.1 推荐应用场景

| 场景 | 推荐理由 | 实现方式 |
|------|----------|----------|
| **首阶段召回** | 替代 BM25，语义更强 | SPLADE 索引 + 倒排 |
| **Hybrid 检索** | 与 Dense 互补 | SPLADE + Dense 混合 |
| **可解释性要求高** | 可查看词项贡献 | SPLADE 稀疏向量分析 |
| **大规模检索** | 稀疏索引高效 | 专门的稀疏向量库 |

### 5.2 性能基准测试

> [!important] SPLADE 在 MS MARCO 上的表现

| 方法 | Recall@100 | NDCG@10 | 查询时间 |
|------|------------|---------|----------|
| **BM25** | 0.182 | 0.312 | 1.0x |
| **SPLADE** | 0.215 | 0.346 | 1.2x |
| **Dense** | 0.238 | 0.389 | 2.5x |
| **Hybrid** | 0.251 | 0.412 | 1.8x > [!note] 测试环境
> - 数据集：MS MARCO dev
> - 索引：10M 文档
> - 硬件：32核 CPU + 128GB RAM

---

## 6. 工程实践要点

### 6.1 稀疏向量存储

> [!tip] 稀疏向量的高效存储

```python
import json
from collections import defaultdict

class SparseVectorIndex:
    def __init__(self, vocab_mapping=None):
        self.doc_vectors = {}  # doc_id -> sparse_vector
        self.inverted_index = defaultdict(dict)  # term -> doc_id -> weight
        self.vocab_mapping = vocab_mapping or {}

    def add_document(self, doc_id, sparse_vector):
        """添加文档到索引"""
        self.doc_vectors[doc_id] = sparse_vector

        # 更新倒排索引
        for term, weight in sparse_vector.items():
            self.inverted_index[term][doc_id] = weight

    def search(self, query_sparse, top_k=10):
        """使用稀疏向量检索"""
        scores = defaultdict(float)

        # 计算得分
        for term, q_weight in query_sparse.items():
            if term in self.inverted_index:
                for doc_id, d_weight in self.inverted_index[term].items():
                    scores[doc_id] += q_weight * d_weight

        # 排序并返回结果
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return ranked_docs

# 使用示例
index = SparseVectorIndex()
index.add_document("doc1", splade_encode("HNSW is an algorithm"))
index.add_document("doc2", splade_encode("Python is a language"))

results = index.search(splade_encode("How does HNSW work?"))
print(results)
```

### 6.2 缓存优化

> [!important] SPLADE 的缓存策略 #LLM/推理

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=10000)
def cached_splade_encode(text: str, model_name: str):
    """带缓存的 SPLADE 编码"""
    cache_key = hashlib.md5((text + model_name).encode()).hexdigest()
    return splade_encode(text)

# 批处理优化
def batch_splade_encode(texts: list[str], batch_size: int = 32):
    """批量编码以提高效率"""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        for text in batch:
            results.append(cached_splade_encode(text, "naver/splade-cocondenser-ensembledistil"))
    return results
```

---

## 相关链接

**所属模块**：[[索引_Embedding与位置编码]]

**前置知识**：
- [[4 查询 文档区分嵌入 (DPR E5 Voyage input_type)|查询文档区分嵌入]] — 理解查询优化
- [[../../04_Transformer核心结构/模型家族/01_Encoder_Only_BERT_Wav2Vec|BERT 架构]] — MLM 原理

**相关主题**：
- [[6 多向量 Late Interaction (ColBERT ColBERTv2)|多向量交互]] — ColBERT 也是稀疏方法
- [[../../4向量数据库与检索引擎/02_专用向量数据库_Milvus_Weaviate|向量数据库]] — 稀疏向量存储
- [[../../3向量嵌入技术核心/03_检索视角的五类表示.md|五类检索表示]] — 更多方法对比

**延伸阅读**：
- [[../../../11_多模态与跨模态/索引_多模态与跨模态|多模态大模型]] — 跨模态稀疏表示
- [[../../../Embedding应用/RAG/index_RAG|检索增强生成 RAG]] — RAG 中的混合检索
