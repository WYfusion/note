---
tags: [LLM/推理]
aliases: [多向量, Late Interaction, ColBERT, 多向量交互]
created: 2025-01-01
updated: 2026-03-28
---

# 多向量 Late Interaction：ColBERT 与细粒度交互

> [!abstract] 摘要
> 多向量 Late Interaction 方法摒弃了传统单向量压缩的思想，为每个文本保留完整的 token 级向量表示。在检索阶段进行轻量级的 token-to-token 交互，通过 MaxSim 机制实现高精度匹配。ColBERT 作为代表方法，在保持检索效率的同时，显著提升了检索质量。

## 0. 统一概念：为什么需要多向量？

> [!important] 从单向量到多向量的演进
> 传统 Dense Bi-Encoder 为整段文本生成**单个向量**，存在三个关键问题：

**单向量压缩的局限性**：
1. **信息丢失**：将复杂语义压缩到固定维度，丢失细节
2. **粒度不匹配**：短查询和长文档使用相同粒度编码
3. **位置信息抹平**：token 间的相对位置关系被忽略

**Late Interaction 的核心思想**：
> 保留所有 token 的原始向量表示，**延迟交互**到检索阶段，实现细粒度匹配

**多向量方法的优势**：
```
文本: "HNSW is an ANN algorithm"
传统: [0.2, 0.5, 0.8, -0.1, 0.3]  # 单向量
多向量: [h₁, h₂, h₃, h₄, h₅]       # 5个token向量，每个128维
```

---

## 1. 与检索方法对比

> [!important] 三种表示方式的权衡

| 方法 | 向量表示 | 检索速度 | 检索质量 | 存储开销 |
|------|----------|----------|----------|----------|
| **Bi-Encoder** | 单向量 (768维) | 极快 (1x) | 中等 | 低 (3KB/doc) |
| **Cross-Encoder** | 无预计算 | 极慢 (50x+) | 最高 | 无索引 |
| **ColBERT** | 多向量 (N×128维) | 中等 (5x) | 高 | 高 (65KB/doc) |

> [!note] 选择建议
> - **追求速度**：Bi-Encoder (第一级召回)
> - **追求质量**：ColBERT (第二级精排)
> - **最佳实践**：Bi-Encoder + ColBERT 两阶段检索

---

## 2. ColBERT 核心原理

### 2.1 架构设计

> [!important] Late Interaction 的工作流程

```
Query 编码: "What is HNSW?"
         ↓
    BERT Encoder (6层)
         ↓
    Token Vectors: [q₁, q₂, q₃, q₄]  ∈ R^(4×128)
         ↓

Document 编码: "HNSW is an approximate nearest neighbor algorithm..."
               ↓
           BERT Encoder (6层)
               ↓
       Token Vectors: [d₁, d₂, ..., d₂₀] ∈ R^(20×128)
               ↓

    MaxSim 计算:
        score(Q,D) = Σ max(q_i·d_j)  对每个 query token
```

### 2.2 MaxSim 相似度计算

> [!important] ColBERT 的评分公式

给定 query 向量集合 $Q = \{q_1, ..., q_m\}$ 和 document 向量集合 $D = \{d_1, ..., d_n\}$：

$$\text{score}(Q, D) = \sum_{i=1}^{m} \max_{j=1}^{n} \text{sim}(q_i, d_j)$$

其中 $\text{sim}(q_i, d_j) = \frac{q_i \cdot d_j}{\|q_i\| \|d_j\|}$ 是余弦相似度。

**计算优化**：
- 使用 FAISS 加速 max 相似度搜索
- 对 document 向量做 L2 归一化
- 只计算非零相似度（阈值过滤）

### 2.3 ColBERTv2 优化策略

> [!tip] 存储与精度的平衡

**残差压缩 (Residual Compression)**：
1. **聚类**：将 document token 向量聚类到 K 个中心点
2. **残差编码**：存储聚类索引 + 残差向量
3. **存储优化**：
   ```python
   # 原始存储：128维 float32 = 512 bytes/token
   original = 512 * N  # N为token数

   # 压缩后：1字节索引 + 4字节残差 = 5 bytes/token
   compressed = 5 * N
   ```

**其他改进**：
- **对抗训练**：添加噪声增强鲁棒性
- **温度缩放**：$\text{sim} = \frac{q_i \cdot d_j}{\tau}$，$\tau=0.07$
- **查询掩码**：只使用 query 中重要 token

---

## 3. Python 实战指南

### 3.1 RAGatouille（最简单实战）

> [!tip] ColBERT 最简单的使用方式 #LLM/推理

```python
from ragatouille import RAGPretrainedModel

# 加载预训练 ColBERT 模型
rag = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# 准备语料
corpus = [
    "HNSW is a graph-based approximate nearest neighbor algorithm "
    "that builds a hierarchical navigable small world graph.",
    "IVF (Inverted File Index) partitions vectors into Voronoi cells "
    "for efficient approximate nearest neighbor search.",
    "Product Quantization compresses vectors by splitting them into "
    "subvectors and quantizing each independently.",
    "Python is a popular programming language used in data science.",
]

# 建立 ColBERT 索引
index_path = rag.index(
    index_name="my_index",
    collection=corpus,
    split_documents=False,  # 不自动切分
)

# 检索
results = rag.search(query="How does graph-based ANN work?", k=3)
for r in results:
    print(f"  [{r['score']:.4f}] {r['content'][:80]}")
```

### 3.2 手动实现 MaxSim

> [!important] 理解 ColBERT 的核心计算

```python
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from faiss import IndexFlatIP

model_name = "colbert-ir/colbertv2.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

def encode_tokens(text: str, max_length: int = 128) -> np.ndarray:
    """编码文本，返回所有 token 的向量 (num_tokens, hidden_dim)"""
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=max_length, padding="max_length"
    )
    with torch.no_grad():
        outputs = model(**inputs)

    # 取 last hidden state，只保留非 padding token
    embeddings = outputs.last_hidden_state.squeeze(0)  # (seq_len, hidden_dim)
    mask = inputs["attention_mask"].squeeze(0).bool()
    token_embs = embeddings[mask].numpy()

    # L2 归一化
    norms = np.linalg.norm(token_embs, axis=1, keepdims=True)
    token_embs = token_embs / (norms + 1e-8)
    return token_embs

def maxsim_score(q_embs: np.ndarray, d_embs: np.ndarray) -> float:
    """ColBERT MaxSim 评分"""
    # q_embs: (nq, dim), d_embs: (nd, dim)
    sim_matrix = q_embs @ d_embs.T  # (nq, nd)
    # 对每个 query token 取最大相似度，然后求和
    return sim_matrix.max(axis=1).sum()

# 使用 FAISS 加速批量计算
def batch_maxsim_search(q_embs: np.ndarray, doc_embs_list: list, top_k: int = 10):
    """批量计算 MaxSim 得分"""
    # 构建文档 FAISS 索引
    dim = q_embs.shape[1]
    index = IndexFlatIP(dim)
    index.add(np.vstack(doc_embs_list))

    # 搜索
    scores, indices = index.search(q_embs, top_k)
    return scores, indices

# 编码
q = encode_tokens("What is HNSW algorithm?")
d1 = encode_tokens("HNSW builds a hierarchical graph for approximate nearest neighbor search.")
d2 = encode_tokens("Python is used for web development and data science.")

print(f"Query vs Doc1 (相关): {maxsim_score(q, d1):.4f}")
print(f"Query vs Doc2 (无关): {maxsim_score(q, d2):.4f}")
```

### 3.3 BGE-M3 多模态融合

> [!note] 同时支持 Dense + Sparse + Multi-vector #LLM/推理

```python
from FlagEmbedding import BGEM3FlagModel
import numpy as np

# BGE-M3 同时支持三种表示
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

queries = ["What is approximate nearest neighbor search?"]
docs = [
    "ANN algorithms like HNSW and IVF enable fast similarity search.",
    "Python is a versatile programming language.",
]

# 编码（同时获取三种表示）
q_output = model.encode(
    queries,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
d_output = model.encode(
    docs,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)

# 1. Dense score
dense_score = q_output["dense_vecs"] @ d_output["dense_vecs"].T
print(f"Dense scores: {dense_score[0]}")

# 2. Sparse score (SPLADE-like)
sparse_scores = model.compute_lexical_matching_score(
    q_output["lexical_weights"][0], d_output["lexical_weights"][0]
)
print(f"Sparse score (doc1): {sparse_scores}")

# 3. ColBERT multi-vector score
colbert_scores = model.compute_colbert_score(
    q_output["colbert_vecs"][0], d_output["colbert_vecs"][0]
)
print(f"ColBERT score (doc1): {colbert_scores}")
```

### 3.4 混合检索策略

> [!important] 两阶段检索的最佳实践 #LLM/推理

```python
def hybrid_colbert_search(query: str, corpus: list[str], top_k: int = 10):
    """Bi-Encoder + ColBERT 两阶段检索"""

    # 第一阶段：Bi-Encoder 召回
    from sentence_transformers import SentenceTransformer
    dense_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    q_dense = dense_model.encode([query])
    d_dense = dense_model.encode(corpus)
    dense_scores = q_dense @ d_dense.T[0]

    # 取 top-50 作为候选
    top_indices = np.argsort(dense_scores)[-50:]
    candidates = [corpus[i] for i in top_indices]

    # 第二阶段：ColBERT 精排
    colbert_rag = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    # 只对候选集建立 ColBERT 索引
    candidate_index = colbert_rag.index(
        index_name="candidate_index",
        collection=candidates,
        split_documents=False,
    )

    # ColBERT 检索
    results = colbert_rag.search(query, k=top_k)

    # 融合得分（可选）
    final_results = []
    for i, result in enumerate(results):
        doc_idx = top_indices[i]
        dense_score = dense_scores[doc_idx]
        colbert_score = result['score']

        # 加权融合
        final_score = 0.3 * dense_score + 0.7 * colbert_score
        final_results.append({
            'content': result['content'],
            'score': final_score,
            'doc_id': doc_idx
        })

    return sorted(final_results, key=lambda x: x['score'], reverse=True)
```

---

## 4. 存储优化策略

### 4.1 存储开销对比

> [!important] 多向量的存储挑战

| 方法 | 每文档存储 | 100万文档 | 压缩方法 |
|------|------------|-----------|----------|
| Dense 768维 float32 | ~3 KB | ~3 GB | - |
| ColBERT 128tokens × 128维 | ~65 KB | ~65 GB | FAISS 量化 |
| ColBERTv2 (压缩后) | ~6-10 KB | ~6-10 GB | 残差压缩 |

### 4.2 FAISS 量化技术

> [!tip] 使用 FAISS 减少存储 #LLM/推理

```python
import faiss
import numpy as np

# 原始向量
doc_vectors = np.random.randn(1000, 128).astype('float32')

# IVF 量化
nlist = 100  # 聚类中心数
quantizer = faiss.IndexFlatIP(128)
index = faiss.IndexIVFFlat(quantizer, 128, nlist, faiss.METRIC_INNER_PRODUCT)

# 训练聚类
index.train(doc_vectors)
index.add(doc_vectors)

# 搜索时解码
def search_with_quantization(query_vector: np.ndarray, index, k: int = 10):
    # 先找到最近的聚类
    D, I = index.search(query_vector.reshape(1, -1), nlist)

    # 只在相关聚类中搜索
    candidates = []
    for cluster_idx in I[0]:
        if cluster_idx != -1:  # 有效聚类
            cluster_D, cluster_I = index.search(
                query_vector.reshape(1, -1),
                k // nlist + 1
            )
            candidates.extend(zip(cluster_D[0], cluster_I[0]))

    # 重新排序
    candidates.sort(reverse=True)
    return candidates[:k]
```

---

## 5. 适用场景分析

### 5.1 推荐应用场景

| 场景 | 推荐理由 | 实现方式 |
|------|----------|----------|
| **高精度检索** | 显著优于单向量 | ColBERT 作为精排层 |
| **问答/RAG** | 捕捉局部匹配 | 替代 Cross-Encoder |
| **短查询匹配长文档** | token 级别交互 | MaxSim 机制 |
| **代码检索** | 语法结构匹配 | 特殊 token 处理 |

### 5.2 性能基准测试

> [!important] MS MARCO 上的性能表现

| 方法 | Recall@100 | NDCG@10 | 查询时间 |
|------|------------|---------|----------|
| **BM25** | 0.182 | 0.312 | 1.0x |
| **Dense Bi-Encoder** | 0.215 | 0.346 | 1.2x |
| **ColBERT** | 0.248 | 0.398 | 5.3x |
| **ColBERTv2** | 0.245 | 0.395 | 4.1x |

> [!note] 测试环境
> - 数据集：MS MARCO dev
> - 索引：10M 文档
> - 硬件：32核 CPU + 128GB RAM

---

## 6. 工程实践要点

### 6.1 支持的向量库

> [!important] ColBERT 原生支持的方案

| 向量库 | 支持程度 | 实现方式 |
|--------|----------|----------|
| **Vespa** | ✅ 原生支持 | 内置 ColBERT 索引 |
| **Qdrant** | ✅ Named Vectors | 多向量字段 |
| **Milvus** | ✅ 支持多向量 | 动态字段 |
| **Pinecone** | ❌ 不支持 | 需要 workaround |

### 6.2 缓存优化

> [!tip] Query 编码缓存 #LLM/推理

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=10000)
def cached_encode(query: str, model_name: str):
    """带缓存的 query 编码"""
    cache_key = hashlib.md5((query + model_name).encode()).hexdigest()
    # 实际编码逻辑
    return encode_tokens(query)

# 批处理优化
def batch_colbert_encode(queries: list[str], batch_size: int = 32):
    """批量编码提高效率"""
    results = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        for query in batch:
            results.append(cached_encode(query, "colbert-ir/colbertv2.0"))
    return results
```

---

## 相关链接

**所属模块**：[[索引_Embedding与位置编码]]

**前置知识**：
- [[2 Dense Bi-Encoder 句向量 (SBERT E5 BGE OpenAI)|稠密句向量]] — 理解单向量编码
- [[5 稀疏学习嵌入 SPLADE|稀疏学习嵌入]] — 理解词汇级匹配

**相关主题**：
- [[7 上下文化 Chunk 嵌入 (Late Chunking Contextual Retrrieval)|Late Chunking]] — 增强检索策略
- [[../../3向量嵌入技术核心/02_训练范式详解.md|训练范式详解]] — Late Interaction 训练
- [[../../4向量数据库与检索引擎/索引_向量数据库|向量数据库]] — 存储和检索

**延伸阅读**：
- [[../../../Embedding应用/RAG/index_RAG|检索增强生成 RAG]] — 完整应用案例
- [ColBERT 论文原文](https://arxiv.org/abs/2004.12832) — 技术细节
