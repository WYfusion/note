## 概述

传统 Dense Embedding 为每段文本生成**一个向量**，压缩了所有语义信息。**多向量 / Late Interaction** 方法为每段文本保留**多个 token 级向量**，检索时通过 token 间的细粒度交互计算相似度，显著提升匹配质量。

> 核心思想：**不把整段文本压缩成一个点，而是保留每个 token 的向量表示**，在检索阶段做轻量级的 token-to-token 交互。
> 

---

## 与其他方法的对比

| 方法 | 向量数 | 精度 | 检索速度 | 存储 |
| --- | --- | --- | --- | --- |
| **Bi-Encoder (单向量)** | 1 per doc | 中 | 极快 | 小 |
| **Cross-Encoder** | 无（联合编码） | 高 | 极慢 | 无索引 |
| **ColBERT (多向量)** | N per doc (N=token数) | 高 | 中 | 大 |

---

## ColBERT 原理

### 架构

```
Query: "What is HNSW?"          Doc: "HNSW is a graph-based ANN index..."
       ↓                               ↓
   BERT Encoder                    BERT Encoder
       ↓                               ↓
  [q₁, q₂, q₃, q₄]              [d₁, d₂, d₃, ..., dₘ]
  (每个 token 一个向量)           (每个 token 一个向量)
       ↓                               ↓
         MaxSim: 对每个 qᵢ 找最匹配的 dⱼ，求和
```

### MaxSim 评分

$\text{score}(Q, D) = \sum_{i=1}^{|Q|} \max_{j=1}^{|D|} q_i^T d_j$

对 query 的每个 token 向量 $q_i$，找到 document 中与之最相似的 token 向量 $d_j$（取最大内积），然后将所有 query token 的最佳匹配分数求和。

### ColBERTv2 改进

- **残差压缩**：对 document token 向量做聚类 + 残差量化，存储从 128 维 float 降低到 ~2 字节/token
- **去噪训练**：更好的硬负例挖掘 + 知识蒸馏
- **效果**：存储减少 6-10 倍，精度几乎不降

---

## Python 实战

### 1. RAGatouille（ColBERT 最简实战库）

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

### 2. 手动实现 MaxSim

```python
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

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

# 编码
q = encode_tokens("What is HNSW algorithm?")
d1 = encode_tokens("HNSW builds a hierarchical graph for approximate nearest neighbor search.")
d2 = encode_tokens("Python is used for web development and data science.")

print(f"Query vs Doc1 (相关): {maxsim_score(q, d1):.4f}")
print(f"Query vs Doc2 (无关): {maxsim_score(q, d2):.4f}")
```

### 3. BGE-M3 同时输出 Dense + Sparse + Multi-vector

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

---

## 存储与索引

多向量的存储开销远大于单向量：

| 方法 | 每文档存储 | 100万文档 |
| --- | --- | --- |
| Dense 768维 float32 | ~3 KB | ~3 GB |
| ColBERT 128tokens × 128维 | ~65 KB | ~65 GB |
| ColBERTv2 (压缩后) | ~6-10 KB | ~6-10 GB |

支持 ColBERT 的向量库/引擎：**Vespa**（原生支持）、**Qdrant**（named vectors）、**Milvus**

---

## 适用场景

- ✅ **高精度检索**：质量显著优于单向量
- ✅ **问答/RAG 精排**：可替代部分 Cross-Encoder 场景
- ✅ **短查询匹配长文档**：token 级别交互捕捉局部匹配
- ⚠️ **成本更高**：索引更大、检索更慢
- ⚠️ **需要专门支持**：不是所有向量库都原生支持多向量

---

## 相关页面

- ↑ 上级：[1.1 文本嵌入 Text Embeddings](1%20文本嵌入%20Text%20Embeddings.md)
- ← 上一节：稀疏嵌入 SPLADE
- → 下一节：上下文化 Chunk 嵌入（Late Chunking）
- → 训练范式：[2.2 训练范式详解](2%20训练范式详解.md)（Late Interaction 部分）
- → 检索五类表示：[2.3 检索视角的五类表示](3%20检索视角的五类表示.md)
- → ANN 索引：[3.1 ANN 索引算法详解](1%20ANN%20索引算法详解.md)