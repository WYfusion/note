## 概述

传统稀疏检索（BM25）基于精确词匹配，无法捕捉同义词和语义关联。**SPLADE**（SParse Lexical AnD Expansion model）是一种**学习型稀疏嵌入**方法，结合了稀疏表示的可解释性和神经网络的语义理解能力。

> 核心思想：用 BERT 为每个文档/查询生成一个**高维稀疏向量**，维度对应词表中的词项（term），非零值表示该词项的重要性权重。模型会自动进行**词项扩展**——即使原文没有出现某个词，模型也可能为语义相关的词赋予非零权重。
> 

---

## 与其他表示的对比

| 方法 | 向量类型 | 维度 | 可解释性 | 语义能力 |
| --- | --- | --- | --- | --- |
| **BM25** | 稀疏（词频统计） | 词表大小（~30K） | ✅ 高 | ❌ 无 |
| **SPLADE** | 稀疏（学习型） | 词表大小（~30K） | ✅ 高 | ✅ 有（词项扩展） |
| **Dense Embedding** | 稠密 | 768–1024 | ❌ 低 | ✅ 强 |

---

## SPLADE 原理

### 架构

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

### 关键公式

给定输入 token 序列 $t_1, ldots, t_n$，SPLADE 输出稀疏向量 $s$：

$s_j = \max_{i=1}^{n} \log(1 + \text{ReLU}(w_j^i))$

其中 $w_j^i$ 是第 $i$ 个 token 在词表第 $j$ 个位置的 MLM logit。

### 词项扩展示例

输入 `"deep learning"` 后，SPLADE 可能输出的非零词项包括：

- `deep`: 2.1, `learning`: 1.8（原文词项）
- `neural`: 1.2, `network`: 0.9, `ai`: 0.7, `machine`: 0.6（扩展词项）

这种扩展让稀疏检索也能匹配语义相关但词面不同的文档。

### 正则化

SPLADE 用 **FLOPS 正则化**控制稀疏度：

$\mathcal{L}_{\text{reg}} = \lambda \sum_{j} \bar{a}_j^2$

其中 $\bar{a}_j$ 是词项 $j$ 在 batch 内的平均激活值。$lambda$ 越大，向量越稀疏（检索越快但精度可能下降）。

---

## Python 实战

### 1. 使用 Hugging Face 推理

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np

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

### 2. 计算稀疏向量相似度

```python
def sparse_dot_product(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """两个稀疏向量的点积"""
    score = 0.0
    for term, weight_a in vec_a.items():
        if term in vec_b:
            score += weight_a * vec_b[term]
    return score

# 相关 query-doc 对
q = splade_encode("How does HNSW work?")
d1 = splade_encode("HNSW is a graph-based approximate nearest neighbor algorithm.")
d2 = splade_encode("Python is a popular programming language.")

print(f"Query vs Doc1 (相关): {sparse_dot_product(q, d1):.4f}")
print(f"Query vs Doc2 (无关): {sparse_dot_product(q, d2):.4f}")
```

### 3. 与 Dense 混合检索（Hybrid）

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Dense 模型
dense_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

def hybrid_search(
    query: str,
    corpus: list[str],
    alpha: float = 0.7,  # dense 权重
) -> list[tuple[int, float]]:
    """Hybrid = alpha * dense_score + (1-alpha) * sparse_score"""
    # Dense scores
    q_dense = dense_model.encode([query], normalize_embeddings=True)
    d_dense = dense_model.encode(corpus, normalize_embeddings=True)
    dense_scores = (q_dense @ d_dense.T)[0]
    
    # Sparse scores (SPLADE)
    q_sparse = splade_encode(query)
    sparse_scores = []
    for doc in corpus:
        d_sparse = splade_encode(doc)
        sparse_scores.append(sparse_dot_product(q_sparse, d_sparse))
    sparse_scores = np.array(sparse_scores)
    
    # 归一化到 [0, 1]
    if dense_scores.max() > 0:
        dense_scores = dense_scores / dense_scores.max()
    if sparse_scores.max() > 0:
        sparse_scores = sparse_scores / sparse_scores.max()
    
    # 混合
    hybrid_scores = alpha * dense_scores + (1 - alpha) * sparse_scores
    
    ranked = sorted(enumerate(hybrid_scores), key=lambda x: x[1], reverse=True)
    return ranked

# 示例
corpus = [
    "HNSW builds a hierarchical graph for fast nearest neighbor search.",
    "IVF partitions vectors into clusters for efficient retrieval.",
    "Python is great for machine learning development.",
]

results = hybrid_search("How does approximate nearest neighbor work?", corpus)
for idx, score in results:
    print(f"  [{score:.4f}] {corpus[idx][:60]}")
```

---

## 适用场景

- ✅ **首阶段召回**：替代 BM25，语义更强且保留词项匹配
- ✅ **Hybrid 检索**：与 Dense 互补，当前最稳方案之一
- ✅ **可解释性要求高**：可以查看哪些词项贡献了匹配分
- ⚠️ **索引支持**：需要向量库支持稀疏向量（Qdrant、Milvus、Vespa、Elasticsearch）

---

## 相关页面

- ↑ 上级：[1.1 文本嵌入 Text Embeddings](1%20文本嵌入%20Text%20Embeddings.md)
- ← 上一节：查询/文档区分嵌入（DPR）
- → 下一节：多向量 / Late Interaction（ColBERT）
- → 检索五类表示：[2.3 检索视角的五类表示](3%20检索视角的五类表示.md)
- → 向量库实战：[3.2 专用向量数据库实战](2%20专用向量数据库实战.md)