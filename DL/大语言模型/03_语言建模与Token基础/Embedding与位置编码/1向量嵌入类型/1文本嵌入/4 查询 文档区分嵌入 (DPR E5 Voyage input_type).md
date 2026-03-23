## 概述

传统 Bi-Encoder 对 query 和 document 使用完全相同的编码方式。而**查询/文档区分嵌入**认为：query 通常是短问句，document 是长段落，两者的语言分布不同，应该区别对待。

> 核心思想：**让模型知道当前输入是"问题"还是"文档"**，通过不同的前缀、prompt 或独立编码器来优化检索效果。
> 

---

## 三种实现范式

### 1. 独立双编码器（DPR 风格）

**DPR**（Dense Passage Retrieval, Facebook 2020）使用两个独立的 BERT 编码器：

- **Question Encoder**：编码用户问题
- **Passage Encoder**：编码文档段落

训练时用 in-batch negatives + hard negatives，让问题与对应段落的向量接近。

### 2. 共享编码器 + 前缀区分（E5 / BGE 风格）

使用同一个模型，但通过不同前缀告知角色：

- E5：`query: ...` vs `passage: ...`
- BGE：query 加 instruction 前缀，document 不加
- Nomic：`search_query: ...` vs `search_document: ...`

### 3. API 参数区分（OpenAI / Voyage 风格）

通过 API 参数 `input_type` 指定：

- OpenAI：虽无显式参数，内部做了优化
- Voyage：`input_type="query"` vs `input_type="document"`

---

## DPR 详解

### 架构

```
Query: "What is HNSW?"        Doc: "HNSW is a graph-based ANN index..."
       ↓                              ↓
  Question Encoder (BERT)       Passage Encoder (BERT)
       ↓                              ↓
    q ∈ R^768                      d ∈ R^768
       ↓                              ↓
              sim(q, d) = q · d
```

### 训练损失

$\mathcal{L} = -\log \frac{\exp(q_i \cdot d_i^+)}{\exp(q_i \cdot d_i^+) + \sum_{j} \exp(q_i \cdot d_j^-)}$

关键：**Hard Negative Mining** —— 用 BM25 检索出的高分但不相关段落作为难负例，显著提升效果。

---

## Python 实战

### 1. 使用 DPR 原始模型（Hugging Face）

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

### 2. E5 query/passage 前缀

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
```

### 3. Voyage API input_type

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

## 为什么区分有效？

- **分布不对称**：query 短且模糊，document 长且具体，统一编码会互相干扰
- **前缀/指令**让模型在编码时「预期」输入的角色，调整注意力分配
- **实测提升**：E5 有/无前缀差距可达 3-5 个 Recall 百分点

---

## 实践要点

<aside>
⚠️

**最常见错误**：忘记加前缀，或 query/document 前缀搞反。务必查阅模型文档确认前缀格式！

</aside>

- 离线索引时用 `passage/document` 前缀编码所有文档
- 在线检索时用 `query` 前缀编码用户输入
- 不要混用：如果文档用了前缀编码，query 也必须用对应前缀

---

## 相关页面

- ↑ 上级：[1.1 文本嵌入 Text Embeddings](1%20文本嵌入%20Text%20Embeddings.md)
- ← 上一节：长上下文文本嵌入
- → 下一节：稀疏嵌入 SPLADE
- → 训练范式：[2.2 训练范式详解](2%20训练范式详解.md)（对比学习 / Bi-Encoder / Cross-Encoder）
- → 检索五类表示：[2.3 检索视角的五类表示](3%20检索视角的五类表示.md)