## 概述

Dense Bi-Encoder 是**当前最主流的文本嵌入方式**。它把任意长度的文本编码为一个固定维度的稠密向量，使得语义相似的文本在向量空间中距离相近。

> 核心架构：**Bi-Encoder（双塔）** —— Query 和 Document 各自独立通过同一个 Encoder 得到向量，用余弦相似度或点积比较。
> 

与 Cross-Encoder 的区别：Bi-Encoder 可以**离线预计算**所有文档向量，检索时只需编码 query + ANN 搜索，适合大规模场景。

---

## 主流模型一览

| 模型 | 维度 | 多语言 | 特点 |
| --- | --- | --- | --- |
| **SBERT** | 384–768 | ✅ (多语言版) | 开山之作；sentence-transformers 生态 |
| **E5 / mE5** | 384–1024 | ✅ | 微软；弱监督+指令微调；MTEB 高分 |
| **BGE / BGE-M3** | 384–1024 | ✅ (100+语言) | 智源；同时支持 dense+sparse+multi-vector |
| **OpenAI text-embedding-3-\*** | 256–3072 | ✅ | 闭源 API；支持 Matryoshka 维度裁剪 |
| **Voyage** | 1024 | ✅ | 闭源 API；检索优化；支持 input_type |
| **Nomic Embed** | 768 | ✅ | 开源；长上下文(8192)；Matryoshka |

---

## Bi-Encoder 训练原理

### 对比学习损失

给定 batch 内的正样本对 $(q_i, d_i^+)$ 和其他样本作为负例，InfoNCE 损失：

$\mathcal{L} = -\log \frac{\exp(\text{sim}(q_i, d_i^+) / \tau)}{\sum_{j} \exp(\text{sim}(q_i, d_j) / \tau)}$

其中 $\tau$ 是温度参数，$text{sim}$ 通常为余弦相似度。

### 训练数据来源

- 自然监督：搜索日志 click、问答对、平行语料
- 弱监督：LLM 合成正负样本对
- 硬负例挖掘：BM25 / 上一轮 dense 检索的 top-k 非正例

---

## Python 实战

### 1. sentence-transformers（开源模型通用方式）

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# 加载模型（首次会自动下载）
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# 编码文本
sentences = [
    "向量数据库用于存储和检索嵌入向量",
    "Qdrant 是一个开源向量搜索引擎",
    "今天天气真不错",
]
embeddings = model.encode(sentences, normalize_embeddings=True)
print(f"Shape: {embeddings.shape}")  # (3, 768)

# 计算余弦相似度
from numpy.linalg import norm
def cosine_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

print(f"句1 vs 句2: {cosine_sim(embeddings[0], embeddings[1]):.4f}")  # 高
print(f"句1 vs 句3: {cosine_sim(embeddings[0], embeddings[2]):.4f}")  # 低
```

### 2. OpenAI API

```python
from openai import OpenAI
import numpy as np

client = OpenAI()  # 需设置 OPENAI_API_KEY

def get_embeddings(texts: list[str], model="text-embedding-3-small", dimensions=512):
    """调用 OpenAI Embedding API"""
    resp = client.embeddings.create(input=texts, model=model, dimensions=dimensions)
    return np.array([d.embedding for d in resp.data])

texts = [
    "How to use vector databases for RAG?",
    "Vector databases store embeddings for fast retrieval",
    "The recipe for chocolate cake",
]
vecs = get_embeddings(texts)
print(f"Shape: {vecs.shape}")  # (3, 512)

# 相似度矩阵
sim_matrix = vecs @ vecs.T
print("相似度矩阵:")
print(np.round(sim_matrix, 3))
```

### 3. BGE 带 instruction prefix

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# BGE 推荐：query 加前缀，document 不加
queries = ["Represent this sentence for searching relevant passages: What is HNSW?"]
docs = [
    "HNSW is a graph-based approximate nearest neighbor algorithm.",
    "Python is a popular programming language.",
]

q_emb = model.encode(queries, normalize_embeddings=True)
d_emb = model.encode(docs, normalize_embeddings=True)

scores = q_emb @ d_emb.T
print(f"Query vs Doc1: {scores[0][0]:.4f}")  # 高
print(f"Query vs Doc2: {scores[0][1]:.4f}")  # 低
```

### 4. 批量编码大规模语料

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("intfloat/e5-base-v2")

# E5 模型需要加 "query: " 或 "passage: " 前缀
corpus = [f"passage: {text}" for text in load_your_corpus()]  # 自定义函数

# 分批编码，避免 OOM
all_embeddings = model.encode(
    corpus,
    batch_size=256,
    show_progress_bar=True,
    normalize_embeddings=True,
    device="cuda",  # GPU 加速
)

# 保存为 numpy 文件
np.save("corpus_embeddings.npy", all_embeddings)
print(f"已编码 {len(all_embeddings)} 条文档，维度 {all_embeddings.shape[1]}")
```

---

## 使用要点

- **归一化**：大多数模型推荐 L2 归一化后用余弦相似度（等价于点积）
- **前缀/指令**：E5 需要 `query:`/`passage:` 前缀；BGE 推荐 query 加 instruction
- **维度选择**：OpenAI 支持 Matryoshka 裁剪（如 3072→512），精度略降但存储大幅减少
- **max_seq_length**：注意模型的最大输入长度（通常 512 tokens），超出会截断

---

## 适用场景

- ✅ **语义检索 / RAG 召回**：当前最主流方案
- ✅ **语义聚类 / 分类**：文本向量 + K-Means / HDBSCAN
- ✅ **推荐 / 去重**：内容相似度计算
- ✅ **大规模场景**：向量可离线预计算，检索只需 ANN

---

## 相关页面

- ↑ 上级：[1.1 文本嵌入 Text Embeddings](1%20文本嵌入%20Text%20Embeddings.md)
- ← 上一节：静态词向量 Word2Vec / GloVe / FastText
- → 下一节：长上下文文本嵌入（Nomic / Jina）
- → 训练详解：[2.2 训练范式详解](2%20训练范式详解.md)
- → 评测：[2.4 评测体系](4%20评测体系.md)
- → 存入向量库：[3.2 专用向量数据库实战](2%20专用向量数据库实战.md)