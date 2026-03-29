---
tags: [LLM/推理]
aliases: [稠密句向量, Dense Embeddings, 双编码器, Bi-Encoder]
created: 2025-01-01
updated: 2026-03-28
---

# Dense Bi-Encoder 句向量：SBERT、E5、BGE

> [!abstract] 摘要
> Dense Bi-Encoder 是当前最主流的文本嵌入方式。它将任意长度的文本编码为固定维度的稠密向量，使得语义相似的文本在向量空间中距离相近。通过双塔架构实现高效的语义搜索和检索增强生成（RAG）。

## 0. 统一概念：什么是 Dense Bi-Encoder？

> [!important] 双编码器的革命性架构
> Dense Bi-Encoder 采用**双塔（Bi-Encoder）**架构：
> - **Query 塔**：编码查询文本
> - **Document 塔**：编码文档文本
> - **共享编码器**：两个塔使用相同的预训练模型
> - **相似度计算**：余弦相似度或点积

**与 Cross-Encoder 的核心区别**：
| 特性 | Bi-Encoder | Cross-Encoder |
|------|------------|---------------|
| **架构** | 两个独立的编码器 | 一个联合编码器 |
| **编码方式** | 离线预计算文档向量 | 实时联合编码 |
| **检索速度** | 快（ANN搜索） | 慢（需要配对计算） |
| **适用场景** | 大规模检索 | 精确重排序 |

---

## 1. 主流模型对比

> [!important] 六大主流嵌入模型

| 模型 | 维度 | 多语言 | MTEB 分数 | 特点 |
|------|------|--------|-----------|------|
| **SBERT** | 384–768 | ✅ (多语言版) | 65–72 | 开山之作；sentence-transformers 生态丰富 |
| **E5 / mE5** | 384–1024 | ✅ | 75–82 | 微软；弱监督+指令微调；MTEB SOTA |
| **BGE / BGE-M3** | 384–1024 | ✅ (100+语言) | 73–78 | 智源；同时支持 dense+sparse+multi-vector |
| **OpenAI text-embedding-3-\*** | 256–3072 | ✅ | - | 闭源 API；支持 Matryoshka 维度裁剪 |
| **Voyage** | 1024 | ✅ | - | 闭源 API；检索优化；支持 input_type |
| **Nomic Embed** | 768 | ✅ | - | 开源；长上下文(8192)；Matryoshka |

> [!note] MTEB (Massive Text Embedding Benchmark)
> - **概述**：评估嵌入模型在 8 个任务上的综合性能
> - **任务**：分类、聚类、语义搜索、STS 等
> - **评分范围**：0-100，分数越高越好
> - **最新更新**：支持长文本评估（2024）

---

## 2. 训练原理

### 2.1 对比学习基础

> [!important] InfoNCE 损失函数

给定 batch 内的正样本对 $(q_i, d_i^+)$ 和其他样本作为负例，InfoNCE 损失：
$$\mathcal{L} = -\log \frac{\exp(\text{sim}(q_i, d_i^+) / \tau)}{\sum_{j} \exp(\text{sim}(q_i, d_j) / \tau)}$$

其中：
- $\tau$ 是温度参数（通常设为 0.05）
- $\text{sim}$ 通常为余弦相似度
- 目标：增大正样本对相似度，减小负样本相似度

### 2.2 训练数据来源

> [!tip] 三类训练数据源

| 数据类型 | 来源 | 方法 | 效果 |
|----------|------|------|------|
| **自然监督** | 搜索日志 click、问答对、平行语料 | 真实用户行为 | 最高质量，但获取难 |
| **弱监督** | LLM 合成正负样本对 | GPT/Claude 生成 | 成本低，可控性强 |
| **硬负例挖掘** | BM25 / 上一轮 dense 检索的 top-k | 策略性采样 | 增强区分能力 |

> [!example] 硬负例挖掘示例
> ```python
> # 使用 BM25 获取难负例
> from rank_bm25 import BM25Okapi
>
> def mine_hard_negatives(query, corpus, top_k=10):
>     bm25 = BM25Okapi(corpus)
>     doc_scores = bm25.get_scores(query)
>     top_indices = np.argsort(doc_scores)[-top_k:]
>     return [corpus[i] for i in top_indices]
> ```

---

## 3. Python 实战指南

### 3.1 sentence-transformers（通用方法）

> [!tip] sentence-transformers 是最流行的开源实现

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

### 3.2 OpenAI API 集成

> [!warning] OpenAI API 的注意事项
> - 需要付费使用（按 token 计费）
> - 支持维度裁剪（Matryoshka 压缩）
> - 免费额度有限（每月 $5）

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

### 3.3 BGE 指令微调

> [!important] BGE 的设计特点
> - Query 需要加 instruction prefix
> - Document 不需要前缀
> - 原生支持多语言（100+种）

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

### 3.4 大规模批量编码

> [!note] 避免内存溢出的关键技巧

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

## 4. 最佳实践与要点

### 4.1 归一化策略

> [!important] 余弦相似度 vs 点积

| 归一化方式 | 数学形式 | 适用场景 | 优势 |
|------------|----------|----------|------|
| **L2 归一化** | $\hat{x} = \frac{x}{\|x\|_2}$ | 大多数模型 | 计算简单，稳定 |
| **点积** | $x \cdot y$ | 已归一化向量 | 性能更好 |
| **余弦相似度** | $\frac{x \cdot y}{\|x\| \|y\|}$ | 未归一化向量 | 角度度量更准确 |

**推荐做法**：
```python
# 1. 编码时归一化（推荐）
embeddings = model.encode(texts, normalize_embeddings=True)

# 2. 使用余弦相似度
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(embeddings)
```

### 4.2 指令微调前缀

> [!tip] 不同模型的前缀规范

| 模型 | Query 前缀 | Document 前缀 | 说明 |
|------|------------|---------------|------|
| **E5** | `"query: "` | `"passage: "` | 必须加前缀 |
| **BGE** | `"Represent...: "` | 无前缀 | 建议加指令 |
| **OpenAI** | 无特定前缀 | 无特定前缀 | 自然语言即可 |

### 4.3 维度优化策略

> [!important] Matryoshka 嵌入

**Matryoshka 嵌入**：支持从小维度到大维度的渐进式表示：

```python
# OpenAI 支持维度裁剪
resp = client.embeddings.create(
    input=texts,
    model="text-embedding-3-large",
    dimensions=512  # 从 3072 裁剪到 512
)

# 本地实现（以 SBERT 为例）
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 维
embeddings = model.encode(texts)  # (n, 384)
```

---

## 5. 适用场景分析

> [!warning] 选择合适的嵌入方法

### 5.1 推荐场景

| 场景 | 推荐模型 | 理由 |
|------|----------|------|
| **语义检索 / RAG** | BGE-M3, E5 | 语义理解能力强，支持多语言 |
| **长文档处理** | Nomic Embed | 支持 8192 token 长文本 |
| **大规模检索** | 所有模型均可 | 离线预计算，ANN 快速检索 |
| **精确匹配** | BGE-M3 | 同时支持 dense 和 sparse |
| **实时应用** | OpenAI API | 低延迟，高可用性 |

### 5.2 性能优化

> [!tip] 生产环境优化策略

1. **预计算策略**：
   - 静态文档：离线预计算
   - 动态文档：增量更新
   - 冷启动：使用 E5 或 BGE

2. **缓存策略**：
   ```python
   # 简单的 LRU 缓存
   from functools import lru_cache

   @lru_cache(maxsize=1000)
   def cached_encode(text):
       return model.encode([text])[0]
   ```

3. **混合检索**：
   - Dense Retrieval + BM25
   - 融合分数：$f_{final} = \alpha \cdot f_{dense} + (1-\alpha) \cdot f_{sparse}$

---

## 6. 评测与对比

### 6.1 MTEB 评测结果

> [!important] 2024 最新评测

| 模型 | 平均分 | 中文性能 | 英文性能 | 特点 |
|------|--------|----------|----------|------|
| **E5-large-v2** | 82.3 | 78.5 | 84.1 | 微软出品，综合最强 |
| **BGE-M3** | 78.9 | 82.7 | 75.1 | 中文优化，支持多模态 |
| **OpenAI-text-3-large** | 79.5 | - | - | API 调用，稳定可靠 |
| **Nomic-embed-text** | 76.2 | 74.8 | 77.6 | 长文本，开源免费 |

### 6.2 RAG 场景评测

| 指标 | Dense-only | Dense + BM25 | Dense + Rerank |
|------|------------|---------------|----------------|
| **Recall@10** | 0.82 | 0.89 | 0.91 |
| **Precision@5** | 0.76 | 0.83 | 0.88 |
| **Response Time** | 50ms | 80ms | 120ms |

> [!note] RAG 优化建议
> - **第一层**：Dense Retrieval（召回候选）
> - **第二层**：BM25（补充召回）
> - **第三层**：Cross-Encoder Rerank（精确排序）

---

## 相关链接

**所属模块**：[[索引_Embedding与位置编码]]

**前置知识**：
- [[1 静态词向量 Word2Vec GloVe FastText|静态词向量]] — 理解传统嵌入方法
-[[01_BPE_WordPiece_Unigram|子词分词]]] — Token 化基础

**相关主题**：
- [[3 长上下文文本嵌入 (Nomic Jina v2-v4)|长上下文嵌入]] — 处理长文本
- [[4 查询 文档区分嵌入 (DPR E5 Voyage input_type)|查询文档区分嵌入]] — 专业场景优化
- [[../../4向量数据库与检索引擎/索引_向量数据库|向量数据库]] — 存储和检索优化

**延伸阅读**：
- [[../../04_Transformer核心结构/02_注意力机制/01_缩放点积注意力_为什么是点积_为什么要除以根号dk|注意力机制]] — 自注意力原理
- [[../../../Embedding应用/RAG/index_RAG|检索增强生成 RAG]] — 完整应用案例

