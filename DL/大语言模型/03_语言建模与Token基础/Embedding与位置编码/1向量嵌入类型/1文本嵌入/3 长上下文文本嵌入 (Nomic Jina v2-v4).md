## 概述

传统 Dense Embedding 模型的最大输入通常限制在 **512 tokens**，对长文档需要先切块（chunking）再分别编码，丢失了段落间的上下文关联。

长上下文文本嵌入模型将最大输入扩展到 **8K–128K tokens**，可以直接对整篇文档、手册章节、论文全文、代码文件生成单个语义向量。

> 核心价值：**减少切块 → 减少信息割裂 → 提升长文检索质量**
> 

---

## 主流模型

| 模型 | 最大长度 | 维度 | 特点 |
| --- | --- | --- | --- |
| **Nomic Embed Text v1.5** | 8192 tokens | 768 | 开源；Matryoshka；Flash Attention；多任务前缀 |
| **Jina Embeddings v2** | 8192 tokens | 768 | ALiBi 位置编码；开源；中英双语版 |
| **Jina Embeddings v3** | 8192 tokens | 1024 | 多任务 LoRA adapter；支持 retrieval/classification/separation 等任务切换 |
| **Jina Embeddings v4** | 8192+ tokens | 可变 | 多模态(text+image/PDF)；同时支持 dense + multi-vector |
| **E5-Mistral-7B-Instruct** | 32768 tokens | 4096 | 基于 Mistral-7B；超长上下文；指令式 |
| **NV-Embed-v2** | 32768 tokens | 4096 | NVIDIA；基于 LLM backbone；MTEB 顶级 |

---

## 技术要点

### 1. 位置编码扩展

标准 BERT 用绝对位置编码，上限 512。长上下文模型采用：

- **ALiBi**（Attention with Linear Biases）：Jina v2 使用，无需位置 embedding
- **RoPE**（Rotary Position Embedding）：Nomic、Mistral 系列使用，支持外推
- **YaRN / NTK-aware 插值**：在推理时扩展 RoPE 的有效长度

### 2. 高效注意力

- **Flash Attention 2**：IO 感知的精确注意力实现，显著降低显存
- **Sliding Window Attention**：局部窗口 + 全局 token 的混合注意力

### 3. Matryoshka Representation Learning（MRL）

部分模型（Nomic、OpenAI）支持 MRL：训练时同时优化多个维度前缀（如 64/128/256/512/768），推理时可按需截断维度，牺牲少量精度换取大幅存储压缩。

---

## Python 实战

### 1. Nomic Embed

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

# Nomic 使用任务前缀
queries = ["search_query: 什么是向量数据库的 HNSW 索引？"]
docs = [
    "search_document: HNSW（Hierarchical Navigable Small World）是一种基于图的近似最近邻索引算法。"
    " 它通过构建多层跳表式图结构，在高维空间中实现对数级别的搜索复杂度。"
    " HNSW 的核心参数包括 M（每个节点的最大连接数）和 efConstruction（建图时的搜索宽度）。"
    " 查询时通过 efSearch 控制搜索精度与速度的平衡。" * 10  # 模拟长文本
]

q_emb = model.encode(queries, normalize_embeddings=True)
d_emb = model.encode(docs, normalize_embeddings=True)

score = q_emb @ d_emb.T
print(f"相似度: {score[0][0]:.4f}")

# Matryoshka 维度裁剪
d_emb_256 = d_emb[:, :256]  # 截取前 256 维
d_emb_256 = d_emb_256 / np.linalg.norm(d_emb_256, axis=1, keepdims=True)
print(f"768维 vs 256维 相似度差异很小")
```

### 2. Jina Embeddings v3（API）

```python
import requests
import numpy as np

JINA_API_KEY = "your-jina-api-key"

def jina_embed(texts: list[str], task: str = "retrieval.passage") -> np.ndarray:
    """调用 Jina Embeddings v3 API
    task 可选: retrieval.query, retrieval.passage, classification,
              text-matching, separation, code
    """
    resp = requests.post(
        "https://api.jina.ai/v1/embeddings",
        headers={"Authorization": f"Bearer {JINA_API_KEY}"},
        json={
            "model": "jina-embeddings-v3",
            "input": texts,
            "task": task,
        },
    )
    data = resp.json()
    return np.array([d["embedding"] for d in data["data"]])

# 编码长文档
long_doc = "这是一篇很长的技术文档..." * 500  # 模拟长文
doc_emb = jina_embed([long_doc], task="retrieval.passage")
query_emb = jina_embed(["HNSW 索引怎么调参？"], task="retrieval.query")

score = query_emb @ doc_emb.T
print(f"Score: {score[0][0]:.4f}")
```

### 3. E5-Mistral 超长上下文

```python
from sentence_transformers import SentenceTransformer

# 基于 Mistral-7B 的 embedding 模型，支持 32K tokens
model = SentenceTransformer("intfloat/e5-mistral-7b-instruct")

# 指令式 query
query = "Instruct: Given a question, retrieve relevant passages\nQuery: How does HNSW work?"
doc = "HNSW builds a multi-layer graph..." * 200  # 长文档

q_emb = model.encode([query], normalize_embeddings=True)
d_emb = model.encode([doc], normalize_embeddings=True)

print(f"Score: {(q_emb @ d_emb.T)[0][0]:.4f}")
```

---

## 长上下文 vs 传统切块

| 方案 | 优点 | 缺点 |
| --- | --- | --- |
| **传统切块 + 短模型** | 灵活；成本低；模型选择多 | 块间语义割裂；切块策略影响大 |
| **长上下文直编** | 保留全文语义；减少切块工程 | 计算/显存更高；超长文仍需切 |
| **长上下文 + Late Chunking** | 兼顾全局语境和细粒度检索 | 实现稍复杂；需要模型支持 |

---

## 适用场景

- ✅ 长技术文档 / 手册 / 论文全文检索
- ✅ 代码文件级检索（整个 .py / .ts 文件编码）
- ✅ PDF / 法律文书 / 合同检索
- ✅ 减少切块工程复杂度

---

## 相关页面

- ↑ 上级：[1.1 文本嵌入 Text Embeddings](1%20文本嵌入%20Text%20Embeddings.md)
- ← 上一节：Dense / Bi-Encoder 句向量
- → 下一节：查询/文档区分嵌入（DPR）
- → Late Chunking 详解：上下文化 Chunk 嵌入
- → 搭配向量库：[3.2 专用向量数据库实战](2%20专用向量数据库实战.md)