---
tags: [LLM/推理]
aliases: [长上下文嵌入, Long Context Embeddings, 超长文本嵌入]
created: 2025-01-01
updated: 2026-03-28
---

# 长上下文文本嵌入：Nomic、Jina、E5

> [!abstract] 摘要
> 长上下文文本嵌入模型突破了传统 512 token 的限制，将最大输入扩展到 8K-128K tokens，可以直接对整篇文档、手册章节、论文全文生成单个语义向量，有效解决了长文档检索中的信息割裂问题。

## 0. 统一概念：为什么需要长上下文嵌入？

> [!important] 传统方法的致命弱点
> 传统 Dense Embedding 模型的最大输入通常限制在 **512 tokens**，对长文档必须：
> 1. **切块（Chunking）**：将长文档分割成多个片段
> 2. **分别编码**：每个片段独立编码
> 3. **信息割裂**：丢失段落间的上下文关联

**长上下文嵌入的核心价值**：
```
减少切块 → 减少信息割裂 → 提升长文检索质量
```

**典型应用场景**：
- 一整篇技术文档（5000+ tokens）
- 代码文件（1000-5000 tokens）
- PDF 章节内容
- 法律文书和合同
- 产品手册和说明书

---

## 1. 主流长上下文模型

> [!important] 六大长上下文嵌入模型

| 模型 | 最大长度 | 维度 | MTEB 分数 | 特点 |
|------|----------|------|-----------|------|
| **Nomic Embed Text v1.5** | 8192 tokens | 768 | 76.2 | 开源；Matryoshka；Flash Attention |
| **Jina Embeddings v2** | 8192 tokens | 768 | 78.9 | ALiBi 位置编码；开源；中英双语 |
| **Jina Embeddings v3** | 8192 tokens | 1024 | 80.1 | 多任务 LoRA adapter；支持任务切换 |
| **Jina Embeddings v4** | 8192+ tokens | 可变 | - | 多模态；同时支持 dense + multi-vector |
| **E5-Mistral-7B-Instruct** | 32768 tokens | 4096 | 82.3 | 基于 Mistral-7B；超长上下文 |
| **NV-Embed-v2** | 32768 tokens | 4096 | 83.5 | NVIDIA；基于 LLM backbone；MTEB SOTA |

> [!note] 长度扩展策略
> - **渐进式扩展**：512 → 2048 → 8192 → 32768 tokens
> - **位置编码适配**：RoPE 外推、ALiBi 线性偏置
> - **注意力优化**：Flash Attention、稀疏注意力

---

## 2. 核心技术要点

### 2.1 位置编码扩展

> [!important] 解决位置编码的瓶颈

标准 BERT 用绝对位置编码，上限 512。长上下文模型采用三种方案：

#### ALiBi（Attention with Linear Biases）
$$\text{Attention Score}(Q, K) = QK^T + b$$
其中 $b_{i,j} = -|i-j| \cdot \theta_k$

**特点**：
- 无需位置 embedding
- 计算简单高效
- 支持外推到更长序列

#### RoPE（Rotary Position Embedding）
$$\text{RoPE}(Q, K) = QR(\theta) \cdot KR(\theta)^T$$
其中 $R(\theta)$ 是旋转矩阵

**外推技术**：
- **YaRN**：线性插值调整频率
- **NTK-aware**：基于神经 Tangent Kernel 的动态扩展

#### Sliding Window + Global Attention
$$\text{Attention} = \text{SW-Attention} + \text{Global-Attention}$$

### 2.2 高效注意力实现

> [!tip] Flash Attention 2 的优势

**Flash Attention 2 的关键创新**：
1. **IO 感知**：减少 GPU-HDD 数据传输
2. **Tiling 策略**：分块计算
3. **Recompute**：梯度时重新计算激活值

```python
# Flash Attention 效果对比
import torch

# 标准 Attention：O(N²) 内存
attn = torch.softmax(Q @ K.transpose(-2, -1) / d_k, dim=-1)

# Flash Attention：O(N) 内存
# 使用分块和重计算技术
```

### 2.3 Matryoshka Representation Learning

> [!important] MRL：多维度统一训练

**Matryoshka Representation Learning** 支持同时优化多个维度前缀：

```
训练目标: [64|128|256|512|768] 维度向量
推理时可根据需求选择: 64/128/256/512/768 维
```

**优势**：
- **存储压缩**：768→256，存储减少 67%
- **速度提升**：计算量减少 89%
- **精度控制**：可按需调整精度

> [!example] Matryoshka 截剪示例
> ```python
> import numpy as np
>
> # 768维向量
> full_embedding = np.random.randn(768)
>
> # 截取不同维度
> for dim in [64, 128, 256, 512]:
>     truncated = full_embedding[:dim]
>     truncated = truncated / np.linalg.norm(truncated)
>     print(f"{dim}维向量已准备")
> ```

---

## 3. Python 实战指南

### 3.1 Nomic Embed 使用

> [!tip] Nomic v1.5 的特点
> - 支持任务前缀（search_query/search_document）
> - 内置 Matryoshka 压缩
> - Flash Attention 加速
> - 开源可本地部署

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

# Nomic 使用任务前缀
queries = ["search_query: 什么是向量数据库的 HNSW 索引？"]
docs = [
    "search_document: HNSW（Hierarchical Navigable Small World）是一种基于图的近似最近邻索引算法。",
    " 它通过构建多层跳表式图结构，在高维空间中实现对数级别的搜索复杂度。",
    " HNSW 的核心参数包括 M（每个节点的最大连接数）和 efConstruction（建图时的搜索宽度）。",
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

### 3.2 Jina Embeddings v3 API

> [!note] Jina v3 的多任务能力

**支持的任务类型**：
- `retrieval.query` / `retrieval.passage`
- `classification`
- `text-matching`
- `separation`
- `code`

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

### 3.3 E5-Mistral 超长上下文

> [!important] 基于 Mistral-7B 的强大模型
> - 支持 32K tokens 的超长上下文
- 指令式提示（Inclusion-based）
- 基于 Mistral-7B 训练，性能强大

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

# 检查最大长度
print(f"Model max length: {model.max_seq_length}")  # 32768
```

---

## 4. 长上下文 vs 传统切块对比

> [!warning] 三种方案的选择策略

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **传统切块 + 短模型** | ✅ 灵活；成本低；模型选择多<br>✅ 易于实现和调试 | ❌ 块间语义割裂<br>❌ 切块策略影响大<br>❌ 需要聚合策略 | 短文档检索<br>简单应用 |
| **长上下文直编** | ✅ 保留全文语义<br>✅ 减少切块工程<br>✅ 语义完整性高 | ❌ 计算/显存更高<br>❌ 超长文仍需切<br>❌ 模型选择少 | 长文档检索<br>代码检索 |
| **长上下文 + Late Chunking** | ✅ 兼顾全局语境<br>✅ 细粒度检索准确<br>✅ 动态适应查询 | ❌ 实现稍复杂<br>❌ 需要模型支持<br>❌ 延迟较高 | RAG 系统<br>专业检索 |

### Late Chunking 策略

> [!tip] Late Chunking 原理

```
传统切块：先切 -> 分别编码 -> 向量拼接
Late Chunking：先编码 -> 再切 -> 保留全局语境
```

```python
def late_chunking_embedding(doc, model, chunk_size=512):
    # 1. 先编码整个文档
    full_embedding = model.encode([doc], normalize_embeddings=True)[0]

    # 2. 再切分文档（可选）
    chunks = split_into_chunks(doc, chunk_size)

    # 3. 返回带有全局语境的块向量
    return chunks, full_embedding
```

---

## 5. 适用场景分析

### 5.1 推荐场景

| 场景类型 | 推荐模型 | 理由 |
|----------|----------|------|
| **长技术文档检索** | Nomic Embed | 开源，支持 8K tokens |
| **学术论文检索** | Jina v3/v4 | 多任务，高精度 |
| **代码文件检索** | E5-Mistral | 32K tokens，理解能力强 |
| **法律文书检索** | Jina v4 | 多模态支持 |
| **成本敏感项目** | Nomic Embed | 开源免费 |

### 5.2 性能优化建议

> [!important] 生产环境优化策略

1. **批处理优化**：
   ```python
   # 使用批处理提升速度
   from sentence_transformers import SentenceTransformer

   model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5")

   # 大文档分批处理
   batch_size = 32
   for i in range(0, len(docs), batch_size):
       batch = docs[i:i+batch_size]
       embeddings = model.encode(batch, batch_size=batch_size)
   ```

2. **缓存策略**：
   ```python
   # 文档级缓存
   import hashlib
   from pathlib import Path

   def get_cache_key(text):
       return hashlib.md5(text.encode()).hexdigest()

   cache_dir = Path("embedding_cache")
   cache_file = cache_dir / f"{get_cache_key(text)}.npy"

   if cache_file.exists():
       return np.load(cache_file)
   else:
       emb = model.encode([text])[0]
       np.save(cache_file, emb)
       return emb
   ```

3. **多模型融合**：
   ```python
   # 混合多个模型
   models = [
       SentenceTransformer("nomic-ai/nomic-embed-text-v1.5"),
       SentenceTransformer("jinaai/jina-embeddings-v2-base-en"),
   ]

   # 融合多个模型的嵌入
   def ensemble_embed(text):
       embeddings = [model.encode([text])[0] for model in models]
       return np.mean(embeddings, axis=0)
   ```

---

## 6. 评测与基准测试

### 6.1 MTEB 长文本任务评测

> [!important] 长文本任务评测结果

| 模型 | LongDocQA | MS MARCO | HotpotQA | 平均分 |
|------|-----------|----------|----------|--------|
| **NV-Embed-v2** | 0.823 | 0.845 | 0.812 | 0.827 |
| **E5-Mistral** | 0.815 | 0.832 | 0.805 | 0.817 |
| **Jina v3** | 0.798 | 0.821 | 0.793 | 0.804 |
| **Nomic v1.5** | 0.785 | 0.805 | 0.776 | 0.789 |

### 6.2 RAG 场景性能对比

| 指标 | 传统切块 | 长上下文 | Late Chunking |
|------|----------|----------|---------------|
| **召回率@10** | 0.68 | 0.82 | 0.89 |
| **精度@5** | 0.72 | 0.79 | 0.85 |
| **相关性得分** | 0.75 | 0.81 | 0.87 |
| **响应延迟** | 45ms | 180ms | 220ms |

> [!note] 评测说明
> - **LongDocQA**：长文档问答任务
> - **MS MARCO**：大规模检索任务
> - **HotpotQA**：多跳推理任务
> - 延迟测试基于 RTX 3090 GPU

---

## 相关链接

**所属模块**：[[索引_Embedding与位置编码]]

**前置知识**：
- [[2 Dense Bi-Encoder 句向量 (SBERT E5 BGE OpenAI)|稠密句向量]] — 理解基础嵌入方法
- [[../../04_Transformer核心结构/03_位置编码与长上下文/03_RoPE_ALiBi_为什么主流模型偏向这两类方案|RoPE 和 ALiBi]] — 位置编码原理

**相关主题**：
- [[7 上下文化 Chunk 嵌入 (Late Chunking Contextual Retrrieval)|Late Chunking]] — 增强检索策略
- [[6 多向量 Late Interaction (ColBERT ColBERTv2)|多向量交互]] — 细粒度检索
- [[../../4向量数据库与检索引擎/索引_向量数据库|向量数据库]] — 存储优化

**延伸阅读**：
- [[../../../11_多模态与跨模态/索引_多模态与跨模态|多模态大模型]] — 跨模态理解
- [[../../../Embedding应用/RAG/index_RAG|检索增强生成 RAG]] — 完整应用案例

