## 概述

传统 RAG 流程先将长文档切成小块（chunk），再对每个 chunk 独立编码。问题在于：**chunk 脱离了原文语境**，代词、缩写、省略的主语都无法还原，导致检索质量下降。

**上下文化 Chunk 嵌入**是当前长文检索的重要升级方向，核心思路是：**先让模型看到全文上下文，再生成每个 chunk 的向量**。

> 两种主流方案：
> 

> - **Late Chunking**：用长上下文模型编码全文后，再按 chunk 边界切分 token 向量并池化
> 

> - **Contextual Retrieval**（Anthropic）：用 LLM 为每个 chunk 生成上下文摘要，拼接后再编码
> 

---

## 传统切块的问题

假设原文为：

> "HNSW 是一种基于图的近似最近邻算法。**它**通过构建多层导航图实现对数级搜索。"
> 

切成两块后：

- Chunk 1: "HNSW 是一种基于图的近似最近邻算法。"
- Chunk 2: "它通过构建多层导航图实现对数级搜索。"

Chunk 2 中的"它"脱离了上下文，embedding 模型无法知道"它"指的是 HNSW，导致检索 "HNSW 如何搜索" 时可能漏掉 Chunk 2。

---

## Late Chunking

### 原理

```
全文: "HNSW 是一种基于图的... 它通过构建多层..."
                    ↓
    长上下文 Embedding 模型编码全文
    → 得到每个 token 的上下文向量
                    ↓
    按预定义的 chunk 边界切分 token 向量
                    ↓
    对每个 chunk 内的 token 向量做 mean pooling
                    ↓
    Chunk 1 向量 (带全文语境)    Chunk 2 向量 (带全文语境)
```

**关键区别**：传统方法是先切再编码（每个 chunk 独立编码），Late Chunking 是先编码再切（全文共享上下文信息后再分割）。

### Python 实现

```python
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

model_name = "jinaai/jina-embeddings-v2-base-en"  # 支持 8192 tokens
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
model.eval()

def late_chunking(
    full_text: str,
    chunk_texts: list[str],
) -> np.ndarray:
    """
    Late Chunking: 先编码全文，再按 chunk 边界切分并池化。
    返回每个 chunk 的向量 (num_chunks, hidden_dim)
    """
    # 1. 编码全文，获取所有 token 的上下文向量
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=8192)
    with torch.no_grad():
        outputs = model(**inputs)
    all_token_embs = outputs.last_hidden_state.squeeze(0).numpy()  # (seq_len, dim)
    
    # 2. 找到每个 chunk 在全文 token 序列中的起止位置
    chunk_embeddings = []
    current_pos = 1  # 跳过 [CLS]
    
    for chunk_text in chunk_texts:
        chunk_tokens = tokenizer(chunk_text, add_special_tokens=False)["input_ids"]
        chunk_len = len(chunk_tokens)
        
        # 提取该 chunk 对应的 token 向量
        chunk_token_embs = all_token_embs[current_pos : current_pos + chunk_len]
        
        # Mean pooling
        chunk_vec = chunk_token_embs.mean(axis=0)
        # L2 归一化
        chunk_vec = chunk_vec / (np.linalg.norm(chunk_vec) + 1e-8)
        chunk_embeddings.append(chunk_vec)
        
        current_pos += chunk_len
    
    return np.array(chunk_embeddings)

# 示例
full_text = (
    "HNSW 是一种基于图的近似最近邻算法。"
    "它通过构建多层导航图实现对数级搜索复杂度。"
    "核心参数包括 M（最大连接数）和 efConstruction（建图搜索宽度）。"
    "查询时通过 efSearch 控制精度与速度的平衡。"
)

chunks = [
    "HNSW 是一种基于图的近似最近邻算法。",
    "它通过构建多层导航图实现对数级搜索复杂度。",
    "核心参数包括 M（最大连接数）和 efConstruction（建图搜索宽度）。",
    "查询时通过 efSearch 控制精度与速度的平衡。",
]

chunk_vecs = late_chunking(full_text, chunks)
print(f"Chunk 向量 shape: {chunk_vecs.shape}")  # (4, 768)

# 用 query 检索
query = "HNSW 怎么搜索"
q_inputs = tokenizer(query, return_tensors="pt")
with torch.no_grad():
    q_output = model(**q_inputs)
q_vec = q_output.last_hidden_state.mean(dim=1).squeeze().numpy()
q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-8)

scores = chunk_vecs @ q_vec
for i, (chunk, score) in enumerate(zip(chunks, scores)):
    print(f"  Chunk {i} [{score:.4f}]: {chunk}")
```

---

## Contextual Retrieval (Anthropic)

### 原理

与 Late Chunking 不同，Contextual Retrieval 不需要长上下文 embedding 模型，而是用 **LLM 为每个 chunk 生成上下文前缀**，然后拼接到 chunk 前面再用普通 embedding 模型编码。

```
原始 Chunk: "它通过构建多层导航图实现对数级搜索。"
                      ↓
    LLM 生成上下文摘要:
    "本文讨论 HNSW 近似最近邻算法。以下是关于其搜索机制的描述："
                      ↓
    拼接后: "本文讨论 HNSW 近似最近邻算法。以下是关于其搜索机制的描述：
             它通过构建多层导航图实现对数级搜索。"
                      ↓
    普通 Embedding 模型编码
                      ↓
    带上下文的 Chunk 向量
```

### Python 实现

```python
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np

client = OpenAI()
embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

def generate_context(
    full_document: str,
    chunk: str,
    model: str = "gpt-4o-mini",
) -> str:
    """用 LLM 为 chunk 生成上下文摘要前缀"""
    prompt = f"""<document>
{full_document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context (1-2 sentences) to situate this chunk 
within the overall document for the purposes of improving search retrieval. 
Answer only with the context, no other text."""
    
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0,
    )
    return resp.choices[0].message.content.strip()

def contextual_retrieval_encode(
    full_document: str,
    chunks: list[str],
) -> np.ndarray:
    """为每个 chunk 生成上下文增强向量"""
    enriched_chunks = []
    for chunk in chunks:
        context = generate_context(full_document, chunk)
        enriched = f"{context}\n\n{chunk}"
        enriched_chunks.append(enriched)
        print(f"  Context: {context[:80]}...")
    
    embeddings = embed_model.encode(enriched_chunks, normalize_embeddings=True)
    return embeddings

# 示例
doc = (
    "HNSW is a graph-based ANN algorithm. "
    "It builds a hierarchical multi-layer graph. "
    "Key parameters include M and efConstruction. "
    "Query uses efSearch to balance speed and accuracy."
)

chunks = [
    "It builds a hierarchical multi-layer graph.",
    "Key parameters include M and efConstruction.",
    "Query uses efSearch to balance speed and accuracy.",
]

# 对比：传统编码 vs 上下文增强编码
traditional_vecs = embed_model.encode(chunks, normalize_embeddings=True)
contextual_vecs = contextual_retrieval_encode(doc, chunks)

query = "How does HNSW search work?"
q_vec = embed_model.encode([query], normalize_embeddings=True)

print("\n传统编码:")
for i, s in enumerate((q_vec @ traditional_vecs.T)[0]):
    print(f"  Chunk {i}: {s:.4f}")

print("\n上下文增强编码:")
for i, s in enumerate((q_vec @ contextual_vecs.T)[0]):
    print(f"  Chunk {i}: {s:.4f}")
```

---

## 两种方案对比

| 维度 | **Late Chunking** | **Contextual Retrieval** |
| --- | --- | --- |
| 依赖 | 长上下文 embedding 模型 | LLM + 普通 embedding 模型 |
| 额外成本 | 低（模型推理时间略增） | 高（每个 chunk 需 LLM 调用） |
| 上下文质量 | 隐式（attention 机制） | 显式（LLM 生成的摘要） |
| 模型要求 | 需支持长输入的 embedding 模型 | 任意 embedding 模型 + LLM |
| 适合场景 | 大规模批量处理 | 对质量要求极高、预算充足 |

---

## 适用场景

- ✅ **长文档 RAG**：论文、手册、法律文书、技术文档
- ✅ **代词/引用多的文档**：新闻、叙述类文本
- ✅ **结构化文档**：章节间有大量交叉引用
- ✅ **当前最重要的检索质量升级点之一**

---

## 相关页面

- ↑ 上级：[1.1 文本嵌入 Text Embeddings](1%20文本嵌入%20Text%20Embeddings.md)
- ← 上一节：多向量 / Late Interaction（ColBERT）
- → 长上下文模型：长上下文文本嵌入（Nomic / Jina）
- → 检索五类表示：[2.3 检索视角的五类表示](3%20检索视角的五类表示.md)
- → 评测对比：[2.4 评测体系](4%20评测体系.md)
- → 综合流程：[6. 综合使用流程](6%20综合使用流程.md)