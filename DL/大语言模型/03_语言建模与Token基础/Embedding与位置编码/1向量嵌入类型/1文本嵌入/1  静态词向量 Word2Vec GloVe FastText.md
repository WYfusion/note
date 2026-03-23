## 概述

静态词向量是最早一批将「词」映射到连续向量空间的方法。每个词对应 **一个固定向量**，不随上下文变化——所以叫"静态"。

> 核心思想：**共现统计 → 低维稠密表示** —— 经常出现在相似上下文中的词，向量距离就近。
> 

---

## 三大经典模型对比

| 模型 | 核心思路 | 训练方式 | 特点 |
| --- | --- | --- | --- |
| **Word2Vec** | 局部窗口预测 | CBOW（上下文→中心词）或 Skip-gram（中心词→上下文） | 快速；经典；无子词信息 |
| **GloVe** | 全局共现矩阵分解 | 对 log 共现计数做加权最小二乘 | 全局统计 + 局部窗口；类比任务表现好 |
| **FastText** | 子词 n-gram 组合 | Skip-gram + 子词 n-gram 向量求和 | 天然支持 OOV 词；形态丰富语言友好 |

---

## Word2Vec 详解

### Skip-gram 原理

给定中心词 $w_c$，预测窗口内上下文词 $w_o$ 的概率：

$P(w_o \mid w_c) = \frac{\exp(v_{w_o}' \cdot v_{w_c})}{\sum_{w \in V} \exp(v_w' \cdot v_{w_c})}$

实际训练用 **负采样（Negative Sampling）** 近似 softmax，大幅降低计算量。

### Python 示例：Gensim 训练 Word2Vec

```python
from gensim.models import Word2Vec

# 准备语料：每个元素是分好词的句子（list of list of str）
sentences = [
    ["向量", "嵌入", "是", "NLP", "基础"],
    ["Word2Vec", "通过", "上下文", "窗口", "学习", "词向量"],
    ["语义", "相近", "的", "词", "向量", "距离", "也", "近"],
]

# 训练 Skip-gram 模型
model = Word2Vec(
    sentences,
    vector_size=100,   # 向量维度
    window=5,          # 上下文窗口大小
    min_count=1,       # 最低词频
    sg=1,              # 1=Skip-gram, 0=CBOW
    epochs=50,
    workers=4,
)

# 获取词向量
vec = model.wv["向量"]
print(f"'向量' 的嵌入维度: {vec.shape}")  # (100,)

# 找最相似的词
print(model.wv.most_similar("向量", topn=3))

# 保存 & 加载
model.save("word2vec.model")
loaded = Word2Vec.load("word2vec.model")
```

---

## GloVe 详解

### 核心公式

GloVe 最小化目标函数：

$J = \sum_{i,j=1}^{V} f(X_{ij}) \left( w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2$

其中 $X_{ij}$ 是词 $i$ 和词 $j$ 的共现次数，$f(x)$ 是权重函数，防止高频词对主导。

### Python 示例：加载预训练 GloVe

```python
import numpy as np

def load_glove(path: str, dim: int = 100) -> dict[str, np.ndarray]:
    """加载 GloVe 预训练词向量文件"""
    embeddings = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            if vec.shape[0] == dim:
                embeddings[word] = vec
    return embeddings

# 使用（需先下载 glove.6B.100d.txt）
glove = load_glove("glove.6B.100d.txt", dim=100)
print(f"词表大小: {len(glove)}")
print(f"'king' 向量前5维: {glove['king'][:5]}")

# 经典类比：king - man + woman ≈ queen
def analogy(a, b, c, embeddings, topn=5):
    """a:b = c:?"""
    vec = embeddings[b] - embeddings[a] + embeddings[c]
    scores = {}
    for word, emb in embeddings.items():
        if word in (a, b, c):
            continue
        scores[word] = np.dot(vec, emb) / (np.linalg.norm(vec) * np.linalg.norm(emb))
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topn]

print(analogy("man", "king", "woman", glove))
```

---

## FastText 详解

### 子词机制

FastText 把每个词拆成字符 n-gram，词向量 = 所有子词向量之和。例如 `where`（n=3）的子词包括：`<wh`, `whe`, `her`, `ere`, `re>`。

**优势**：对拼写错误、词缀变化、未登录词（OOV）天然鲁棒。

### Python 示例：Gensim 训练 FastText

```python
from gensim.models import FastText

sentences = [
    ["embedding", "models", "map", "words", "to", "vectors"],
    ["fasttext", "uses", "subword", "n-grams", "for", "embeddings"],
    ["out-of-vocabulary", "words", "get", "vectors", "from", "subwords"],
]

model = FastText(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,           # Skip-gram
    min_n=3,        # 最短子词长度
    max_n=6,        # 最长子词长度
    epochs=50,
)

# OOV 词也能获得向量！
vec_oov = model.wv["embeddingggg"]
print(f"OOV 词 'embeddingggg' 向量维度: {vec_oov.shape}")

print(model.wv.most_similar("embedding", topn=3))
```

### 使用 Facebook 官方 fasttext 库

```python
import fasttext

# 训练无监督模型
model = fasttext.train_unsupervised(
    "corpus.txt",       # 每行一句
    model="skipgram",
    dim=100,
    minCount=1,
    minn=3,
    maxn=6,
    epoch=25,
)

print(model.get_word_vector("hello").shape)   # (100,)
print(model.get_nearest_neighbors("hello", k=5))
```

---

## 适用场景与局限

| 维度 | 说明 |
| --- | --- |
| ✅ 适合 | 词级特征输入、轻量部署、资源受限环境、词类比/聚类、下游任务初始化 |
| ⚠️ 局限 | 无法区分多义词（bank=银行 vs 河岸）；句子语义需额外聚合；已被上下文模型大幅超越 |
| 📌 现状 | 作为"经典基础"教学和轻量场景仍有价值，主流检索/RAG 已转向 Dense Bi-Encoder |

---

## 相关页面

- ↑ 上级：[1.1 文本嵌入 Text Embeddings](1%20文本嵌入%20Text%20Embeddings.md)
- → 下一节：Dense / Bi-Encoder 句向量（SBERT / E5 / BGE / OpenAI）
- → 对比进阶：[2.1 表示层与距离度量](1%20表示层与距离度量.md)
- → 训练原理：[2.2 训练范式详解](2%20训练范式详解.md)