---
aliases:
  - 词嵌入
  - Word Embedding
  - 静态向量
  - Word2Vec
created: 2025-01-01
updated: 2026-03-28
---

# 静态词向量：Word2Vec、GloVe、FastText

> [!abstract] 摘要
> 静态词向量是文本嵌入的基石，将离散词映射为连续向量空间。Word2Vec、GloVe、FastText 三大经典模型通过共现统计实现词的语义表示，为后续的上下文化嵌入奠定基础。

## 0. 统一概念：什么是静态词向量？

> [!important] 静态词向量的本质
> 静态词向量是最早一批将「词」映射到连续向量空间的方法。每个词对应 **一个固定向量**，不随上下文变化——所以叫"静态"。

> [!tip] 核心思想
> **共现统计 → 低维稠密表示** —— 经常出现在相似上下文中的词，向量距离就近。

**发展历程**：
- 2013: **Word2Vec** - 局部窗口预测
- 2014: **GloVe** - 全局共现矩阵分解
- 2017: **FastText** - 子词 n-gram 组合

---

## 1. 三大经典模型对比

> [!important] 静态词向量三巨头

| 模型 | 核心思路 | 训练方式 | 特点 |
| --- | --- | --- | --- |
| **Word2Vec** | 局部窗口预测 | CBOW（上下文→中心词）或 Skip-gram（中心词→上下文） | 快速；经典；无子词信息 |
| **GloVe** | 全局共现矩阵分解 | 对 log 共现计数做加权最小二乘 | 全局统计 + 局部窗口；类比任务表现好 |
| **FastText** | 子词 n-gram 组合 | Skip-gram + 子词 n-gram 向量求和 | 天然支持 OOV 词；形态丰富语言友好 |

> [!note] 性能对比（词类比任务）
> | 模型 | Google Analogy 任务准确率 |
> |------|--------------------------|
> | Word2Vec | 76% |
> | GloVe | 82% |
> | FastText | 88% |

---

## 2. Word2Vec 详解

### 2.1 Skip-gram 原理

> [!important] Skip-gram 的核心思想
> 给定中心词 $w_c$，预测窗口内上下文词 $w_o$ 的概率：

$$P(w_o \mid w_c) = \frac{\exp(v_{w_o}' \cdot v_{w_c})}{\sum_{w \in V} \exp(v_w' \cdot v_{w_c})}$$

> [!warning] Softmax 的计算瓶颈
> 实际训练用 **负采样（Negative Sampling）** 近似 softmax，大幅降低计算量。

**负采样目标函数**：
$$\mathcal{L} = -\log \sigma(v_{w_o}' \cdot v_{w_c}) - \sum_{k=1}^{K} \log \sigma(-v_{w_k}' \cdot v_{w_c})$$

其中 $w_k$ 是负样本，$\sigma$ 是 sigmoid 函数，$K$ 是负样本数量。

### 2.2 CBOW vs Skip-gram

| 特性 | CBOW (Continuous Bag-of-Words) | Skip-gram |
|------|-------------------------------|-----------|
| **输入** | 上下文词 | 中心词 |
| **输出** | 中心词 | 上下文词 |
| **训练速度** | 快 | 慢 |
| **稀有词效果** | 差 | 好 |
| **语义表示** | 更平滑 | 更精确 |

> [!example] 示例对比
> 句子：`"the quick brown fox jumps"`
> - CBOW：输入 `["quick", "brown", "fox"]`，预测 `"the"`
> - Skip-gram：输入 `"the"`，预测 `["quick", "brown", "fox"]`

### 2.3 Gensim 实现示例

> [!tip] Gensim 是 Python 中最流行的 Word2Vec 实现

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

### 2.4 Word2Vec 的优缺点

| 优点 | 缺点 |
|------|------|
| ✅ 训练速度快 | ❌ 无法处理多义词 |
| ✅ 语义效果好 | ❌ 需要大量语料 |
| ✅ 实现简单 | ❌ 无子词信息 |
| ✅ 可解释性强 | ❌ 上下文信息丢失 |

---

## 3. GloVe 详解

### 3.1 核心思想与算法

> [!important] 全局统计与局部预测的结合
> GloVe (Global Vectors for Word Representation) 结合了全局矩阵分解和局部上下文窗口的优点。

**核心目标函数**：
$$J = \sum_{i,j=1}^{V} f(X_{ij}) \left( w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2$$

其中：
- $X_{ij}$ 是词 $i$ 和词 $j$ 的共现次数
- $f(x)$ 是权重函数，防止高频词对主导
- $w_i, \tilde{w}_j$ 是词向量
- $b_i, \tilde{b}_j$ 是偏置项

### 3.2 权重函数设计

> [!note] 权重函数的作用
> $f(x) = \min(x, x_{\max})^\alpha$，其中：
> - $x_{\max} = 100$：设置最大共现次数
> - $\alpha = 0.75$：衰减因子

**为什么需要权重函数**：
- 高频词（如"the"）共现次数太多，会主导训练
- 低频词（如专有名词）共现次数太少，学习困难
- 权重函数平衡了词频影响

### 3.3 GloVe 的独特优势

1. **语义类比**：$vec_{king} - vec_{man} + vec_{woman} \approx vec_{queen}$
2. **全局统计**：考虑整个语料的词共现关系
3. **局部预测**：保留局部上下文窗口的优势

> [!example] 语义类比示例
> ```python
> # 计算词向量差异
> analogy_vec = glove["king"] - glove["man"] + glove["woman"]
>
> # 找到最接近的词
> similar_words = []
> for word in glove:
>     if word not in ["king", "man", "woman"]:
>         similarity = cosine_similarity(analogy_vec, glove[word])
>         similar_words.append((word, similarity))
>
> # 排序后应该能找到 "queen"
> sorted(similar_words, key=lambda x: x[1], reverse=True)[:5]
> ```

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

## 4. FastText 详解

### 4.1 子词机制

> [!important] 子词 n-gram 的创新
> FastText 把每个词拆成字符 n-gram，词向量 = 所有子词向量之和。

**子词拆分示例**：
- `embedding` (n=3): `"<em", "emb", "mbe", "bed", "edd", "din", "ing", "ng>"`
- `embedding` (n=4): `"<emb", "embe", "mbed", "bedd", "eddin", "ding", "ing>"`
- `embedding` (n=5): `"<embe", "embed", "mbedd", "eddin", "ding>", "ng>"`

> [!tip] 子词边界标记
> 使用 `<` 和 `>` 作为开始和结束标记，避免子词跨词。

### 4.2 OOV 处理能力

> [!warning] 传统方法的局限性
> Word2Vec 和 GloVe 无法处理未登录词（OOV），而 FastText 天然支持。

**OOV 词的处理流程**：
1. 将 OOV 词拆分为子词 n-gram
2. 从预训练的子词表中查找对应向量
3. 将子词向量求和得到完整词向量

```python
# OOV 词示例
oov_word = "embeddable"  # 训练语料中未出现过
subwords = ["<emb", "embe", "embed", "mbedd", "edda", "ddab", "dabl", "able", "ble>"]
vec = sum([model[subword] for subword in subwords if subword in model.wv])
```

### 4.3 FastText 的优势与局限

| 优势 | 局限 |
|------|------|
| ✅ 天然支持 OOV 词 | ❌ 计算开销大（存储大量子词） |
| ✅ 处理形态丰富语言（如芬兰语） | ❌ 子词粒度难以确定 |
| ✅ 对拼写错误鲁棒 | ❌ 稀疏语言中子词过多 |
| ✅ 词汇边界清晰 | ❌ 长词的子词向量可能重复 |

> [!note] FastText 的变种
> - **fasttext supervised**: 有监督分类任务
> - **fasttext cbow**: CBOW 模式的 FastText
> - **fasttext skipgram**: Skip-gram 模式的 FastText

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

### 4.4 Facebook FastText 实现示例

```python
import fasttext

# 训练无监督模型
model = fasttext.train_unsupervised(
    "corpus.txt",       # 每行一句
    model="skipgram",
    dim=100,
    minCount=1,
    minn=3,            # 最短子词长度
    maxn=6,            # 最长子词长度
    epoch=25,
)

print(model.get_word_vector("hello").shape)   # (100,)
print(model.get_nearest_neighbors("hello", k=5))
```

---

## 5. 适用场景与局限性

### 5.1 适用场景分析

| 维度 | 说明 |
| --- | --- |
| ✅ 适合场景 | 词级特征输入、轻量部署、资源受限环境、词类比/聚类、下游任务初始化 |
| ✅ 教学价值 | 理解词嵌入的基本原理，NLP 入门概念 |
| ✅ 轻量级应用 | 移动端、嵌入式设备的语义搜索 |
| ✅ 特定领域 | 专业术语较少的领域、词汇表相对固定 |

### 5.2 核心局限性

| 局限性 | 具体表现 | 影响 |
|--------|----------|------|
| **多义词无法区分** | "bank"（银行 vs 河岸）使用相同向量 | 语义理解不准确 |
| **上下文信息丢失** | 同一个词在不同语境中向量相同 | 无法理解语境依赖 |
| **句子语义需额外聚合** | 需要额外方法处理句子级语义 | 增加复杂度 |
| **已被上下文模型超越** | BERT 等模型性能远超静态词向量 | 主流应用场景受限 |

> [!warning] 当前地位
> **现状**：作为"经典基础"教学和轻量场景仍有价值，主流检索/RAG 已转向 Dense Bi-Encoder。

---

## 6. 与现代嵌入模型的对比

### 6.1 性能对比

| 指标 | Word2Vec/GloVe | SBERT/E5 | BGE |
|------|----------------|----------|-----|
| **多义词处理** | ❌ 单一向量 | ✅ 上下文相关 | ✅ 上下文相关 |
| **长文本支持** | ❌ 词级 | ✅ 句子级 | ✅ 长文档级 |
| **微调能力** | ❌ 不可微调 | ✅ 可微调 | ✅ 可微调 |
| **领域适应** | ❌ 通用 | ✅ 指令微调 | ✅ 领域微调 |

### 6.2 应用场景演进

```
2013-2017: 静态词向量主导
  ↳ 词类比、词相似度、简单分类

2017-2020: 上下文化嵌入兴起
  ↳ SBERT、Sentence-BERT、USE

2020-至今: 指令微调与长文本
  ↳ E5、BGE、Nomic、GTE
```

---

## 相关链接

**所属模块**：[[索引_Embedding与位置编码]]

**前置知识**：
-[[01_BPE_WordPiece_Unigram|子词分词]]] — Token 化基础

**相关主题**：
- [[../02_Dense_Bi-Encoder_句向量_SBERT_E5_BGE|稠密句向量]] — 现代嵌入方法
- [[../03_长上下文文本嵌入_Nomic_Jina|长上下文嵌入]] — 处理长文本

**延伸阅读**：
[[00_缩放点积注意力_为什么是点积_为什么要除以根号dk|注意力机制]]]] — 自注意力原理
- [[../../../11_多模态与跨模态/索引_多模态与跨模态|多模态大模型]] — 跨模态嵌入

