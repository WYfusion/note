## 是什么

在 token 级别做延迟交互匹配的检索方法。不像 Dense 把整个文本压缩成一个向量，而是保留每个 token 的向量，在检索时做 token 级别的细粒度匹配。

## 核心原理

`query tokens → 各自编码为向量`

`doc tokens → 各自编码为向量`

`匹配 = 每个 query token 找最相似的 doc token，取最大相似度之和`

## 代表模型：ColBERT

ColBERT（Contextualized Late Interaction over BERT）是延迟交互的代表作。

- **编码**：query 和 doc 分别用 BERT 编码每个 token
- **匹配**：MaxSim 操作——每个 query token 找 doc 中最相似的 token，取相似度之和
- **ColBERTv2**：改进版，压缩存储，实用性大增

## 在检索体系中的位置

`Sparse（BM25）→ Dense（单向量）→ Late Interaction（多向量）→ Cross-Encoder（全交互）`

- 精度：Sparse < Dense < Late Interaction < Cross-Encoder
- 速度：Sparse > Dense > Late Interaction > Cross-Encoder

## 优势

- 精度高于 Dense（单向量压缩丢失信息）
- 速度远快于 Cross-Encoder（可预计算文档向量）
- token 级匹配更细粒度

## 劣势

- 存储开销大（每个 token 一个向量）
- 索引构建复杂
- 模型选择少（主要就 ColBERT 系列）

## 适用场景

- 对精度要求高但 Cross-Encoder 太慢的场景
- 作为召回阶段的高质量替代方案
- 需要 token 级别匹配的细粒度检索

---

## 📖 深入了解模型原理

- [[ColBERT]] — MaxSim 操作、ColBERTv2 压缩与 PLAID 引擎
