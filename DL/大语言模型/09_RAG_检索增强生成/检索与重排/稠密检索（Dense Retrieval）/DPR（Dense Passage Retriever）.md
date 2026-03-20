## 一句话概括

DPR 是 Meta（Facebook）于 2020 年提出的开创性双编码器稠密检索模型，首次证明了纯神经网络检索可以超越 BM25，开启了稠密检索时代。

## 核心原理

### 基本思路

用两个独立的 BERT 编码器，分别把 query 和 passage 编码成固定长度的向量，然后用向量间的点积（或余弦相似度）衡量相关性。

### 架构

`Query → Query Encoder (BERT) → query 向量 q ∈ R^768`

`Passage → Passage Encoder (BERT) → passage 向量 p ∈ R^768`

`相关性分数 = q · p（点积）`

两个编码器**独立运行**，互不干扰。这意味着：

- 所有 passage 可以**离线预编码**存入向量库
- 查询时只需编码 query，然后做 ANN 检索

### 为什么叫「双编码器」？

因为 query 和 passage 各有一个编码器。与之对比：

- **Cross-Encoder**：把 query 和 passage 拼在一起输入一个编码器（精度高但速度慢）
- **DPR（Bi-Encoder）**：分开编码（速度快，可预计算）

### 训练方式

使用**对比学习**：

- **正样本**：query 与其正确答案所在的 passage
- **负样本**：同 batch 内其他 query 的 passage（in-batch negatives）+ BM25 召回的高分但不正确的 passage（hard negatives）
- **损失函数**：让正样本的点积分数尽量高，负样本的点积分数尽量低

$L = -\log \frac{e^{q \cdot p^+}}{e^{q \cdot p^+} + \sum_i e^{q \cdot p_i^-}}$

### Hard Negatives 的重要性

DPR 论文的一个关键发现：**用 BM25 检索到的高分但不正确的 passage 作为难负样本，比随机负样本效果好很多**。因为这些 passage 在词面上与 query 相关，迫使模型学习更深层的语义区分能力。

## 检索流程

**离线阶段**

`所有 passage → Passage Encoder → 向量 → 存入 FAISS / 向量数据库`

**在线阶段**

`Query → Query Encoder → 向量 → FAISS ANN 检索 → 返回 top-k passage`

## 优势

- **开创性工作**：首次证明密集检索可以超越 BM25
- **语义理解**：「汽车」和「轿车」可以匹配
- **高效检索**：passage 预编码 + ANN 检索，毫秒级响应
- **架构简洁**：两个标准 BERT，容易理解和实现

## 局限

- **精确匹配弱**：对编号、专有名词等精确匹配不如 BM25
- **训练数据依赖**：需要 query-passage 标注对
- **单向量瓶颈**：把整段文本压缩成一个向量，信息损失不可避免
- **域迁移差**：在一个领域训练的模型，换到另一个领域效果可能下降明显

## 历史地位

DPR 是稠密检索领域的**奠基之作**。后续的 Contriever、E5、BGE 等模型都在其基础上改进：

- Contriever → 解决了 DPR 需要标注数据的问题
- E5 / BGE / GTE → 在效果和通用性上大幅超越

## 工程实现

- Hugging Face：`facebook/dpr-question_encoder-single-nq-base`
- FAISS：Meta 开源的高效向量检索库
- Pyserini / Tevatron：学术检索框架

---

**相关页面**：[Dense Retrieval（稠密检索）](https://www.notion.so/Dense-Retrieval-8263fa030b7740a98bc72321e43033cf?pvs=21) · [Sparse Retrieval（稀疏检索）](https://www.notion.so/Sparse-Retrieval-dcc60a4e987541589511862507f5e2a7?pvs=21) · [Late Interaction（延迟交互）](https://www.notion.so/Late-Interaction-e66d107ed562470694c2e9bc5f017a45?pvs=21)