## 一句话概括

Cross-Encoder 是最经典的精排方法，将 query 和 document 拼接后输入同一个 Transformer 做全交叉注意力，输出一个相关性分数。精度最高，但因为无法预计算，速度最慢。

## 核心原理

### 基本架构

`[CLS] query [SEP] document [SEP] → BERT → [CLS] 向量 → 线性层 → 相关性分数`

关键点：query 和 document **拼接在一起作为一个序列**输入 BERT。这意味着 BERT 的 Self-Attention 可以在 query token 和 document token 之间做**全交叉注意力**。

### 为什么精度最高？

对比三种架构的注意力方式：

- **Bi-Encoder（DPR）**：query 和 doc 各自编码，两者之间**零交互**，最后只用一个点积
- **Late Interaction（ColBERT）**：各自编码后，做 **token 级浅层交互**（MaxSim）
- **Cross-Encoder**：从第一层开始，query 和 doc 的每个 token 就在**全面深度交互**

全交叉注意力意味着模型能看到 query 和 doc 之间所有可能的对应关系，理解复杂的语义关联。

### 为什么慢？

问题在于**无法预计算**：

- Bi-Encoder：文档向量可以离线预计算，查询时只需编码 query + ANN 检索
- Cross-Encoder：每个 (query, doc) 对都需要完整的前向传播。如果有 100 个候选文档，就需要运行 100 次 BERT

这就是为什么 Cross-Encoder 只能用作**精排**（对少量候选重排），不能用作**召回**（从百万文档中检索）。

### 训练方式

**分类任务**：将排序问题转化为二分类——输入 (query, doc) 对，输出「相关/不相关」的概率。

$P(\text{relevant} | q, d) = \sigma(W \cdot h_{[CLS]} + b)$

- $h_{[CLS]}$：[CLS] token 的隐藏状态
- $sigma$：Sigmoid 函数
- 训练损失：交叉熵损失

## 工作流程

`召回阶段（BM25/Dense）→ top-50~200 候选 → Cross-Encoder 逐对打分 → 按分数重排 → 保留 top-5~20`

## 代表模型

|模型|基础架构|特点|
|---|---|---|
|ms-marco-MiniLM-L-12-v2|MiniLM (12层)|速度与精度平衡|
|bge-reranker-large|BERT-large|更高精度|

## 优势

- **精度最高**：全交叉注意力，理解最深
- **架构简单**：标准 BERT + 线性层，容易实现
- **对复杂查询鲁棒**：能理解 query 和 doc 之间的细微语义关系
- **可微调**：在特定领域数据上微调效果提升显著

## 局限

- **速度慢**：每对需要完整前向传播，只能用于精排
- **无法扩展到大规模检索**：无法对百万文档逐一打分
- **GPU 密集**：需要 GPU 加速
- **输入长度限制**：BERT 最大 512 token，超长文档需截断

## 加速技巧

- **Batch Scoring**：将多个 (query, doc) 对合并成一个 batch 并行推理
- **模型蒸馏**：用大 Cross-Encoder 的分数训练小模型
- **ONNX / TensorRT**：推理优化
- **早停策略**：对明显不相关的文档提前停止推理

## 与 Bi-Encoder 的互补关系

在 RAG 系统中，Cross-Encoder 和 Bi-Encoder 是**互补关系**而非替代关系：

`Bi-Encoder（速度快，大范围召回）→ Cross-Encoder（精度高，少量精排）`

这就是 Two-Stage RAG 的核心思路。

---

**相关页面**：[Reranker（精排器）](https://www.notion.so/Reranker-c676e27a75fe4518bde38787ee24006e?pvs=21) · [Dense Retrieval（稠密检索）](https://www.notion.so/Dense-Retrieval-8263fa030b7740a98bc72321e43033cf?pvs=21) · [Two-Stage RAG](https://www.notion.so/Two-Stage-RAG-5e5cd2078fa14b5d8e6bc4bcfafc7404?pvs=21)