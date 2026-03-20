---
tags:
  - 机器学习
  - 深度学习
  - Transformer
  - 学习指引
  - 阅读顺序
created: 2025-01-18
modified: 2025-01-18
difficulty: 中等
related:
  - [[Transformer/01_任务与范式/01_符号与问题定义]]
  - [[Transformer/02_输入表示]]
  - [[Transformer/03_注意力机制]]
  - [[Transformer/04_模块与整体结构]]
  - [[Transformer/05_训练与推理]]
  - [[Transformer/06_效率与扩展]]
  - [[Transformer/07_序列对齐与CTC]]
---

# Transformer 学习指引

## 这份笔记怎么读

### 顺序建议

从 **01_任务与范式** 开始，先搞清符号、因果假设与任务形式，再读 **02_输入表示**、**03_注意力机制**，最后到 **04_模块与整体结构**、**05_训练与推理**、**06_效率与扩展**。

对于语音或不定长序列对齐任务，请参阅 **07_序列对齐与CTC**。

### 纵深关系

每个章节先给直观描述，再落到公式/流程，最后给常见变体与注意点，便于由浅入深。

### 配套旧笔记映射

旧文件 `Transformer.md`/`Multi-Head Self-Attention.md`/`Embedding Patched.md` 等里的图示仍可参考，必要时可在对应新章节插入 `![[...]]` 的引用。

### 约定

除特别说明：
- `L` 为序列长度
- `d_model` 为模型宽度
- `d_k/d_v` 为单头维度
- `h` 为头数
- `N` 为层数

### 目标

形成一份逻辑连贯的"从输入到输出"的路径，对齐当前主流 Transformer 实践（含多查询注意力、截断注意力、非自回归等扩展）。

---

## 学习路径

### 第一阶段：基础概念

1. **01_任务与范式**：理解 Transformer 能解决什么问题
   - [[Transformer/01_任务与范式/01_符号与问题定义]]
   - [[Transformer/01_任务与范式/02_自回归与非自回归]]

### 第二阶段：输入处理

2. **02_输入表示**：理解如何将输入转换为模型可处理的表示
   - [[Transformer/02_输入表示/01_分词与词表]]
   - [[Transformer/02_输入表示/02_词嵌入与输入层]]
   - [[Transformer/02_输入表示/03_位置编码]]
   - [[Transformer/02_输入表示/04_Patched_Embedding]]
   - [[Transformer/02_输入表示/05_位置编码的原因-置换等变性]]

### 第三阶段：核心机制

3. **03_注意力机制**：深入理解 Transformer 的核心创新
   - [[Transformer/03_注意力机制/01_自注意力基础]]
   - [[Transformer/03_注意力机制/02_多头自注意力]]
   - [[Transformer/03_注意力机制/03_掩码与因果性]]
   - [[Transformer/03_注意力机制/04_交叉注意力]]
   - [[Transformer/03_注意力机制/05_多查询注意力]]
   - [[Transformer/03_注意力机制/06_分组注意力]]
   - [[Transformer/03_注意力机制/07_局部与截断注意力]]

### 第四阶段：网络结构

4. **04_模块与整体结构**：理解 Transformer 的完整架构
   - [[Transformer/04_模块与整体结构/01_Encoder_Block]]
   - [[Transformer/04_模块与整体结构/02_Decoder_Block]]
   - [[Transformer/04_模块与整体结构/03_Encoder_Decoder_transformer流程]]
   - [[Transformer/04_模块与整体结构/04_归一化与前馈层]]

### 第五阶段：训练与推理

5. **05_训练与推理**：理解如何训练和使用 Transformer
   - [[Transformer/05_训练与推理/01_训练目标与数据流]]
   - [[Transformer/05_训练与推理/02_推理与解码]]

### 第六阶段：优化与扩展

6. **06_效率与扩展**：了解 Transformer 的改进和变体
   - [[Transformer/06_效率与扩展/01_复杂度与优化思路]]
   - [[Transformer/06_效率与扩展/02_模型变体]]

### 第七阶段：特殊应用

7. **07_序列对齐与CTC**：了解语音等特殊场景的处理
   - [[Transformer/07_序列对齐与CTC/01_背景_不定长序列对齐]]
   - [[Transformer/07_序列对齐与CTC/02_核心机制_Blank与路径映射]]
   - [[Transformer/07_序列对齐与CTC/03_算法原理_前向后向计算]]
   - [[Transformer/07_序列对齐与CTC/04_解码策略_从Greedy到BeamSearch]]
   - [[Transformer/07_序列对齐与CTC/05_CTC_vs_Attention_与总结]]

---

## 快速参考

### 核心符号

| 符号 | 含义 |
|------|------|
| $x_{1:L}$ | 输入序列 |
| $y_{1:L'}$ | 输出或目标序列 |
| $d_{model}$ | 模型宽度 |
| $d_k, d_v$ | 单头维度 |
| $h$ | 头数 |
| $N$ | 层数 |
| $W^Q, W^K, W^V$ | 注意力权重矩阵 |

### 常见模型

- **BERT**：Encoder-only，双向理解
- **GPT**：Decoder-only，自回归生成
- **T5/BART**：Encoder-Decoder，序列到序列

---

## 学习建议

### 初学者

1. 先理解自注意力机制的核心思想
2. 掌握多头注意力的原理
3. 理解 Encoder 和 Decoder 的区别
4. 实践：尝试实现一个简单的 Transformer

### 进阶者

1. 深入研究各种变体（多查询注意力、分组注意力等）
2. 理解训练技巧（学习率预热、正则化等）
3. 研究特定领域的应用（NLP、CV、Speech 等）
4. 实践：基于预训练模型进行微调

---

## 参考资料

- Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*
