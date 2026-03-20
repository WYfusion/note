# 自回归(AR)、自编码(AE)与序列到序列(Seq2Seq)对比

在语言建模领域，根据训练目标和注意力机制的不同，主要分为三大流派。理解它们的区别对于选择合适的模型架构至关重要。

## 1. 自回归模型 (Auto-Regressive, AR)

AR 模型也被称为 **Decoder-only** 模型。

### 1.1 核心思想
利用上文预测下一个 Token。这是标准的“语言模型”定义。
$$ P(x) = \prod_{t=1}^T P(x_t | x_{<t}) $$

### 1.2 架构特点
*   **单向注意力 (Causal Mask)**: 在计算 $x_t$ 的注意力时，强制 Mask 掉 $t$ 之后的所有 Token。
*   **代表模型**: GPT 系列, Llama, Qwen, Claude。

### 1.3 优缺点
*   **优点**: 天然适合**生成**任务（文本续写）。
*   **缺点**: 只能利用单向上下文，无法同时看到“下文”，在理解任务（如分类）上理论上弱于双向模型。

---

## 2. 自编码模型 (Auto-Encoding, AE)

AE 模型也被称为 **Encoder-only** 模型。

### 2.1 核心思想
利用上下文（上文 + 下文）来还原被 Mask 掉的 Token。
$$ P(x_t | x_{\setminus t}) $$

### 2.2 架构特点
*   **双向注意力 (Bidirectional Attention)**: 每个 Token 都能看到整个句子中的其他所有 Token。
*   **预训练任务**: Masked Language Modeling (MLM)。
*   **代表模型**: BERT, RoBERTa。

### 2.3 优缺点
*   **优点**: 具备**全局视野**，理解能力极强，适合判别式任务（情感分析、NER、抽取）。
*   **缺点**: 预训练和推理不一致（推理时没有 [MASK]），且不擅长生成长文本（因为生成是自回归的过程）。

---

## 3. 序列到序列模型 (Sequence-to-Sequence, Seq2Seq)

Seq2Seq 模型也被称为 **Encoder-Decoder** 模型。

### 3.1 核心思想
将输入序列编码为向量，再解码为输出序列。它是 AR 和 AE 的结合体。

### 3.2 架构特点
*   **Encoder**: 使用双向注意力，负责“理解”输入。
*   **Decoder**: 使用单向注意力，负责“生成”输出。
*   **Cross-Attention**: Decoder 每一层都会关注 Encoder 的输出。
*   **预训练任务**: Span Corruption (T5) 或 Denoising (BART)。
*   **代表模型**: T5, BART, GLM (早期版本)。

### 3.3 优缺点
*   **优点**: 兼顾了理解和生成，特别适合**翻译**、**摘要**等输入输出有强对应关系的任务。
*   **缺点**: 架构复杂，参数效率不如 Decoder-only（在相同参数量下，Decoder-only 的 Zero-shot 能力通常更强）。

---

## 4. 总结对比表

| 特性 | 自回归 (AR) | 自编码 (AE) | 序列到序列 (Seq2Seq) |
| :--- | :--- | :--- | :--- |
| **架构** | Decoder-only | Encoder-only | Encoder-Decoder |
| **注意力** | 单向 (Causal) | 双向 (Full) | 双向 (Enc) + 单向 (Dec) |
| **代表作** | GPT, Llama | BERT | T5, BART |
| **优势** | 文本生成 (Generation) | 文本理解 (Understanding) | 翻译, 摘要 (Translation) |
| **主流地位** | ⭐⭐⭐⭐⭐ (目前统治 LLM) | ⭐⭐⭐ (特定领域仍用) | ⭐⭐⭐ (特定任务仍用) |

### 为什么 AR (GPT) 赢了？
虽然 AE 理解能力更强，但随着模型规模扩大 (Scaling)，AR 模型展现出了惊人的**涌现能力**和**通用性**。通过 Prompt Engineering，AR 模型可以很好地完成理解任务，而 AE 模型很难完成生成任务。因此，通用大模型几乎都选择了 AR 架构。
