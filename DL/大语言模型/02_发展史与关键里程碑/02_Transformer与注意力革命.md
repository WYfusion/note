# Transformer 与注意力革命

2017 年，Google 团队发表了划时代的论文 **"Attention Is All You Need"**，提出了 Transformer 架构。它彻底抛弃了 RNN 和 CNN 的循环/卷积结构，完全基于注意力机制（Self-Attention），实现了训练的并行化，开启了大模型时代。

## 1. 为什么需要 Transformer？

RNN (LSTM/GRU) 存在两个致命弱点：
1.  **无法并行 (Sequential Computation)**: 计算 $h_t$ 必须等待 $h_{t-1}$，导致在 GPU 上训练效率极低。
2.  **长距离依赖 (Long-term Dependency)**: 尽管有 LSTM，但信息在长序列传递中仍会衰减。

Transformer 通过 **Self-Attention** 机制，让序列中的每个词都能**同时**关注到其他所有词，解决了这两个问题。

---

## 2. 核心组件：Scaled Dot-Product Attention

这是 Transformer 的心脏。

### 2.1 Q, K, V 的定义
对于输入矩阵 $X$（每一行是一个词向量），我们通过三个线性变换得到 Query, Key, Value 矩阵：
$$
Q = X W^Q, \quad K = X W^K, \quad V = X W^V
$$
*   **Query ($Q$)**: 查询向量。代表当前词“想找什么”。
*   **Key ($K$)**: 键向量。代表当前词“有什么特征”。
*   **Value ($V$)**: 值向量。代表当前词的“内容信息”。

### 2.2 注意力计算公式
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**推导步骤**:
1.  **相似度计算 ($QK^T$)**: 计算每个 Query 和所有 Key 的点积。点积越大，相关性越高。得到一个 $N \times N$ 的分数矩阵。
2.  **缩放 ($\frac{1}{\sqrt{d_k}}$)**: 除以 $\sqrt{d_k}$（Key 的维度）。
    *   *原因*: 当 $d_k$ 很大时，点积结果会很大，导致 Softmax 进入饱和区（梯度接近 0）。缩放是为了让梯度更稳定。
3.  **归一化 (Softmax)**: 将分数转换为概率分布（权重和为 1）。
4.  **加权求和 ($\cdot V$)**: 用权重对 Value 向量进行加权求和，得到最终的上下文表示。

---

## 3. Multi-Head Attention (多头注意力)

单头注意力只能关注一种类型的相关性。为了捕捉多维度的特征（如语法关系、语义指代、位置关系），Transformer 使用了多头机制。

**公式**:
$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O
$$
$$
\text{where } \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

*   将 $d_{model}$ 维的向量拆分为 $h$ 个头，每个头维度 $d_k = d_{model} / h$。
*   每个头独立计算 Attention，最后拼接并通过线性层 $W^O$ 融合。

---

## 4. 位置编码 (Positional Encoding)

Self-Attention 是**置换不变 (Permutation Invariant)** 的。如果你打乱句子中词的顺序，Attention 的输出结果是一样的（除了顺序变化）。

为了让模型理解“顺序”，必须显式加入位置信息。原论文使用了正弦/余弦编码：

$$
PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})
$$
$$
PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})
$$

*   $pos$: 词在句子中的位置。
*   $i$: 向量维度的索引。
*   这种编码方式允许模型通过线性变换学习相对位置关系（因为 $\sin(\alpha+\beta)$ 可以展开）。

*(注：现代 LLM 多使用 RoPE 旋转位置编码，见 `04_Transformer核心结构`)*

---

## 5. 整体架构 (Encoder-Decoder)

原始 Transformer 是为机器翻译设计的 Encoder-Decoder 架构。

### 5.1 Encoder (编码器)
*   由 $N$ 个相同的层堆叠而成。
*   每层包含两个子层：
    1.  **Multi-Head Self-Attention**
    2.  **Position-wise Feed-Forward Networks (FFN)**: 两个线性层中间夹一个 ReLU。
        $$ \text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2 $$
*   **残差连接 (Residual Connection)** 和 **层归一化 (Layer Normalization)** 应用于每个子层：
    $$ \text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x)) $$

### 5.2 Decoder (解码器)
*   也由 $N$ 个层堆叠。
*   包含三个子层：
    1.  **Masked Multi-Head Self-Attention**: 掩码注意力。保证预测第 $t$ 个词时，只能看到 $t$ 之前的词（不能偷看答案）。
    2.  **Encoder-Decoder Attention**: Query 来自 Decoder，Key/Value 来自 Encoder。让解码器关注原文。
    3.  **FFN**。

---

## 6. 架构的演变：Encoder-only, Decoder-only

随着发展，Transformer 分化出了三种主流架构：

1.  **Encoder-only (如 BERT)**:
    *   只使用 Encoder。
    *   **双向注意力**：能同时看到上下文。
    *   适合：文本分类、实体识别、情感分析（理解任务）。

2.  **Decoder-only (如 GPT 系列, Llama)**:
    *   只使用 Decoder（去掉中间的 Cross-Attention）。
    *   **单向注意力 (Causal Mask)**：只能看到左边的词。
    *   适合：文本生成（生成任务）。**这是目前 LLM 的主流架构。**

3.  **Encoder-Decoder (如 T5, BART)**:
    *   保留完整架构。
    *   适合：翻译、摘要（Seq2Seq 任务）。

## 7. 总结

Transformer 的成功在于：
1.  **并行计算**：极大地提升了训练速度，使得在海量数据上训练超大模型成为可能。
2.  **全局视野**：Self-Attention 直接捕捉长距离依赖，无视距离限制。
3.  **通用性**：不仅用于 NLP，还扩展到了 CV (ViT) 和多模态领域。
