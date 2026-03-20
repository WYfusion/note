# Transformer 总览：数据流与张量形状

深入理解 Transformer 的第一步是搞清楚数据在模型内部是如何流动的，以及每一步的张量形状（Tensor Shape）是如何变化的。这对于调试代码和理解显存占用至关重要。

## 1. 符号定义

为了保持严谨，我们统一使用以下符号：

*   $B$: Batch Size (批次大小，例如 32)
*   $L$: Sequence Length (序列长度，例如 512 或 2048)
*   $D$ / $d_{model}$: Embedding Dimension (隐藏层维度，例如 768, 4096)
*   $V$: Vocabulary Size (词表大小，例如 32000, 128000)
*   $H$: Number of Heads (注意力头数，例如 12, 32)
*   $d_k$: Head Dimension (每个头的维度，通常 $d_k = D/H$)

## 2. 宏观数据流 (The Big Picture)

Transformer 的核心处理流程可以概括为：
**离散 ID $\to$ 连续向量 $\to$ 层层变换 $\to$ 概率分布**

### 2.1 输入阶段 (Input Stage)
*   **输入**: `Input IDs`
    *   形状: `[B, L]` (整数)
    *   *例子*: `[[101, 2054, ...], [101, 3021, ...]]`
*   **Embedding**: 查表操作
    *   操作: `Embedding(Input IDs)`
    *   形状: `[B, L] -> [B, L, D]`
    *   *注*: 对于**语音大模型**，输入不是 ID，而是连续的声学特征（如 Mel-spectrogram），形状通常是 `[B, T, F]` (T帧数, F频带数)，经过卷积层后变为 `[B, L, D]`。

### 2.2 骨干网络 (Backbone)
由 $N$ 个 Transformer Block 堆叠而成。输入和输出形状保持不变（Residual Connection 的要求）。
*   **输入**: `Hidden States` `[B, L, D]`
*   **Block 1**: `[B, L, D] -> [B, L, D]`
*   ...
*   **Block N**: `[B, L, D] -> [B, L, D]`

### 2.3 输出阶段 (Output Stage)
*   **Unembedding / LM Head**: 将向量映射回词表空间。
    *   操作: `Linear(Hidden States)`
    *   形状: `[B, L, D] @ [D, V] -> [B, L, V]`
*   **Softmax**: 计算概率。
    *   形状: `[B, L, V]` (概率分布)

---

## 3. 单个 Transformer Block 内部细节

一个标准的 Block 包含两个主要子层：Attention 和 FFN。

### 3.1 Attention 层
1.  **Q, K, V 投影**:
    *   输入: `x` `[B, L, D]`
    *   权重: $W_Q, W_K, W_V$ `[D, D]`
    *   输出: `q, k, v` `[B, L, D]`
2.  **拆分多头 (Split Heads)**:
    *   变形: `[B, L, D] -> [B, L, H, d_k]`
    *   转置: `[B, H, L, d_k]` (为了让 H 维度参与并行计算)
3.  **Attention Score**:
    *   操作: $QK^T$
    *   形状: `[B, H, L, d_k] @ [B, H, d_k, L] -> [B, H, L, L]`
    *   *注*: 这个 `[L, L]` 矩阵就是显存杀手，对于长音频（L=10000+）如果不优化会直接 OOM。
4.  **Weighted Sum**:
    *   操作: $Score \cdot V$
    *   形状: `[B, H, L, L] @ [B, H, L, d_k] -> [B, H, L, d_k]`
5.  **合并多头 (Merge Heads)**:
    *   转置 + 变形: `[B, L, H, d_k] -> [B, L, D]`
6.  **输出投影 (Output Projection)**:
    *   权重: $W_O$ `[D, D]`
    *   形状: `[B, L, D]`

### 3.2 FFN 层 (Feed-Forward Network)
通常是一个两层的 MLP，中间层维度膨胀（通常是 $4D$ 或 $8/3 D$）。
1.  **Up Projection**:
    *   输入: `[B, L, D]`
    *   权重: `[D, 4D]`
    *   输出: `[B, L, 4D]`
2.  **Activation**: ReLU / GELU / SwiGLU
    *   形状: `[B, L, 4D]`
3.  **Down Projection**:
    *   权重: `[4D, D]`
    *   输出: `[B, L, D]`

---

## 4. 语音大模型的特殊性

虽然核心结构相同，但语音数据带来的张量变化主要体现在 **L (Sequence Length)** 上。

*   **文本**: 1000 个 Token 大约对应 700 个单词。
*   **语音**: 1 秒音频 (16kHz) 对应 16000 个采样点。
    *   即使经过卷积下采样（如 stride=320），10 秒音频也可能产生 500 个 Frame。
    *   Whisper 的 Encoder 输入限制为 30秒 (3000 frames)。
    *   **长音频挑战**: 处理 1 小时录音时，L 极大，必须使用 **FlashAttention** 或 **Window Attention** 来避免 `[L, L]` 矩阵爆炸。
