# Multi-Head Attention (多头注意力) 实现细节

单头注意力可能只能捕捉到一种类型的依赖关系（例如语法依赖）。为了增强模型的表达能力，Transformer 引入了多头机制。

## 1. 核心思想

**"三个臭皮匠，顶个诸葛亮"**。
将 Embedding 空间切分为 $H$ 个子空间，每个子空间独立计算 Attention，最后再拼接起来。

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_H)W^O
$$
$$
\text{where } \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

## 2. 参数量分析

假设 $d_{model} = 512, H = 8$，则每个头的维度 $d_k = 512 / 8 = 64$。

*   **投影矩阵**:
    *   $W_Q, W_K, W_V$: 形状均为 $[d_{model}, d_{model}]$。
    *   注意：虽然逻辑上是分头的，但在实现时通常是一个大矩阵一次性投影，然后再 reshape。
*   **输出矩阵**:
    *   $W_O$: 形状 $[d_{model}, d_{model}]$。

**总参数量**: $4 \times d_{model}^2$ (忽略 bias)。

## 3. 为什么多头有效？

1.  **多视角 (Diversity)**: 不同的头可以关注不同的模式。
    *   Head 1 可能关注局部信息（如相邻词）。
    *   Head 2 可能关注长距离依赖（如主谓一致）。
    *   Head 3 可能关注特定语义（如指代消解）。
2.  **鲁棒性**: 即使某个头“走神”了，其他头还能补充信息。

## 4. 语音大模型中的多头冗余性

研究发现，在语音模型（如 Wav2Vec 2.0, Whisper）中，许多 Attention Head 是**冗余**的。
*   **对角线头 (Diagonal Heads)**: 许多头只关注当前帧附近的几帧（类似于卷积）。
*   **全局头 (Global Heads)**: 少数头关注整个音频序列。
*   **剪枝 (Pruning)**: 实验表明，在推理时剪掉一半以上的头，语音识别的准确率几乎不下降。这为语音模型在端侧设备的加速提供了空间。

## 5. Grouped-Query Attention (GQA)

在 Llama 2/3 和许多现代模型中，为了减少 KV Cache 的显存占用（特别是在长序列推理时），使用了 GQA。

*   **MHA (Multi-Head)**: $H$ 个 Query, $H$ 个 Key, $H$ 个 Value。
*   **MQA (Multi-Query)**: $H$ 个 Query, **1** 个 Key, **1** 个 Value。(所有头共享 KV)
*   **GQA (Grouped-Query)**: $H$ 个 Query, $G$ 个 Key, $G$ 个 Value。($H$ 是 $G$ 的倍数，如 8 个 Q 共享 1 个 KV)。

**优势**: 保持了 MHA 的大部分性能，同时将推理时的显存占用和带宽需求降低了 $H/G$ 倍。这对于长音频处理（如 1小时录音摘要）至关重要。
