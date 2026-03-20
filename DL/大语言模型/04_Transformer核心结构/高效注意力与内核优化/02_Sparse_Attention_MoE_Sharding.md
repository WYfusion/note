# Sparse Attention, MoE 与 Sharding

当模型规模和序列长度进一步增长时，我们需要更激进的稀疏化和并行化策略。

## 1. Sparse Attention (稀疏注意力)

全注意力机制计算所有 Token 对之间的关系，很多时候是冗余的。

### 1.1 常见模式
*   **Local Attention (Sliding Window)**: 只关注周围 $w$ 个 Token。适合语音这种局部性强的信号。
*   **Global Attention**: 选定几个特殊的 Token（如 `[CLS]`）关注全局。
*   **Random Attention**: 随机关注一些 Token。
*   **BigBird**: 结合了 Local + Global + Random。

### 1.2 语音中的应用
*   **Conformer**: 结合了 CNN (局部特征) 和 Transformer (全局特征)。
*   **Emformer**: 使用 Block-wise 的注意力机制，实现流式处理。

## 2. MoE (Mixture of Experts, 混合专家)

### 2.1 原理
将 FFN 层替换为多个 Expert 网络（通常也是 FFN）。
$$ y = \sum_{i=1}^N G(x)_i E_i(x) $$
其中 $G(x)$ 是门控网络 (Gating Network)，决定输入 $x$ 由哪些 Expert 处理（通常只选 Top-2）。

### 2.2 优势
*   **参数量巨大，计算量不变**: 可以将模型参数扩展到万亿级别，但每次推理只激活一小部分参数。
*   **Switch Transformer / Mixtral**: 著名的 MoE 模型。

## 3. Sharding (分片 / 并行)

单卡显存放不下大模型时，需要并行策略。

### 3.1 Data Parallelism (DP)
复制模型到多张卡，每张卡处理不同的数据 Batch。

### 3.2 Tensor Parallelism (TP)
将矩阵乘法切分到多张卡上计算。
*   **Megatron-LM**: 经典的 TP 实现。

### 3.3 Pipeline Parallelism (PP)
将模型的不同层切分到不同卡上，像流水线一样处理。

### 3.4 ZeRO / FSDP (Fully Sharded Data Parallel)
将优化器状态、梯度、参数切分到所有卡上。
*   **DeepSpeed**: 微软推出的深度学习优化库，集成了 ZeRO 技术。
