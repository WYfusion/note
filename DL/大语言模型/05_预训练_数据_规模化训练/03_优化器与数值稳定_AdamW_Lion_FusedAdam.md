# 优化器与数值稳定：AdamW, Lion 与 Fused Kernels

在大规模训练中，选择合适的优化器并保证数值稳定性是成功的关键。

## 1. AdamW (Adam with Weight Decay, 权重衰减Adam)

目前 LLM 训练的默认选择。

### 1.1 核心机制
*   **动量 (Momentum)**: 累积梯度的一阶矩（均值）和二阶矩（方差）。
*   **自适应学习率**: 为每个参数单独调整学习率。
*   **Weight Decay 解耦**: 传统的 L2 正则化在 Adam 中效果不佳。AdamW 将 Weight Decay 直接作用于权重更新步骤，而不是加在梯度上。

### 1.2 典型参数
*   $\beta_1 = 0.9$
*   $\beta_2 = 0.95$ (LLM 常用，比默认的 0.999 小，有助于稳定性)
*   $\epsilon = 1e-8$
*   Weight Decay $\approx 0.1$

## 2. Lion (Evolved Sign Momentum, 进化的符号动量)

Google 提出的新一代优化器，号称比 AdamW 更快、更省显存。

### 2.1 特点
*   **符号函数 (Sign)**: 只使用梯度的符号（方向），忽略大小。
*   **省显存**: 不需要存储二阶矩（方差），显存占用减少约 1/3。
*   **更新公式**:
    $$ \theta_{t+1} = \theta_t - \eta (\text{sign}(m_t) + \lambda \theta_t) $$

## 3. Fused Kernels (算子融合)

为了加速训练，通常使用 NVIDIA Apex 或 PyTorch 自带的 `FusedAdam`。
*   **原理**: 将参数更新的所有操作（读取梯度、计算动量、更新权重）融合为一个 CUDA Kernel，减少 GPU 显存读写次数。

## 4. 语音大模型训练的数值稳定性

语音数据通常比文本数据更不稳定（噪声、静音、动态范围大），容易导致梯度异常。

### 4.1 梯度裁剪 (Gradient Clipping)
*   **必选项**: 防止梯度爆炸。
*   **做法**: 当梯度的 L2 范数超过阈值（如 1.0）时，对梯度进行缩放。
    $$ g \leftarrow g \times \min(1, \frac{\text{threshold}}{||g||}) $$
*   **语音特例**: 在训练 RNN-T 或深层 Conformer 时，梯度爆炸非常频繁，可能需要更激进的裁剪。

### 4.2 混合精度训练 (Mixed Precision)
*   **FP16**: 容易溢出（Overflow）。需要 Loss Scaling。
*   **BF16 (BFloat16)**: 具有与 FP32 相同的动态范围，极大地减少了溢出风险。**强烈推荐**用于语音大模型训练。

### 4.3 长度不均衡问题
*   语音 Batch 中可能包含 1秒的短音频和 30秒的长音频。
*   **Padding**: 大量的 Padding 会浪费计算资源，且可能影响 Batch Norm / Layer Norm 的统计量。
*   **解决方案**:
    *   **Bucket Batching**: 将长度相近的音频分到同一个 Batch。
    *   **Masking**: 确保 Attention 和 Loss 计算严格忽略 Padding 部分。
