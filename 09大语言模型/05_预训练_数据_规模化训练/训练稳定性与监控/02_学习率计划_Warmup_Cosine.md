# 学习率计划 (Learning Rate Schedule)

学习率是训练中最重要的超参数。

## 1. Warmup (热身)

在训练初期，线性地将学习率从 0 增加到最大值 $LR_{max}$。

### 1.1 为什么需要 Warmup?
*   **Transformer 初始化**: 初始阶段梯度方差较大，直接用大 LR 容易导致模型发散（尤其是 Post-LN 结构）。
*   **Adam 统计量**: Adam 需要一段时间来积累准确的一阶和二阶矩估计。

### 1.2 语音模型的 Warmup
*   **Wav2Vec 2.0**: 前 10% - 30% 的 Step 都在 Warmup。
*   **Whisper**: 2048 个 Warmup Steps。

## 2. Decay (衰减)

Warmup 结束后，学习率需要逐渐降低。

### 2.1 Cosine Decay (余弦衰减)
目前最主流的策略。
$$ \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t}{T}\pi)) $$
*   **特点**: 下降平滑，最后收敛到很小的值。

### 2.2 Linear Decay (线性衰减)
简单直观，但在训练后期可能不够细腻。

### 2.3 Noam Scheduler
Transformer 论文提出的策略。
$$ LR = d_{model}^{-0.5} \cdot \min(step^{-0.5}, step \cdot warmup^{-1.5}) $$
*   **特点**: 自动根据模型宽度调整 LR。

## 3. Tri-stage Scheduler (三阶段调度)

在语音识别工具包（如 ESPnet）中非常流行。

1.  **Warmup**: 线性增加。
2.  **Hold**: 保持 $LR_{max}$ 不变一段时间。
3.  **Decay**: 线性或指数衰减。

**原因**: 语音任务通常需要较长时间的“强学习”阶段（Hold 阶段），让模型充分拟合声学特征，然后再精细调整。
