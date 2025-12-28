# Pre-LN 与 Post-LN 的稳定性分析

在 Transformer 架构的演进中，Layer Normalization (LN) 的位置选择（Pre-LN vs Post-LN）对模型的训练稳定性起着决定性作用。

## 1. 结构对比

### 1.1 Post-LN (原始 Transformer / BERT)
$$ x_{t+1} = \text{LN}(x_t + \text{Sublayer}(x_t)) $$
*   **特点**: LN 放在残差连接之后。
*   **梯度流**: 梯度经过 LN 时会被归一化，导致反向传播到底层的梯度变小。
*   **问题**: 在初始化阶段，输出层的方差会随着层数 $L$ 线性增长，导致梯度爆炸或消失。必须配合 Warmup 才能训练。

### 1.2 Pre-LN (GPT-2 / Llama / Whisper)
$$ x_{t+1} = x_t + \text{Sublayer}(\text{LN}(x_t)) $$
*   **特点**: LN 放在子层输入之前，残差连接直接相加。
*   **梯度流**: 存在一条直通的“高速公路”（Identity Path），梯度可以直接传到底层。
*   **优势**: 训练极其稳定，甚至不需要 Warmup 也能收敛。

## 2. 理论分析：为什么 Post-LN 难训练？

### 2.1 梯度范数分析
对于 Post-LN，最后一层的梯度范数 $\approx \sqrt{L}$，而第一层的梯度范数 $\approx 1/\sqrt{L}$。
这意味着深层参数更新快，浅层参数更新慢，导致训练初期极不稳定。

### 2.2 Pre-LN 的梯度范数
Pre-LN 的梯度范数在各层之间保持相对一致（$\approx 1$），这使得使用较大的学习率成为可能。

## 3. 语音大模型中的选择

### 3.1 语音信号的特殊性
语音特征（如 Mel-spectrogram）的动态范围通常比文本 Embedding 大，且包含更多噪声。

### 3.2 Whisper 的选择
Whisper 明确选择了 **Pre-LN** 结构。
*   **原因**: 语音识别任务对对齐（Alignment）非常敏感。Pre-LN 提供的稳定梯度流有助于 Attention 机制快速学习到正确的对齐关系。
*   **效果**: 即使在没有复杂 Warmup 策略的情况下，也能在 68 万小时的含噪数据上稳定收敛。

### 3.3 Conformer 的选择
Conformer（ASR 中常用的 Encoder）通常也采用 Pre-LN 结构，或者使用由 Pre-LN 演变而来的 **Sandwich-Norm**（在残差分支上也加 LN）来进一步增强稳定性。
