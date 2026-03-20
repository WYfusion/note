# Attention Mask: Causal, Padding 与 Audio Masking

Mask（掩码）是 Transformer 中控制信息流动的关键机制。它决定了“谁能看谁，谁不能看谁”。

## 1. Padding Mask (填充掩码)

### 1.1 问题背景
Batch 训练时，不同样本长度不一。我们需要用 `[PAD]` Token (通常 ID 为 0) 将短序列补齐到 Batch 中的最大长度。
*   样本 A: `[101, 20, 30]` (长度 3)
*   样本 B: `[101, 55, 66, 77, 88]` (长度 5)
*   Padding 后 A: `[101, 20, 30, 0, 0]`

### 1.2 作用
Attention 机制不应该关注这些无意义的 `0`。
我们需要构建一个 Mask 矩阵，在 `[PAD]` 的位置填入 `-inf`。
$$ \text{Softmax}([0.8, 0.1, -inf, -inf]) \to [0.6, 0.4, 0, 0] $$

### 1.3 语音中的 Padding
音频通常以 Batch 形式输入，长度差异巨大（有的 2秒，有的 30秒）。
*   **Feature Masking**: 在卷积层之后，必须记录真实的音频长度，并在 Attention 中 Mask 掉填充的静音帧。否则模型会把填充的静音误认为是停顿，导致幻觉（如 Whisper 在静音段疯狂输出 "Thank you"）。

## 2. Causal Mask (因果掩码 / Look-ahead Mask)

### 2.1 作用
用于 Decoder-only 模型（GPT）或 Seq2Seq 的 Decoder。
**目标**: 预测第 $t$ 个词时，只能看到 $1 \dots t-1$ 的词，绝不能偷看 $t+1$ 及其之后的词。

### 2.2 实现
使用一个上三角矩阵（Upper Triangular Matrix），对角线及以下为 0，上方为 `-inf`。

$$
\text{Mask} = \begin{bmatrix} 
0 & -\infty & -\infty \\
0 & 0 & -\infty \\
0 & 0 & 0 
\end{bmatrix}
$$

## 3. 语音特有的 Masking 策略

### 3.1 SpecAugment
这不是 Attention Mask，而是数据增强 Mask。
在训练时，随机 Mask 掉频谱图上的：
*   **时间段 (Time Masking)**: 遮盖某几帧。
*   **频带 (Frequency Masking)**: 遮盖某几个频率。
这迫使模型不依赖单一特征，增强鲁棒性。

### 3.2 Chunk-wise Masking (流式语音识别)
为了实现**实时 (Streaming)** 语音识别，模型不能等待整句话说完再处理。
*   **策略**: 将 Attention 限制在一个固定的窗口（Chunk）内，或者只允许看过去 $N$ 帧。
*   **Latency vs Accuracy**: 窗口越小，延迟越低，但缺少上下文导致准确率下降。

### 3.3 VALL-E 的 Masking
VALL-E 使用了特殊的 Masking 策略来训练声学 Token 的生成。
*   它不仅有自回归（AR）部分，还有非自回归（NAR）部分。
*   在 NAR 阶段，它会随机 Mask 掉部分 Quantizer 层级的 Token，让模型去预测填空。
