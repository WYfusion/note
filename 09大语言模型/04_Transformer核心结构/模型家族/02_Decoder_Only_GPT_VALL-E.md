# Decoder-Only 架构：从 GPT 到 VALL-E

Decoder-Only 架构仅使用 Transformer 的 Decoder 部分（通常去掉了 Cross-Attention 层，只保留 Masked Self-Attention）。其核心特征是**自回归（Auto-regressive）**。

## 1. 文本领域的代表：GPT 系列

### 1.1 结构特点
*   **单向注意力 (Causal Mask)**: 位置 $i$ 只能看到位置 $j \le i$ 的信息。
*   **生成能力**: 天然适合文本生成任务。

### 1.2 预训练目标
*   **Next Token Prediction**: 给定 $x_1, ..., x_{t-1}$，预测 $x_t$。
    $$ P(x) = \prod_{t=1}^T P(x_t | x_{<t}) $$

### 1.3 适用场景
*   文本生成、对话系统、代码生成。
*   **In-context Learning**: 通过 Prompt 激发模型能力。

## 2. 语音领域的代表：VALL-E & AudioLM

随着神经音频编解码器（Neural Audio Codec）的发展，语音也可以被离散化为 Token，从而直接套用 GPT 架构。

### 2.1 核心前提：Audio Tokenization
使用 EnCodec 或 SoundStream 等模型，将连续音频压缩为离散的 Token 序列。
*   **残差量化 (RVQ)**: 一个时间步对应多个 Token（例如 8 层量化器，每帧有 8 个 Token）。

### 2.2 VALL-E (Neural Codec Language Model)
微软提出的 VALL-E 是典型的 Decoder-Only 语音模型。

*   **输入**: 文本 Token 序列 + 3秒提示音频的 Acoustic Token 序列。
*   **输出**: 目标音频的 Acoustic Token 序列。
*   **层级生成**:
    1.  **AR (Auto-regressive)**: 预测第一层量化器的 Token（粗粒度声学信息）。
    2.  **NAR (Non-Auto-regressive)**: 并行预测后续层量化器的 Token（细粒度细节）。

### 2.3 为什么 VALL-E 能做 Zero-Shot TTS？
因为它本质上是一个**条件语言模型**。
*   Prompt 音频提供了说话人的音色特征（Speaker Embedding）。
*   模型学会了在保持音色的同时，根据文本内容生成对应的声学 Token。

### 2.4 其他模型
*   **AudioLM (Google)**: 先生成语义 Token (w2v-BERT)，再生成声学 Token (SoundStream)。
*   **MusicGen (Meta)**: 用于音乐生成的 Decoder-Only 模型，使用了特殊的交错模式处理多层 Codebook。
