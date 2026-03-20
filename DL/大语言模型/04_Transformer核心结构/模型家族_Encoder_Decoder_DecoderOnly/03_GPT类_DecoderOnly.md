# GPT 类：Decoder-Only 架构详解

Decoder-Only 架构是生成式任务（Generative Tasks）的主宰，也是当前大语言模型（LLM）的主流范式。

## 1. 架构核心
*   **单向注意力 (Causal Attention)**: 严格限制每个 Token 只能看到它之前的 Token。
*   **自回归 (Auto-regressive)**: 逐个生成 Token，当前时刻的输出作为下一时刻的输入。

## 2. 代表模型：GPT (Generative Pre-trained Transformer)
*   **预训练任务**: Next Token Prediction (NTP)。
    $$ \max \sum_t \log P(x_t | x_{<t}) $$
*   **演进**: GPT-1 -> GPT-2 (Zero-shot) -> GPT-3 (In-context Learning) -> GPT-4。

## 3. 语音领域的 GPT：VALL-E & AudioLM

随着**神经音频编解码器 (Neural Audio Codec)** 的成熟，语音终于可以被离散化为 Token，从而直接套用 GPT 架构。

### 3.1 核心前提：Audio Tokenization
*   **EnCodec / SoundStream**: 将连续音频压缩为离散的 Codebook Indices。
*   **多层量化 (RVQ)**: 一个时间步对应多个 Token（例如 8 层）。这导致语音 Token 序列比文本长得多。

### 3.2 VALL-E
*   **任务**: TTS (Text-to-Speech)。
*   **Prompt**: 给定 3 秒参考音频（Acoustic Prompts）和目标文本（Phoneme Prompts）。
*   **生成**: 自回归地预测目标音频的第一层 Quantizer Token。
*   **涌现能力**: 展现出了上下文学习能力，能零样本克隆说话人的音色和情感。

### 3.3 Speech-LLaMA / Qwen-Audio
*   **多模态融合**: 将语音 Encoder 的输出直接映射到 LLM 的 Embedding 空间。
*   **Decoder-Only 的优势**: 统一了理解和生成。模型可以像处理文本一样处理语音，进行复杂的推理和对话。
