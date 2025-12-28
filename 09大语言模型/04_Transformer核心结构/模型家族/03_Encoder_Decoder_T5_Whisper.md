# Encoder-Decoder 架构：从 T5 到 Whisper

Encoder-Decoder 架构完整保留了 Transformer 的两部分。Encoder 负责理解输入，Decoder 负责生成输出。两者通过 **Cross-Attention** 连接。

## 1. 文本领域的代表：T5 & BART

### 1.1 结构特点
*   **Encoder**: 双向注意力，提取输入特征。
*   **Decoder**: 单向注意力（Causal），自回归生成输出。
*   **Cross-Attention**: Decoder 的每一层都会查询 Encoder 的输出。

### 1.2 预训练目标
*   **T5 (Text-to-Text Transfer Transformer)**: "Span Corruption"。将输入中的一段文本替换为 `<extra_id_0>`，要求 Decoder 生成被替换的内容。
*   **BART**: 类似于去噪自编码器。输入被破坏的文本（打乱顺序、删除词），Decoder 还原原始文本。

### 1.3 适用场景
*   机器翻译（Translation）、文本摘要（Summarization）。

## 2. 语音领域的代表：Whisper

OpenAI 的 Whisper 是目前最流行的 Encoder-Decoder 语音模型。

### 2.1 结构细节
*   **Encoder**:
    *   输入：80 维 Log-Mel Spectrogram。
    *   预处理：2 个卷积层（步长为 2），将时间分辨率降低 4 倍。
    *   主体：标准的 Transformer Encoder（Pre-Norm）。
    *   位置编码：Sinusoidal。
*   **Decoder**:
    *   标准的 Transformer Decoder。
    *   输入：特殊的 Task Tokens（如 `<|startoftranscript|>`, `<|en|>`, `<|transcribe|>`, `<|notimestamps|>`）。

### 2.2 训练任务
Whisper 没有使用复杂的自监督预训练，而是直接在大规模弱监督数据（68万小时）上进行**多任务有监督训练**。
*   **ASR**: 语音 -> 文本。
*   **Speech Translation**: 语音（任意语言） -> 英文文本。
*   **VAD**: 语音 -> 时间戳。

### 2.3 为什么 Whisper 选择 Enc-Dec？
*   **模态解耦**: Encoder 专门处理声学特征（连续、噪声大、变长），Decoder 专门处理文本特征（离散、语义丰富）。
*   **Cross-Attention 的作用**: 实现了从“声学空间”到“语义空间”的对齐。Decoder 在生成每个词时，可以动态地“看”Encoder 输出的对应时间片段。

### 2.4 与 Decoder-Only (如 Speech-LLaMA) 的对比
*   **Whisper**: 也就是所谓的“端到端 ASR”。适合纯转录任务。
*   **Speech-LLaMA**: 将 Encoder 输出作为 Prompt 喂给 LLM。适合需要复杂推理、对话的语音交互任务。
