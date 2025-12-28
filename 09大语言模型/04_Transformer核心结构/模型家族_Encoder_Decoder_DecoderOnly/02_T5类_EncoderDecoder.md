# T5 类：Encoder-Decoder 架构详解

Encoder-Decoder 架构（Seq2Seq）是转换型任务（Translation/Transduction Tasks）的标准选择。

## 1. 架构核心
*   **Encoder**: 双向注意力，负责提取输入序列的上下文特征。
*   **Decoder**: 单向注意力（Causal），负责自回归生成输出序列。
*   **Cross-Attention**: 连接 Encoder 和 Decoder 的桥梁。Decoder 的每一层都会查询 Encoder 的输出。

## 2. 代表模型：T5 (Text-to-Text Transfer Transformer)
*   **理念**: "Every task is a text-to-text problem"。
*   **预训练任务**: Span Corruption。将文本中的片段替换为 `<extra_id_X>`，要求 Decoder 还原这些片段。
*   **应用**: 机器翻译、摘要生成、问答。

## 3. 语音领域的 T5：Whisper & Audio-T5

### 3.1 Whisper
*   **定位**: 通用语音识别与翻译模型。
*   **输入**: Log-Mel Spectrogram (Encoder)。
*   **输出**: 文本 Token (Decoder)。
*   **特点**:
    *   **多任务**: 通过特殊的 Task Token (如 `<|transcribe|>`, `<|translate|>`) 控制 Decoder 的行为。
    *   **弱监督**: 不依赖人工标注的对齐信息，直接在大规模含噪数据上训练。

### 3.2 SpeechT5
*   **统一模态**: 试图统一语音和文本。
*   **结构**: 共享的 Encoder-Decoder 主干 + 模态特定的前处理/后处理网络（Pre-nets / Post-nets）。
*   **能力**: 可以做 ASR (语音->文本)，TTS (文本->语音)，VC (语音->语音转换)。

### 3.3 为什么 ASR 偏爱 Enc-Dec？
相比于 Decoder-Only，Enc-Dec 架构显式地分离了**声学建模**（Encoder）和**语言建模**（Decoder）。
*   Encoder 可以专注于处理充满噪声、变长的声学信号。
*   Decoder 可以专注于生成符合语法规则的文本。
*   Cross-Attention 承担了最困难的**对齐**工作。
