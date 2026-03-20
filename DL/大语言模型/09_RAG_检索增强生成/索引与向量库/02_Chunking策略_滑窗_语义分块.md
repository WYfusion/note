# Chunking 策略：滑窗与语义分块

## 1. 为什么需要 Chunking (切分)？
-   **上下文限制**：LLM 的 Context Window 有限，无法一次性输入整本书或整段长录音。
-   **检索精度**："大海捞针"（Needle in a Haystack）。切分越细，检索到的片段与 Query 的相关性通常越高，噪声越少。

## 2. 文本 Chunking 策略

### 2.1 固定大小切分 (Fixed-size Chunking)
最简单的方法，按字符数或 Token 数切分。
-   **参数**：`chunk_size` (如 512 tokens), `chunk_overlap` (如 50 tokens)。
-   **Overlap 的作用**：防止切分点切断了完整的句子或语义，保持上下文连贯性。

### 2.2 滑动窗口 (Sliding Window)
与固定大小类似，但窗口移动步长较小，生成更多重叠的片段。

### 2.3 语义分块 (Semantic Chunking)
基于内容的语义变化进行切分，而不是机械地按长度切。
-   **方法**：计算相邻句子的 Embedding 相似度。如果相似度低于阈值，说明话题发生了转换，在此处切分。
-   **优势**：保证每个 Chunk 内部的语义是完整的、独立的。

## 3. 语音 Chunking 策略 (Audio Chunking)
音频数据的切分比文本更复杂，因为音频是时间连续的信号。

### 3.1 基于时间的切分 (Time-based Chunking)
-   **方法**：将音频按固定时长（如 30秒）切分，保留一定的重叠（如 5秒）。
-   **适用**：背景音乐、环境音、无明显停顿的连续语音。
-   **缺点**：容易切断单词或乐句。

### 3.2 基于 VAD 的切分 (Silence-based Chunking)
利用 **VAD (Voice Activity Detection)** 技术检测静音段。
-   **方法**：在静音时长超过阈值（如 500ms）的地方进行切分。
-   **优势**：自然地按句子或短语边界切分，不会切断语音。
-   **工具**：Silero VAD, WebRTC VAD。

### 3.3 基于说话人的切分 (Speaker-based Chunking)
利用 **Speaker Diarization (说话人分离)** 技术。
-   **方法**：当说话人发生变化时（如从 A 换到 B）进行切分。
-   **适用**：会议记录、访谈、播客。
-   **优势**：每个 Chunk 只包含一个人的发言，便于后续的检索和归属分析。

### 3.4 基于 ASR 语义的切分 (ASR-Semantic Chunking)
先进行 ASR 转录，获得带时间戳的文本，然后在文本层面进行语义切分，最后映射回音频时间戳。
1.  **ASR**：Audio $\rightarrow$ Text + Timestamps。
2.  **Text Chunking**：使用 NLP 方法将 Text 切分为语义完整的段落。
3.  **Mapping**：根据 Text 的时间戳，反向截取对应的 Audio 片段。
-   **这是 Audio RAG 中最高级的切分策略**，兼顾了语义完整性和音频边界。

## 4. 最佳实践建议
-   **短 Query 场景**：使用较小的 Chunk Size（如 10-20秒音频），提高检索精准度。
-   **长摘要场景**：使用较大的 Chunk Size，或检索后扩展上下文（Small-to-Big 策略：检索小块，送入大块）。
-   **多模态对齐**：确保切分后的音频片段长度适中（如 < 30s），以适配 CLAP 等模型的输入限制。
