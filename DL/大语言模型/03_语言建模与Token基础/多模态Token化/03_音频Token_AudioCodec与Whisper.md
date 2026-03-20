# 音频 Token：AudioCodec 与 Whisper

音频是一种连续的波形信号。要让 LLM 处理音频（如 GPT-4o 的语音模式），必须将其 Token 化。

## 1. 音频的特性

*   **高采样率**: 1秒钟的 16kHz 音频包含 16,000 个采样点。直接输入 Transformer 序列太长。
*   **连续性**: 声音是连续变化的波。

## 2. 两种 Token 化路线

### 2.1 语义 Token (Semantic Tokens) —— Whisper
OpenAI 的 Whisper 模型将音频映射为**文本 Token** 或**语义特征**。
*   **Log-Mel Spectrogram**: 先将波形转为对数梅尔频谱图（类似图像）。
*   **Encoder**: 使用 Transformer 编码器提取特征。
*   **目标**: 主要用于语音识别 (ASR) 和翻译。它丢弃了语调、情感等声学细节，只保留语义。

### 2.2 声学 Token (Acoustic Tokens) —— AudioCodec
为了生成语音（TTS）或进行语音对话，必须保留音色、语调、背景音等信息。这需要 **Neural Audio Codec**。

#### EnCodec (Meta) 与 SoundStream (Google)
这些模型类似于 VQ-VAE，但针对音频进行了优化。

1.  **Encoder**: 使用卷积神经网络将高频波形下采样。
2.  **Residual Vector Quantization (RVQ, 残差向量量化)**:
    *   音频信息量大，单一码本不够用。
    *   使用多个层级的码本。第一层量化主要轮廓，第二层量化残差（误差），第三层量化更细的残差...
    *   例如：使用 8 个码本，每个码本大小 1024。
3.  **Decoder**: 将量化后的 Token 序列还原为波形。

## 3. 语音大模型 (Audio LLM)

### 3.1 VALL-E (Microsoft)
*   **输入**: 文本 Token + 3秒参考音频的声学 Token。
*   **输出**: 目标音频的声学 Token。
*   **原理**: 把 TTS 变成了语言建模任务。

### 3.2 GPT-4o (Omni)
GPT-4o 是端到端 (End-to-End) 的多模态模型。
*   它可能直接输入/输出 Audio Token，跳过了“语音转文字 -> 文字处理 -> 文字转语音”的级联过程。
*   这使得它能感知情感、呼吸声，并以极低的延迟（毫秒级）进行打断和响应。

## 4. 总结

*   **Whisper**: 听懂你在说什么 (Content)。
*   **AudioCodec**: 听懂你是怎么说的 (Timbre, Emotion, Prosody)。
*   **Audio LLM**: 像处理文本一样处理这些 Token，实现语音的理解与生成。
