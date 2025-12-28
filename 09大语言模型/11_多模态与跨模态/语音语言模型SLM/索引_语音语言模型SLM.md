# 语音语言模型 (SLM)

本章节深入探讨 Speech LLM 的核心技术栈，从底层的声学特征到顶层的模型架构。

## 目录

### [01_语音表征_离散化_Codec.md](./01_语音表征_离散化_Codec.md)
- **语音表征**：Spectrogram, Wav2Vec 2.0, HuBERT。
- **离散化 (Tokenization)**：
  - **Semantic Tokens**: K-Means on SSL features (Content)。
  - **Acoustic Tokens**: Neural Codec Quantization (Timbre, Detail)。
- **Neural Audio Codec**：
  - **SoundStream**: Residual Vector Quantization (RVQ)。
  - **EnCodec**: High-fidelity at low bitrate。
- **典型架构**：
  - **AudioLM**: Semantic -> Acoustic 两阶段生成。
  - **VALL-E**: Zero-shot TTS via Conditional Language Modeling。
  - **Qwen-Audio**: Universal Audio Understanding Adapter。

