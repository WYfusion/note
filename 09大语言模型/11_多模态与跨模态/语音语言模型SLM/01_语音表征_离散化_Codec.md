# 语音表征、离散化与 Codec

要将大语言模型（LLM）的能力迁移到语音领域，核心挑战在于如何将连续的音频信号转化为 LLM 能够理解和生成的离散 Token。本节详细介绍语音表征学习、离散化策略以及主流的 Neural Audio Codec。

## 1. 语音表征 (Audio Representation)

### 1.1 传统声学特征
*   **波形 (Waveform)**: 原始的一维时间序列，维度极高（16kHz 采样率下，1秒包含 16000 个点）。
*   **频谱图 (Spectrogram)**: 通过 STFT 变换得到的时频图。
*   **梅尔频谱 (Mel-Spectrogram)**: 符合人耳听觉特性的频域压缩特征，是大多数语音模型的标准输入。

### 1.2 自监督学习表征 (SSL)
类似于 NLP 中的 BERT，语音领域通过掩码预测任务学习上下文表征。
*   **Wav2Vec 2.0**: 通过对比学习，将量化的潜在表示与上下文表示拉近。
*   **HuBERT**: 预测被 Mask 掉的帧的聚类中心（K-Means Cluster ID）。
*   **特点**: 包含丰富的语义和音素信息，但丢失了部分说话人音色和背景噪声细节。

---

## 2. 离散化 (Discretization)

LLM 本质上是一个离散符号的概率预测机。为了让 LLM 生成音频，必须将连续音频离散化。

### 2.1 语义 Token (Semantic Tokens)
*   **来源**: 对 HuBERT 或 w2v-BERT 的输出进行 K-Means 聚类。
*   **作用**: 捕捉语言内容（说了什么），忽略音色和噪声。
*   **应用**: AudioLM 的第一阶段生成。

### 2.2 声学 Token (Acoustic Tokens)
*   **来源**: Neural Audio Codec 的量化器输出。
*   **作用**: 重构高质量波形，包含音色、情感、录音环境。
*   **应用**: AudioLM 的第二阶段生成，VALL-E。

---

## 3. Neural Audio Codec

Neural Codec 是现代 Speech LLM 的基石，它充当了 Tokenizer 和 Detokenizer 的角色。

### 3.1 核心架构：VQ-VAE / VQ-GAN
$$
\text{Audio} \xrightarrow{Encoder} \text{Latent} \xrightarrow{Quantizer} \text{Indices (Tokens)} \xrightarrow{Decoder} \text{Reconstructed Audio}
$$

### 3.2 残差矢量量化 (Residual Vector Quantization, RVQ)
为了在高压缩率下保持高保真度，SoundStream 和 EnCodec 引入了 RVQ。
*   **原理**: 使用多个量化器（Codebook）级联。
    *   第 1 个量化器逼近原始向量，得到残差 $R_1$。
    *   第 2 个量化器逼近 $R_1$，得到残差 $R_2$。
    *   ...
    *   第 $N$ 个量化器逼近 $R_{N-1}$。
*   **结果**: 每个时间步对应 $N$ 个 Token（例如 8 层 RVQ）。
*   **分层建模**: 粗粒度 Token 决定内容，细粒度 Token 决定音质细节。

### 3.3 主流 Codec 模型
1.  **SoundStream (Google)**: 首个通用的 Neural Audio Codec，支持语音和音乐。
2.  **EnCodec (Meta)**: 引入了 Transformer 和 GAN Loss，在极低码率下（如 3kbps）仍能保持高音质。
3.  **DAC (Descript Audio Codec)**: 目前 SOTA 的高保真 Codec。

---

## 4. Speech LLM 典型架构

### 4.1 AudioLM (Google)
*   **两阶段生成**:
    1.  **Semantic Generation**: Text -> Semantic Tokens (HuBERT)。
    2.  **Acoustic Generation**: Semantic Tokens -> Acoustic Tokens (SoundStream)。
*   **特点**: 实现了语言内容与声学细节的解耦与重组。

### 4.2 VALL-E (Microsoft)
*   **任务**: Zero-shot TTS。
*   **输入**: Text Tokens + 3秒参考音频的 Acoustic Tokens。
*   **输出**: 目标音频的 Acoustic Tokens (EnCodec)。
*   **核心**: 将 TTS 视为条件语言建模任务 (Conditional Language Modeling)。

### 4.3 Qwen-Audio (Alibaba)
*   **架构**: Whisper-large-v3 (Encoder) -> Average Pooling -> MLP -> Qwen-7B。
*   **特点**: 强大的通用音频理解能力（ASR, SQA, 音乐分析），但不直接生成音频。
