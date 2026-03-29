---
tags:
  - LLM/多模态
  - LLM/语音
  - #语音/ASR
  - #语音/TTS
  - #语音/Codec
aliases:
  - 音频Token
  - 语音Token
  - AudioCodec
  - Whisper
created: 2025-01-01
updated: 2026-03-28
---

# 音频 Token：AudioCodec 与 Whisper

> [!abstract] 摘要
> 音频 Token 是将连续的音频波形离散化为序列的关键技术。语义 Token（Whisper）专注于内容理解，声学 Token（AudioCodec）保留声音细节，共同构建了语音大模型的基础。这一技术在 GPT-4o Audio、VALL-E 等语音大模型中发挥着核心作用。

## 0. 统一概念：为什么需要音频 Token？

音频是一种连续的波形信号。要让 LLM 处理音频（如 GPT-4o 的语音模式），必须将其 Token 化。

## 1. 音频的挑战特性

> [!important] 音频 Token 化的独特挑战

### 1.1 高采样率 #语音/ASR

*   **高采样率**: 1秒钟的 16kHz 音频包含 16,000 个采样点。直接输入 Transformer 序列太长。
*   **计算爆炸**：1分钟音频 = 960,000 个采样点 = 远超 Transformer 上下文窗口

> [!warning] 序列长度问题
> 假设 Transformer 最大长度为 4096：
> - 16kHz 音频：256ms = 4,096 个采样点
> - 采样率必须降到 1kHz 才能放入窗口
> - 但 1kHz 会丢失高频信息（>5kHz）

### 1.2 时空连续性 #语音/TTS

> [!important] 声音的连续性
> 声音是连续变化的波，具有：
> - **时间连续性**：相邻采样点高度相关
> - **频率连续性**：频谱随时间平滑变化
> - **多尺度信息**：同时存在语音音素（百ms）和韵律特征（秒级）

**音频 vs 文本的关键差异**：
| 特性 | 文本 | 音频 |
|------|------|------|
| **离散性** | 字符离散 | 波形连续 |
| **时间分辨率** | 字符级 | 采样点级 |
| **信息密度** | 低 | 高 |
| **冗余度** | 低 | 高 |

## 2. 两种 Token 化路线  #LLM/语音

> [!important] 语义 vs 声学：两种不同的路径

### 2.1 语义 Token (Semantic Tokens) —— Whisper #语音/ASR

> [!tip] Whisper 的设计哲学
> OpenAI 的 Whisper 模型将音频映射为**文本 Token** 或**语义特征**。

**处理流程**：
```
音频波形 → 预处理 → Log-Mel Spectrogram → Transformer Encoder → 文本 Token
    ↓
  [预处理]
  - 重采样到 16kHz
  - 去除静音段
  - 添加特殊标记（如 <|notimestamps|>）
```

> [!important] 关键技术细节
> **Log-Mel Spectrogram**：
> 1. **短时傅里叶变换 (STFT)**：将时域信号转为时频域
>    $$X_{\text{STFT}}[m, \omega] = \sum_{t=0}^{N-1} x[t] \cdot w[t-m] \cdot e^{-j2\pi\omega t/N}$$
> 2. **Mel 滤波器组**：将线性频率转换为 Mel 尺度
>    $$M(f) = 2595 \cdot \log_{10}(1 + f/700)$$
> 3. **对数压缩**：增强动态范围
>    $$S = \log_{10}(1 + \text{Mel}(X_{\text{STFT}}))$$

> [!note] Whisper 的局限性
> **目标**：主要用于语音识别 (ASR) 和翻译。它丢弃了语调、情感等声学细节，只保留语义。

### 2.2 声学 Token (Acoustic Tokens) —— AudioCodec #语音/TTS

> [!important] 声学 Token 的必要性
> 为了生成语音（TTS）或进行语音对话，必须保留音色、语调、背景音等信息。这需要 **Neural Audio Codec**。

> [!example] 声学信息的重要性
> **丢失的信息**：
> - 音色特征（说话人身份）
> - 语调情感（高兴、悲伤）
> - 韵律特征（重音、语速）
> - 环境音（背景噪音、回声）

#### EnCodec (Meta) 与 SoundStream (Google) #语音/Codec

> [!tip] 神经音频编解码器
> 这些模型类似于 VQ-VAE，但针对音频进行了优化。

**Residual Vector Quantization (RVQ, 残差向量量化)** ：
```
原始音频 x → Encoder → 特征 z
↓ ↓ ↓
第一层码本：量化主要轮廓 z₁ = Q₁(z)
第二层码本：量化残差 z₂ = Q₂(z - z₁)
第三层码本：量化更细残差 z₃ = Q₃(z - z₁ - z₂)
...
最终：Token序列 [i₁, i₂, i₃, ..., iₙ]
```

**RVQ 的优势**：
- **信息分层**：主要信息 → 细节信息
- **压缩率高**：使用多个小码本比一个大码本更高效
- **渐进解码**：可以只解码前几层得到基本内容

> [!example] EnCodec 的参数配置
> ```python
> # EnCodec 配置示例
> encoder_stride = 320  # 10ms @ 32kHz
> latent_dim = 2048     # 每个帧的维度
> n_codebooks = 8       # 8个RVQ码本
> codebook_size = 1024   # 每个码本大小
> ```

---

## 3. 语音大模型 (Audio LLM) #LLM/语音

### 3.1 VALL-E (Microsoft) #语音/TTS

> [!important] 零样本语音合成的突破

**架构**：
```
文本提示 + 参考音频 → GPT-2 → 声学 Token → EnCodec → 语音输出
    ↑
  [文本编码]
  [音频编码]
  [音色提取]
```

**关键创新**：
- **零样本学习**：只需 3 秒参考音频
- **音色克隆**：保留说话人音色特征
- **情感保留**：参考音频的情感被保留
- **多语言支持**：支持多种语言合成

### 3.2 GPT-4o (Omni) #语音/ASR

> [!important] 端到端多模态的革命
> GPT-4o 是端到端 (End-to-End) 的多模态模型。

**音频处理能力**：
- **实时双向交流**：232ms 端到端延迟
- **多模态理解**：同时理解语音、文字、图像
- **情感识别**：识别语音中的情绪、语调
- **打断能力**：毫秒级响应，支持自然对话

> [!note] GPT-4o Audio 的技术特点
> - **直接 Audio Token**：可能直接输入/输出 Audio Token，跳过了"语音转文字 -> 文字处理 -> 文字转语音"的级联过程
> - **多尺度处理**：同时处理音素、词、句子级别的音频特征
- **长上下文**：支持 1 小时以上的音频上下文

### 3.3 主流语音大模型对比

| 模型 | 类型 | Token 化方式 | 主要能力 | 特点 |
|------|------|--------------|----------|------|
| **Whisper** | Encoder-only | 语义 Token | 语音识别/翻译 | 高精度 ASR |
| **VALL-E** | Decoder-only | 声学 Token | 语音合成 | 零样本 TTS |
| **GPT-4o** | Encoder-Decoder | 混合 Token | 多模态交互 | 端到端理解 |
| **Qwen-Audio** | Encoder-Decoder | 混合 Token | 中文语音能力 | 中文优化 |

## 3. 语音大模型 (Audio LLM)

### 3.1 VALL-E (Microsoft)
*   **输入**: 文本 Token + 3秒参考音频的声学 Token。
*   **输出**: 目标音频的声学 Token。
*   **原理**: 把 TTS 变成了语言建模任务。

### 3.2 GPT-4o (Omni)
GPT-4o 是端到端 (End-to-End) 的多模态模型。
*   它可能直接输入/输出 Audio Token，跳过了“语音转文字 -> 文字处理 -> 文字转语音”的级联过程。
*   这使得它能感知情感、呼吸声，并以极低的延迟（毫秒级）进行打断和响应。

## 4. 音频 Token 技术选型指南

### 4.1 三种 Token 的定位

> [!important] 不同 Token 的适用场景

| Token 类型 | 听懂什么 | 用途 | 模型示例 |
|-----------|----------|------|----------|
| **语义 Token** | 你在说什么 (Content) | ASR、翻译、内容理解 | Whisper |
| **声学 Token** | 你是怎么说的 (Timbre, Emotion, Prosody) | TTS、语音克隆、情感交互 | VALL-E, AudioLM |
| **混合 Token** | 既懂内容又懂表达 | 多模态对话、语音助手 | GPT-4o, Qwen-Audio |

### 4.2 选择策略

> [!tip] 如何选择音频 Token 方案

**选择语义 Token 的场景**：
- ✅ 只需要理解内容
- ✅ 任务：语音转文字、翻译、字幕
- ✅ 计算资源有限
- ❌ 需要保留音色情感

**选择声学 Token 的场景**：
- ✅ 需要生成语音
- ✅ 任务：语音克隆、TTS、语音编辑
- ✅ 需要保留音色情感
- ❌ 不需要理解语义内容

**选择混合 Token 的场景**：
- ✅ 需要端到端语音交互
- ✅ 任务：语音助手、多模态对话
- ✅ 需要理解+生成一体化
- ❌ 计算资源要求高

---

## 5. 最新进展：多模态音频理解

### 5.1 Whisper V3 的改进

> [!note] 更强大的语音理解
> - **更长上下文**：支持 30 秒音频处理
> - **多语言支持**：99 种语言
> - **时间戳精度**：毫秒级
- **语音活动检测**：更精确的 VAD

### 5.2 音频-文本联合嵌入

> [!example] CLAP 风格的音频理解
> ```python
> # 音频-文本对齐
> audio_embed = audio_encoder(waveform)
> text_embed = text_encoder(prompt)
> similarity = cosine_similarity(audio_embed, text_embed)
> ```

### 5.3 长音频处理技术

> [!warning> 长音频的挑战
> - **上下文窗口限制**：标准 Transformer 上下文有限
> - **注意力计算复杂度**：O(n²) 问题
> - **信息压缩**：如何在压缩中保留关键信息

**解决方案**：
- **分层注意力**：局部注意力 + 全局注意力
- **滑动窗口**：处理超出上下文的音频
- **关键帧提取**：保留语义关键点

---

## 相关链接

**所属模块**：[[索引_多模态Token化]]

**前置知识**：
- [[../../01_语言建模目标_MLE与交叉熵困惑度|语言建模目标]] — 自回归建模基础
- [[../../Tokenizer与分词/01_BPE_WordPiece_Unigram|子词分词]] — 离散化原理

**相关主题**：
- [[../01_视觉Token_Patch与ViT|视觉 Token (ViT)]] — 视觉 Token 化对比
- [[../02_离散视觉Token_VQVAE与dVAE|离散视觉 Token]] — VQ-VAE 原理
- [[../../04_Transformer核心结构/模型家族/03_Encoder_Decoder_T5_Whisper|Whisper 架构]] — 详细架构解析

**延伸阅读**：
- [[../../11_多模态与跨模态/索引_多模态与跨模态|多模态大模型]] — 完整多模态架构
- [[../../11_多模态与跨模态/语音语言模型SLM/索引_语音语言模型SLM|语音语言模型]] — 专用语音 LLM 知识

