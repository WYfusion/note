---
tags:
  - LLM/多模态
aliases:
  - 多模态Tokenization
  - 多模态分词
  - 视觉Token
  - 音频Token
created: 2025-01-01
updated: 2026-03-28
---

# 多模态 Token 化：索引

> [!abstract] 模块概览
> 多模态 Token 化是将非文本数据（视觉、音频等）转化为离散 Token 序列的关键技术。本模块涵盖视觉 Patch 化、离散视觉表示（VQ-VAE/dVAE）、音频编解码（EnCodec/Whisper）等核心技术，实现文本、图像、音频的统一表示。

## 知识地图

```
多模态 Token 化
├── [[01_视觉Token_Patch与ViT|视觉 Token (ViT)]] — Patch 化基础
│   └── Image Patch → Token 序列
├── [[02_离散视觉Token_VQVAE与dVAE|离散视觉 Token]] — 量化表示
│   └── VQ-VAE/dVAE → 离散视觉 Token
└── [[03_音频Token_AudioCodec与Whisper|音频 Token (Codec)]] — 音频离散化 #语音/多模态
    └── EnCodec/Whisper → 音频 Token 序列
```

## 核心概念

### [[01_视觉Token_Patch与ViT|视觉 Token (ViT)]]
将图像分割为固定大小的 Patch，展平后映射为 Token 序列。ViT (Vision Transformer) 的核心创新，是视觉领域大模型的基础。

### [[02_离散视觉Token_VQVAE与dVAE|离散视觉 Token]]
使用 VQ-VAE 或 dVAE 将连续的视觉特征量化为离散的视觉 Token，实现图像的离散化表示，支持类似文本的处理方式。

### [[03_音频Token_AudioCodec与Whisper|音频 Token (Codec)]]  #语音/多模态
通过神经音频编解码器（如 EnCodec）或语音识别模型（如 Whisper）将连续的音频信号离散化为 Token 序列，支持端到端的语音大模型。

## 子模块导航

### [[01_视觉Token_Patch与ViT|视觉 Token (ViT)]] #LLM/多模态
> [!summary] 图像 Patch 化
> 将图像转换为离散 Token 的基础方法。
- **ViT 架构** — Transformer 如何处理图像
- **Patch 分割** — 图像到 Token 的转换
- **位置编码** — 空间信息保留

### [[02_离散视觉Token_VQVAE与dVAE|离散视觉 Token]] #LLM/多模态
> [!summary] 视觉量化表示
> 使用深度生成模型实现视觉数据的离散化。
- **VQ-VAE** — 向量量化自编码器
- **dVAE** — 分散的 VAE 变体
- **离散化挑战** — 信息保持与压缩比

### [[03_音频Token_AudioCodec与Whisper|音频 Token (Codec)]] #LLM/多模态
> [!summary] 音频离散化技术
> 音频信号的 Token 化方法。
- **神经编解码器** — EnCodec, DAC
- **语音识别模型** — Whisper 的音频 Token
- **多模态融合** — 文本-音频联合处理

## 前置与延伸

**前置知识**：
-[[01_语言建模目标_MLE与交叉熵困惑度|语言建模目标]]] — 离散化理论基础
- [[../01_BPE_WordPiece_Unigram|子词分词]] — 文本分词类比

**相关主题**：
-[[索引_Embedding与位置编码|Embedding 与位置编码]]] — 多模态嵌入
- [[../04_Transformer核心结构/索引_Transformer核心结构|Transformer 核心结构]] — 多模态 Transformer

**延伸阅读**：
- [[../../11_多模态与跨模态/索引_多模态与跨模态|多模态大模型]] — 完整多模态架构
- [[../../11_多模态与跨模态/语音语言模型SLM/索引_语音语言模型SLM|语音语言模型]] — 语音专用模型
