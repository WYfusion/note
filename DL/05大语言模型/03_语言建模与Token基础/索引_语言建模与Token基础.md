---
tags:
  - LLM/语言建模
aliases:
  - 语言建模与Token基础
  - Tokenization基础
created: 2025-01-01
updated: 2026-03-28
---

# 语言建模与 Token 基础：索引

> [!abstract] 模块概览
> 本模块是理解大语言模型的基础，涵盖语言建模的数学目标、分词算法（Tokenization）、向量嵌入（Embedding）以及多模态 Token 化的完整知识体系。从「语言是什么」到「文本如何变成 Token」，再到「Token 如何变成向量」和「多模态数据如何统一表示」。

## 知识地图

```
语言建模与Token基础
├── [[01_语言建模目标_MLE与交叉熵困惑度|语言建模目标]] — MLE, 交叉熵, 困惑度
├── [[02_自回归AR_自编码AE_序列到序列Seq2Seq对比|序列模型对比]] — AR vs AE vs Seq2Seq
├── Tokenizer 与分词
│   ├── [[Tokenizer与分词/01_BPE_WordPiece_Unigram|BPE / WordPiece / Unigram]] — 子词分词算法
│   ├── [[Tokenizer与分词/02_特殊Token_对齐与模板的影响|特殊 Token 与对齐]] — 特殊 Token 作用
│   ├── [[Tokenizer与分词/03_OOV_长度膨胀_多语言与代码|OOV 与长度膨胀]] — 分词挑战
│   ├── [[Tokenizer与分词/04_Tokenizer的训练与构建流程|Tokenizer 训练流程]] — 训练与构建
│   └── [[Tokenizer与分词/05_中文分词与多语言挑战|中文分词挑战]] — 中文特性
├── 多模态 Token 化
│   ├── [[多模态Token化/01_视觉Token_Patch与ViT|视觉 Token (ViT)]] — Patch 化
│   ├── [[多模态Token化/02_离散视觉Token_VQVAE与dVAE|离散视觉 Token]] — VQ-VAE/dVAE
│   └── [[多模态Token化/03_音频Token_AudioCodec与Whisper|音频 Token (Codec)]] — EnCodec/Whisper #语音/多模态
└── Embedding 与位置编码
    ├── [[Embedding与位置编码/1向量嵌入类型/向量嵌入类型|向量嵌入类型]] — 文本/代码/图片/音频/视频/多模态嵌入
    ├── [[Embedding与位置编码/2向量嵌入技术核心|向量嵌入技术核心]] — 训练范式, 距离度量, 评测体系
    ├── [[Embedding与位置编码/3向量数据库与检索引擎|向量数据库与检索引擎]] — ANN 索引, 向量数据库
    ├── [[Embedding与位置编码/4_方案演进脉络|方案演进脉络]] — 技术演进
    ├── [[Embedding与位置编码/5_主流组合与选型指南|选型指南]] — 实战选型
    └── [[Embedding与位置编码/6_综合使用流程|综合使用流程]] — 完整流程
```

## 核心概念

### [[01_语言建模目标_MLE与交叉熵困惑度|语言建模目标]]
简要描述语言建模的数学目标、最大似然估计（MLE）、交叉熵损失函数以及困惑度（Perplexity）等关键评估指标。

### [[02_自回归AR_自编码AE_序列到序列Seq2Seq对比|序列模型对比]]
对比自回归（AR）、自编码（AE）和序列到序列（Seq2Seq）三种范式，帮助理解不同模型架构的适用场景。

###[[01_BPE_WordPiece_Unigram|BPE / WordPiece / Unigram]]]
详细介绍三大子词分词算法（BPE、WordPiece、Unigram）的原理、训练过程和优缺点对比。

###[[03_音频Token_AudioCodec与Whisper|音频 Token (Codec)]]]  #语音/Codec
详解语音数据如何通过 Neural Audio Codec（如 EnCodec）和 Whisper 实现离散化，支持端到端语音大模型。

### [[Embedding与位置编码/1向量嵌入类型/2_Dense_Bi-Encoder_句向量|Dense Bi-Encoder 句向量]]
介绍 SBERT、E5、BGE 等 Dense Bi-Encoder 向量模型的工作原理和应用场景。

### [[Embedding与位置编码/3向量数据库与检索引擎|向量数据库与检索引擎]]
讲解 ANN（近似最近邻）搜索算法、专用向量数据库（Milvus、Weaviate）以及搜索引擎集成。

## 子模块导航

###[[索引_Tokenizer与分词|Tokenizer 与分词]]]
> [!summary] 子词分词模块
> 从原始文本到离散 Token 序列的完整流程。
-[[01_BPE_WordPiece_Unigram|BPE / WordPiece / Unigram]]] — 三大算法详解
-[[02_特殊Token_对齐与模板的影响|特殊 Token 与对齐]]] — BOS/EOS/PAD 作用
-[[03_OOV_长度膨胀_多语言与代码|OOV 与长度膨胀]]] — 常见问题与解决方案
-[[04_Tokenizer的训练与构建流程|Tokenizer 训练]]] — 如何训练新 Tokenizer
-[[05_中文分词与多语言挑战|中文分词挑战]]] — 中文语言特殊性

###[[索引_多模态Token化|多模态 Token 化]]] #LLM/多模态
> [!summary] 多模态数据离散化
> 将视觉、音频等连续数据转化为 LLM 可处理的离散 Token。
-[[01_视觉Token_Patch与ViT|视觉 Token (ViT)]]] — 图像 Patch 化
-[[02_离散视觉Token_VQVAE与dVAE|离散视觉 Token]]] — VQ-VAE/dVAE 量化
-[[03_音频Token_AudioCodec与Whisper|音频 Token (Codec)]]] — EnCodec/Whisper #语音/多模态

###[[索引_Embedding与位置编码|Embedding 与位置编码]]]
> [!summary] 向量嵌入与检索
> 从静态词向量到现代多模态嵌入的完整技术栈。
- [[Embedding与位置编码/1向量嵌入类型/向量嵌入类型|向量嵌入类型]] — 全部嵌入类型索引
- [[Embedding与位置编码/2向量嵌入技术核心|向量嵌入技术核心]] — 核心技术与理论
- [[Embedding与位置编码/3向量数据库与检索引擎|向量数据库与检索引擎]] — 检索引擎专题
- [[Embedding与位置编码/4_方案演进脉络|方案演进脉络]] — 技术发展脉络
- [[Embedding与位置编码/5_主流组合与选型指南|选型指南]] — 实战选型指南
- [[Embedding与位置编码/6_综合使用流程|综合使用流程]] — 端到端流程

## 前置与延伸

**前置知识**：
-[[01_从统计语言模型到神经LM|语言模型基础]]] — 统计语言模型到神经 LM 的演进

**延伸学习**：
-[[索引_Transformer核心结构|Transformer 核心结构]]] — Transformer 架构如何使用 Token 和 Embedding
- [[../06_指令微调与参数高效微调PEFT/2向量模型微调|向量模型微调]] — 如何微调 Embedding 模型
- [[../11_多模态与跨模态/语音语言模型SLM/索引_语音语言模型SLM|语音语言模型]] — 语音大模型中的 Token 处理

