---
tags:
  - LLM/语言建模
aliases:
  - Tokenizer与分词
  - 分词算法
created: 2025-01-01
updated: 2026-03-28
---

# Tokenizer 与分词：索引

> [!abstract] 模块概览
> Tokenizer 是将人类可读文本转换为 LLM 可处理的离散 Token 序列的「翻译器」。本模块涵盖三大子词分词算法（BPE、WordPiece、Unigram）、特殊 Token 的作用、多语言挑战以及中文分词的特殊性。

## 知识地图

```
Tokenizer 与分词
├── [[01_BPE_WordPiece_Unigram|BPE / WordPiece / Unigram]] — 三大算法详解
├── [[02_特殊Token_对齐与模板的影响|特殊 Token 与对齐]] — BOS/EOS/PAD/SPECIAL
├── [[03_OOV_长度膨胀_多语言与代码|OOV 与长度膨胀]] — 常见问题与解决方案
├── [[04_Tokenizer的训练与构建流程|Tokenizer 训练与构建]] — 训练新 Tokenizer 流程
└── [[05_中文分词与多语言挑战|中文分词挑战]] — 中文语言特殊性
```

## 核心概念

### [[01_BPE_WordPiece_Unigram|BPE / WordPiece / Unigram]]
详细介绍三大子词分词算法的核心思想、训练过程和优缺点对比。

### [[02_特殊Token_对齐与模板的影响|特殊 Token 与对齐]]
解析 BOS（序列开始）、EOS（序列结束）、PAD（填充）、MASK（掩码）等特殊 Token 的作用，以及 ChatML 等 Chat 模板的 Token 对齐规则。

### [[03_OOV_长度膨胀_多语言与代码|OOV 与长度膨胀]]
讨论未登录词（Out-of-Vocabulary, OOV）问题、Token 长度膨胀问题，以及多语言和代码场景下的分词挑战。

### [[04_Tokenizer的训练与构建流程|Tokenizer 训练与构建]]
从零开始训练自定义 Tokenizer 的完整流程，包括数据准备、算法选择、训练和验证。

### [[05_中文分词与多语言挑战|中文分词挑战]]
中文分词的特殊性（无空格、多粒度），以及多语言模型支持的考虑。

## 子模块导航

### [[01_BPE_WordPiece_Unigram|BPE / WordPiece / Unigram]]
> [!summary] 子词分词算法
> 三大算法的原理、训练、推理对比。
- **BPE** — 频繁相邻对合并
- **WordPiece** — 贪心最长匹配 + ## 前缀
- **Unigram** — 概率化子词模型

### [[02_特殊Token_对齐与模板的影响|特殊 Token 与对齐]]
> [!summary] 特殊 Token 作用
> 特殊 Token 的分类、使用场景和对齐技巧。
- **特殊 Token** — BOS/EOS/PAD/MASK/UNK 分类
- **Chat 模板** — ChatML/Llama3 对齐
- **对齐策略** — 对齐失败问题

### [[03_OOV_长度膨胀_多语言与代码|OOV 与长度膨胀]]
> [!summary] 分词挑战
> 常见问题与解决方案。
- **OOV 问题** — 未登录词处理
- **长度膨胀** — Token 效率优化
- **代码分词** — 代码场景特殊性
- **多语言支持** — 多语言 Tokenizer

### [[04_Tokenizer的训练与构建流程|Tokenizer 训练与构建]]
> [!summary] 训练流程
> 从零开始训练 Tokenizer 的实践指南。
- **训练流程** — 数据准备 → 训练 → 验证
- **工具选择** — SentencePiece, Tokenizers
- **质量评估** — 评估指标

### [[05_中文分词与多语言挑战|中文分词挑战]]
> [!summary] 中文分词
> 中文分词的特殊挑战。
- **中文特性** — 无空格、字符-词-词三层
- **分词粒度** — 字/词/子词
- **多语言模型** — Qwen, Llama 多语言支持

## 前置与延伸

**前置知识**：
- [[01_语言建模目标_MLE与交叉熵困惑度|语言建模目标]] — 语言建模数学基础
- 字符编码（UTF-8, Unicode）

**相关主题**：
- [[../04_Transformer核心结构/索引_Transformer核心结构|Transformer 核心结构]] — Transformer 如何使用 Token
- [[../06_指令微调与参数高效微调PEFT/7训练范式与多阶段流程|训练范式与多阶段流程]] — 分词在训练中的作用

**延伸阅读**：
-[[01_视觉Token_Patch与ViT|视觉 Token (ViT)]]] — 视觉 Patch 化对比
-[[03_音频Token_AudioCodec与Whisper|音频 Token (Codec)]]] — 语音分词对比
- [[../Embedding与位置编码/1向量嵌入类型/2_Dense_Bi-Encoder_句向量|Dense Bi-Encoder 句向量]] — Embedding 与 Token 的关系

