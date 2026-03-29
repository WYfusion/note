---
tags:
  - LLM/架构
aliases:
  - Encoder Only
created: 2026-03-28
updated: 2026-03-29
---

# Encoder-Only：BERT 与 ViT

> [!abstract] 摘要
> Encoder-only 的核心优势不是“结构更简单”，而是每个位置都能双向访问上下文，因此非常适合理解、检索、分类和表征学习。它不天然擅长自回归生成，但在很多非生成任务里仍然是高效主力。

## 这页先讲清什么

- Encoder-only 的信息流为什么偏向“理解”而不是“生成”。
- BERT 与 ViT 如何把同一骨架用在文本和视觉上。
- 为什么 Encoder-only 到今天依然没有过时。

## 关键结论

- 双向上下文是 Encoder-only 最大优势。
- BERT 说明它适合语言理解，ViT 说明它也能迁移到视觉表征。
- 只要任务目标是“把输入编码好”，Encoder-only 通常仍非常有竞争力。

## 子页导航

- [[01_BERT式双向编码器适合什么任务|BERT 式双向编码器适合什么任务]]
- [[02_ViT如何把Transformer带进视觉|ViT 如何把 Transformer 带进视觉]]

## 最短闭环解释

Encoder-only 模型没有自回归生成约束，因此每个 token 能同时看左右文。这使它特别适合理解和表征任务，因为模型的目标是把整段输入压成高质量表示，而不是一步一步往后写。

BERT 用 MLM 训练这一骨架，把它用在文本理解、检索和分类上；ViT 则说明，只要图像能被切成 patch token，Encoder-only 也能成为强视觉骨架。

所以 Encoder-only 不是旧时代产物，而是“当目标是编码而不是生成”时，依然非常自然的结构选择。

## 相关链接

- [[02_EncoderDecoder_T5_BART_Whisper|Encoder-Decoder]]
- [[03_DecoderOnly_GPT_LLaMA_Qwen|Decoder-Only]]
- [[05_跨模态Transformer_语音_视觉_视频的结构适配|跨模态 Transformer]]
