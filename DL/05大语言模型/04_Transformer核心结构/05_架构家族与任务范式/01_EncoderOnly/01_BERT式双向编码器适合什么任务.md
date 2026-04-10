---
tags:
  - LLM/架构
created: 2026-03-29
updated: 2026-03-29
---

# BERT 式双向编码器适合什么任务

## 问题定义

什么样的任务最适合用 BERT 一类双向编码器，而不是自回归生成模型？

## 直觉解释

当任务目标是“理解整段输入”而不是“续写输出”时，双向访问上下文更值钱。

## 形式化推导

BERT 类模型通过双向 self-attention 让每个 token 同时读取左右文，更适合构造整段输入的上下文化表示。

## 工程意义

分类、抽取、检索、rerank、表征学习都常受益于这种骨架。

## 常见误解

> [!warning] 常见误解
> - “有了 LLM，BERT 就没用了。” 不对，很多理解型服务仍更适合 encoder。

## 例子或反例

在 rerank 系统里，双向编码器常能以更低代价给出更稳定的匹配表示。

## 相关链接

- [[02_ViT如何把Transformer带进视觉|ViT 如何把 Transformer 带进视觉]]
- [[00_EncoderOnly_BERT_ViT|Encoder-Only]]
- [[00_DecoderOnly_GPT_LLaMA_Qwen|Decoder-Only]]
