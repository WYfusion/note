---
tags:
  - LLM/架构
  - LLM/多模态
created: 2026-03-28
updated: 2026-03-29
---

# Encoder-Decoder：T5、BART 与 Whisper

> [!abstract] 摘要
> Encoder-Decoder 的价值在于把“理解源输入”和“生成目标输出”拆成两个角色。只要输入输出模态、长度分布或任务约束不同，这种拆分通常比单路自回归更自然。

## 这页先讲清什么

- Encoder-Decoder 为什么适合条件生成。
- T5、BART、Whisper 如何体现同一骨架的不同用途。
- 为什么这类结构至今仍然没有被完全淘汰。

## 关键结论

- Encoder-Decoder 天然适合“先读懂输入，再生成输出”的任务。
- Cross-attention 是这类架构的关键接口，而不是可有可无的附件。
- 当输入输出模态差异大时，这种结构尤其自然。

## 子页导航

- [[01_EncoderDecoder为什么天然适合条件生成|Encoder-Decoder 为什么天然适合条件生成]]
- [[02_T5_BART_Whisper的差异到底在哪里|T5、BART、Whisper 的差异到底在哪里]]

## 最短闭环解释

条件生成的关键不只是“继续写”，而是“先读懂某个来源，再基于它生成目标”。Encoder-Decoder 把这两步显式拆开：encoder 负责构建源表示，decoder 负责在自回归约束下读取这些表示并生成输出。

T5 用统一文本到文本接口贯彻这个思想，BART 把去噪重建也放进这个框架，Whisper 则把音频编码和文本生成自然接到一起。

所以，Encoder-Decoder 并不是“层更多的 Decoder-only”，而是一种专为条件生成设计的信息流骨架。

## 相关链接

- [[00_DecoderOnly_GPT_LLaMA_Qwen|Decoder-Only]]
- [[00_AttentionMask_Causal_Padding_CrossAttention|Attention Mask]]
- [[00_Transformer整体数据流_张量形状_EncoderDecoder|整体数据流与张量形状]]
