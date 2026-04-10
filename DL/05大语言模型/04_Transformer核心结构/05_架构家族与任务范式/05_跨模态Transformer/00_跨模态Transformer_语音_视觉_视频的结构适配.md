---
tags:
  - LLM/架构
  - LLM/多模态
created: 2026-03-28
updated: 2026-03-29
---

# 跨模态 Transformer：语音、视觉、视频的结构适配

> [!abstract] 摘要
> 跨模态 Transformer 不是简单把文本模型搬去处理图片或音频。真正要改的是：输入如何 token 化、位置如何定义、模态之间怎么融合，以及高密度连续信号带来的长度问题如何控制。

## 这页先讲清什么

- 跨模态任务里哪些部分能沿用文本 Transformer，哪些必须改。
- 视觉、语音、视频为什么需要不同 token 化和位置设计。
- 为什么多模态系统的难点常常是长度和对齐，而不只是模型主干。

## 关键结论

- 跨模态的第一步不是换 attention，而是先把不同模态变成可处理的 token。
- 位置编码在视觉、语音、视频里不可能完全照搬文本。
- 多模态融合既可能靠拼接，也可能靠 cross-attention，取决于任务结构。

## 子页导航

- [[01_模态Token化如何接入Transformer|模态 Token 化如何接入 Transformer]]
- [[02_语音_视觉_视频为什么需要不同结构适配|语音、视觉、视频为什么需要不同结构适配]]

## 最短闭环解释

文本是离散 token，图像是二维 patch，音频和视频是高密度连续序列。只要输入形态变了，token 化、位置表示和长度控制就必须重写。

有些多模态模型会把模态 token 先投到统一空间后直接拼给 Decoder-only 模型，有些则保留单独 encoder，再用 cross-attention 融合。哪种更合适，取决于模态对齐方式、长度预算和任务目标。

所以，跨模态 Transformer 的难点从来不只是“能不能用 Transformer”，而是“怎样把不同模态合理地变成 Transformer 愿意处理的输入”。

## 相关链接

- [[00_Transformer整体数据流_张量形状_EncoderDecoder|整体数据流与张量形状]]
- [[00_长上下文工程_Chunking_Streaming_RingAttention|长上下文工程]]
- [[03_RoPE_ALiBi_为什么主流模型偏向这两类方案|RoPE 与 ALiBi]]
