---
tags:
  - LLM/架构
  - 长上下文
created: 2026-03-29
updated: 2026-03-29
---

# Streaming Attention 解决了什么延迟问题

## 问题定义

为什么流式任务不能简单套用长窗口 attention，而需要 streaming attention 一类设计？

## 直觉解释

因为流式系统关心的不只是最终能不能看全历史，还关心每一步等待多久才能出结果。

## 形式化推导

Streaming attention 通过限制可见历史、保留有限状态或引入局部未来窗口，来控制实时延迟预算。

## 工程意义

语音、视频、实时代理等场景中，延迟约束往往比绝对质量更先决定结构。

## 常见误解

> [!warning] 常见误解
> - “streaming 只是更短窗口。” 不完整，它本质上还在做状态与延迟管理。

## 例子或反例

实时语音转写若每次都等完整句子再解码，体验上就已经失败了。

## 相关链接

- [[01_Chunking如何在训练与推理里工作|Chunking 如何在训练与推理里工作]]
- [[03_RingAttention为什么更像系统并行策略|RingAttention 为什么更像系统并行策略]]
- [[03_长上下文工程_Chunking_Streaming_RingAttention|长上下文工程]]
