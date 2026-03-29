---
tags:
  - LLM/架构
created: 2026-03-29
updated: 2026-03-29
---

# Encoder-Decoder 为什么天然适合条件生成

## 问题定义

为什么翻译、摘要、ASR 等任务会天然偏好 Encoder-Decoder？

## 直觉解释

因为这类任务先有一段“条件输入”要被读懂，再有一段“目标输出”要被生成，双流骨架非常自然。

## 形式化推导

Encoder 负责构建条件表示，Decoder 在 causal 约束下通过 cross-attention 读取它们并生成目标序列。

## 工程意义

当输入输出长度、模态或采样率差异很大时，这种拆分尤其有价值。

## 常见误解

> [!warning] 常见误解
> - “Decoder-only 足够大就总能替代它。” 不绝对，结构清晰度仍有现实意义。

## 例子或反例

语音转文本里，先编码长音频再解码短文本通常比单路续写更自然。

## 相关链接

- [[02_T5_BART_Whisper的差异到底在哪里|T5、BART、Whisper 的差异到底在哪里]]
- [[02_EncoderDecoder_T5_BART_Whisper|Encoder-Decoder]]
- [[03_Transformer整体数据流_张量形状_EncoderDecoder|整体数据流与张量形状]]
