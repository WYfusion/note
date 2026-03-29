---
tags:
  - LLM/架构
created: 2026-03-29
updated: 2026-03-29
---

# Cross-Attention Mask 如何约束条件信息流

## 问题定义

在 Encoder-Decoder 或多模态模型里，cross-attention mask 屏蔽的是什么？为什么它不是 encoder 侧或 decoder 侧 mask 的简单复制？

## 直觉解释

Cross-attention 读取的是“外部条件流”，所以它需要知道源侧哪些位置是真实可读的，哪些只是 pad、空白帧或被禁用区域。

## 形式化推导

Cross-attention 使用的 Query 来自 decoder，Key/Value 来自 encoder。其 mask 作用在 decoder-query 到 encoder-key 的可见性矩阵上，而不是 decoder 自注意力的下三角结构上。

## 工程意义

翻译、ASR、多模态问答中，源输入往往长度不规则、模态不同、预处理不同。cross-attention mask 是把这些输入有效范围显式对齐到 decoder 的关键接口。

## 常见误解

> [!warning] 常见误解
> - “cross-attention 只要沿用 padding mask 就够了。” 不完整。它还要匹配不同来源、不同长度的 query-key 交互。
> - “decoder 的 causal mask 已经保护了一切。” 不对。它管不了源输入区域是否合法。

## 例子或反例

Whisper 中音频帧经过 encoder 编码后长度与文本 token 完全不同。若 cross-attention 不正确屏蔽无效音频位置，decoder 会把静音填充当成有效条件。

## 相关链接

-[[02_EncoderDecoder_T5_BART_Whisper|Encoder-Decoder]]]
- [[01_PaddingMask为什么不是小细节|Padding Mask 为什么不是小细节]]
- [[02_AttentionMask_Causal_Padding_CrossAttention|Attention Mask]]

