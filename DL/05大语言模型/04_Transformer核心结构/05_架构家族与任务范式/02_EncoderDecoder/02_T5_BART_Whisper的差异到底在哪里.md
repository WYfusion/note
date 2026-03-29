---
tags:
  - LLM/架构
  - LLM/多模态
created: 2026-03-29
updated: 2026-03-29
---

# T5、BART、Whisper 的差异到底在哪里

## 问题定义

同样是 Encoder-Decoder，T5、BART 和 Whisper 的侧重点为什么不一样？

## 直觉解释

骨架相似，但训练目标和输入模态不同，于是模型性格不同。

## 形式化推导

T5 统一成 text-to-text，BART 强调去噪重建，Whisper 把音频作为 encoder 输入、文本作为 decoder 输出。

## 工程意义

同一骨架可以服务不同任务，只要训练目标和输入接口设计得当。

## 常见误解

> [!warning] 常见误解
> - “架构一样，模型能力就差不多。” 不对，训练目标和输入模态会显著改变行为。

## 例子或反例

Whisper 的强项来自音频 encoder 和转录训练目标，而不是单靠“用了 Transformer”。

## 相关链接

- [[01_EncoderDecoder为什么天然适合条件生成|Encoder-Decoder 为什么天然适合条件生成]]
- [[02_EncoderDecoder_T5_BART_Whisper|Encoder-Decoder]]
- [[05_跨模态Transformer_语音_视觉_视频的结构适配|跨模态 Transformer]]
