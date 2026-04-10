---
tags:
  - LLM/架构
created: 2026-03-29
updated: 2026-03-29
---

# Encoder-Decoder 数据流与 Cross-Attention 位置

## 问题定义

Encoder-Decoder 相比 Decoder-only 多了什么数据流？Cross-attention 究竟插在哪里，起什么作用？

## 直觉解释

Encoder 像“把源输入读懂并记在脑子里”，Decoder 像“看着这份记忆一边写一边生成目标输出”。Cross-attention 就是 Decoder 读取这份外部记忆的接口。

## 形式化推导

源序列路径：

$$
X_{\text{src}} \to \text{Encoder} \to H_{\text{enc}} \in \mathbb{R}^{B \times L_s \times D}
$$

目标序列路径：

$$
X_{\text{tgt}} \to \text{Decoder Self-Attn} \to \text{Cross-Attn}(Q_{\text{dec}}, K_{\text{enc}}, V_{\text{enc}}) \to \text{FFN}
$$

因此 cross-attention 往往位于 decoder block 内部，介于 decoder 自注意力与 FFN 之间。

## 工程意义

这种拆分让输入输出长度、模态、采样率可以完全不同。翻译里源句和目标句长度不同，Whisper 里音频帧和文本 token 粒度差异更大，靠的都是这条桥梁。

## 常见误解

> [!warning] 常见误解
> - “cross-attention 只是多一次 attention。” 不完整。它是两个序列流之间的结构性接口。
> - “Encoder-Decoder 只是 Decoder-only 多几层。” 不对，信息流类型已经变了。

## 例子或反例

ASR 任务里，若把长音频帧直接塞进单路 Decoder-only，训练和推理都可能很重；先用 encoder 压缩音频，再让 decoder 生成文本通常更自然。

## 相关链接

- [[01_DecoderOnly数据流与张量形状|Decoder-Only 数据流与张量形状]]
- [[00_AttentionMask_Causal_Padding_CrossAttention|Attention Mask]]
- [[00_EncoderDecoder_T5_BART_Whisper|Encoder-Decoder]]

