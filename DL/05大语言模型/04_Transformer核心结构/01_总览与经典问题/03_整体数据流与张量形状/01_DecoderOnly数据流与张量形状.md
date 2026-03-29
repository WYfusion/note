---
tags:
  - LLM/架构
created: 2026-03-29
updated: 2026-03-29
---

# Decoder-Only 数据流与张量形状

## 问题定义

Decoder-only LLM 从 `input_ids` 到 logits，主张量如何流动？Prefill 和 decode 的张量路径为什么不同？

## 直觉解释

训练时整段序列一起进模型，像一次性批改整篇文章；推理时每次只新来一个 token，更像在已有稿子后面续一字一句。

## 形式化推导

典型路径为：

$$
[B, L] \xrightarrow{\text{Embedding}} [B, L, D]
\xrightarrow{\text{N 个 Blocks}} [B, L, D]
\xrightarrow{\text{LM Head}} [B, L, V]
$$

其中每个 Block 内部仍保持 `[B, L, D]` 主形状不变。训练时使用 causal mask，整段并行；推理 prefill 时也是整段前缀并行，decode 时则近似变成 `[B, 1, D]` 的单步追加。

## 工程意义

理解这条主线后，KV cache 的意义就很清楚了：历史 token 的 K/V 可以缓存，新 token 只需再走一遍局部路径。很多系统优化都围绕“如何让 decode 这条单步路径更快”展开。

## 常见误解

> [!warning] 常见误解
> - “推理时形状一直是 `[B, L, D]`。” 不完整。逻辑历史长度在增长，但计算上常只新算 `[B, 1, D]`。
> - “attention 会改变主通道维度。” 通常不会，变的是内部投影和信息内容。

## 例子或反例

聊天模型看到 4K 前缀时，prefill 是一次大批量矩阵运算；开始生成后，每一步只追加一个 token，但仍要与 4K 历史 K/V 交互，这就是 decode 越往后越受带宽影响的原因。

## 相关链接

- [[02_EncoderDecoder数据流与CrossAttention位置|Encoder-Decoder 数据流与 Cross-Attention 位置]]
- [[04_KVCache_Prefill_Decode_PagedAttention|KV Cache]]
-[[03_DecoderOnly_GPT_LLaMA_Qwen|Decoder-Only]]]

