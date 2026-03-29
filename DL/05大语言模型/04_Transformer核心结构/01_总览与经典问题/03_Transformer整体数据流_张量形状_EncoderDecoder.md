---
tags:
  - LLM/架构
aliases:
  - Transformer 数据流与张量形状
created: 2026-03-28
updated: 2026-03-29
---

# Transformer 整体数据流：张量形状与 Encoder-Decoder

> [!abstract] 摘要
> 结构理解很多时候不是被公式卡住，而是被张量流卡住。把“输入 -> embedding -> 多层 block -> logits”这条数据流走清楚后，mask、cross-attention、KV cache、prefill 和 decode 的位置就都不再抽象。

## 这页先讲清什么

- Decoder-only、Encoder-only、Encoder-Decoder 在张量流上真正差在哪。
- 为什么大多数 Transformer block 都保持 `[B, L, D]` 的主形状不变。
- Cross-attention、mask、输出头分别插在数据流的什么位置。

## 关键结论

- 大多数 Transformer 层不改主形状，是为了让残差连接、层堆叠和 cache 设计保持稳定。
- Encoder-Decoder 的特殊之处不是“层更多”，而是存在一条显式的“源表示 -> 目标生成”桥梁。
- Prefill 和 decode 的差异，本质上也是数据流差异：前者整段并行，后者单步追加。

## 子页导航

-[[01_DecoderOnly数据流与张量形状|Decoder-Only 数据流与张量形状]]]
-[[02_EncoderDecoder数据流与CrossAttention位置|Encoder-Decoder 数据流与 Cross-Attention 位置]]]

## 最短闭环解释

以 Decoder-only 为例，离散 `input_ids` 先被映射到 `[B, L, D]` 的隐藏状态，然后经过多层 attention 和 FFN 反复改写，最后投到 `[B, L, V]` 的 logits。这里的关键不是每一层都变了多少，而是大多数层都保持主形状不变，于是残差、Norm、cache 和 kernel 优化都能围绕同一接口展开。

Encoder-Decoder 多了一步：输入先在 encoder 里被压成一组上下文表示，decoder 再在自回归生成时通过 cross-attention 去读取它。于是“理解输入”和“生成输出”被拆成两条不同但可连接的数据流。

一旦把这条主线理清，很多工程点就自然落位了。Mask 是约束信息流，position 是改变注意力几何，KV cache 是在 decode 阶段缓存那部分可复用状态，FlashAttention 则是优化 attention 子图里的内存路径。

## 相关链接

- [[04_一个TransformerBlock到底在做什么_信息混合与特征变换|一个 Transformer Block 在做什么]]
-[[02_AttentionMask_Causal_Padding_CrossAttention|Attention Mask]]]
-[[04_KVCache_Prefill_Decode_PagedAttention|KV Cache]]]
-[[02_EncoderDecoder_T5_BART_Whisper|Encoder-Decoder]]]

