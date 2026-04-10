---
tags:
  - LLM/架构
created: 2026-03-28
updated: 2026-03-29
---

# Decoder-Only：GPT、LLaMA 与 Qwen

> [!abstract] 摘要
> Decoder-only 成为大语言模型主流，不是因为它在所有任务上都最好，而是因为它把训练目标、数据格式、推理接口统一到了极致。因果语言建模很简单，但这种简单性带来了巨大的工程扩展性。

## 这页先讲清什么

- Decoder-only 为什么特别适合大规模预训练。
- GPT、LLaMA、Qwen 共用的主骨架和差异在哪里。
- 这种统一性换来的代价是什么。

## 关键结论

- “给前缀，预测下一个 token”是最统一的训练和推理接口。
- 现代 LLM 的很多骨架演化都发生在 Decoder-only 主线内部。
- 它的弱点主要体现在条件结构不天然、decode 带宽压力大。

## 子页导航

- [[01_DecoderOnly为什么适合统一自回归建模|Decoder-Only 为什么适合统一自回归建模]]
- [[02_GPT_LLaMA_Qwen的骨架差异与共性|GPT、LLaMA、Qwen 的骨架差异与共性]]

## 最短闭环解释

Decoder-only 的强项在于统一。任何文本都能变成“前缀 -> 下一个 token”的训练样本，推理接口也天然统一成自回归生成。这种简单性极其适合做大规模数据清洗、预训练、微调、服务化和工具链整合。

GPT 奠定了这条路线，LLaMA、Qwen 等则在这条主干上不断把位置编码、Norm、FFN、[[06_分组注意力GQA|GQA]]、KV cache 等工程细节打磨成熟。

所以 Decoder-only 的主导地位，本质上是“结构统一性和工程可扩展性”的胜利，而不只是“参数更大”。

## 相关链接

- [[00_EncoderOnly_BERT_ViT|Encoder-Only]]
- [[00_EncoderDecoder_T5_BART_Whisper|Encoder-Decoder]]
- [[00_KVCache_Prefill_Decode_PagedAttention|KV Cache]]
