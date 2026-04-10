---
tags:
  - LLM/架构
aliases:
  - Attention Mask
created: 2026-03-28
updated: 2026-03-29
---

# Attention Mask：Causal、Padding 与 Cross-Attention

> [!abstract] 摘要
> Mask 不是 attention 的边角料，而是信息流约束器。它决定谁能看谁，因此直接决定模型是在做双向理解、自回归生成，还是条件生成。

## 这页先讲清什么

- Padding、causal、cross-attention mask 各自屏蔽的是什么。
- 为什么 mask 与位置编码不是一回事。
- 为什么很多推理 bug 实际上来自 mask 逻辑或广播形状错误。

## 关键结论

- Padding mask 屏蔽的是“无效内容”，causal mask 屏蔽的是“未来信息”，cross-attention mask 屏蔽的是“条件输入中的无效区域”。
- Mask 决定可见性，位置编码决定几何；两者功能不同，不能互相替代。
- 在变长、多模态和流式场景里，mask 往往同时承担语义约束和系统约束。

## 子页导航

- [[01_PaddingMask为什么不是小细节|Padding Mask 为什么不是小细节]]]
- [[02_CausalMask为什么不等于位置编码|Causal Mask 为什么不等于位置编码]]]
- [[03_CrossAttentionMask如何约束条件信息流|Cross-Attention Mask 如何约束条件信息流]]]

## 最短闭环解释

attention 默认会让当前位置去看所有位置，但很多任务并不允许这样。分类模型不该看 padding 垃圾位，自回归解码不该偷看未来，Seq2Seq decoder 也不该把 encoder 的空白位置当有效上下文。这就是 mask 的职责。

需要特别区分的是：mask 只规定“允许不允许看”，并不告诉模型“两个位置差了多远”或“谁在谁前面多少步”。所以 causal mask 能防止作弊，却不能替代顺序编码。

工程上，mask 往往比公式更容易出错，因为它涉及形状广播、dtype、填充值和 kernel 约束。一处 mask 错误，轻则性能退化，重则直接 NaN 或信息泄露。

## 相关链接

- [[01_绝对位置编码_Sinusoidal_Learnable|绝对位置编码]]
- [[00_Transformer整体数据流_张量形状_EncoderDecoder|整体数据流与张量形状]]
- [[00_KVCache_Prefill_Decode_PagedAttention|KV Cache]]

