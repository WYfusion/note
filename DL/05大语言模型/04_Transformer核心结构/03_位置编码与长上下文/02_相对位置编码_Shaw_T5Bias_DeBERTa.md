---
tags:
  - LLM/架构
aliases:
  - 相对位置编码
created: 2026-03-28
updated: 2026-03-29
---

# 相对位置编码：Shaw、T5 Bias 与 DeBERTa

> [!abstract] 摘要
> 相对位置方案的出发点是：很多任务真正关心的是“离多远、在左还是在右”，而不是“我是第 137 个 token”。于是位置不再只是输入侧加法，而是直接进入 attention 关系。

## 这页先讲清什么

- 相对位置方案相对绝对位置到底改了什么。
- Shaw、T5 Relative Bias、DeBERTa 分别把位置信息加在什么地方。
- 为什么相对位置更贴近关系建模，但实现也更复杂。

## 关键结论

- 相对位置更关注 token 间的距离和方向，而不是孤立编号。
- T5 Bias 之所以常见，是因为它直接改 logits、实现简单、工程兼容性好。
- DeBERTa 的价值在于把内容和位置解耦，而不是简单再加一个 bias。

## 子页导航

- [[01_Shaw相对位置编码怎么把距离写进注意力|Shaw 相对位置编码怎么把距离写进注意力]]
- [[02_T5RelativeBias为什么工程上更常用|T5 Relative Bias 为什么工程上更常用]]
- [[03_DeBERTa为什么要把内容和位置拆开|DeBERTa 为什么要把内容和位置拆开]]

## 最短闭环解释

绝对位置告诉模型“我是谁”，相对位置更像在告诉模型“我和你是什么关系”。这种思路特别适合语言、语音这类关系比编号更重要的任务。

Shaw 类方法会把相对距离项写进 key/value 交互；T5 Bias 直接把距离信息写到 attention logits 上；DeBERTa 则进一步把内容表示和位置表示拆开，分别建模它们之间的关系。

所以，相对位置的意义不是“另一种位置 embedding”，而是把位置从输入特征提升为注意力关系的一部分。

## 相关链接

- [[01_绝对位置编码_Sinusoidal_Learnable|绝对位置编码]]
- [[03_RoPE_ALiBi_为什么主流模型偏向这两类方案|RoPE 与 ALiBi]]
- [[02_AttentionMask_Causal_Padding_CrossAttention|Attention Mask]]
