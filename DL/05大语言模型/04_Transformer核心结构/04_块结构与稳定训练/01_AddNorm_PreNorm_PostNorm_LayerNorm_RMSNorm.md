---
tags:
  - LLM/训练
  - RMSNorm
created: 2026-03-28
updated: 2026-03-29
---

# AddNorm、PreNorm、PostNorm、LayerNorm 与 RMSNorm

> [!abstract] 摘要
> Norm 和残差的排列决定了深层 Transformer 的梯度通路是否稳定。现代大模型偏爱 Pre-Norm 和 RMSNorm，不是偶然，而是训练动力学长期筛选出的结果。

## 这页先讲清什么

- 为什么 Pre-Norm 和 Post-Norm 会带来不同的梯度路径。
- LayerNorm 与 RMSNorm 分别在归一化什么。
- 为什么 Norm 位置会直接影响深层可训练性。

## 关键结论

- Pre-Norm 更容易保留显式恒等梯度通路。
- RMSNorm 关注尺度控制，不强制零均值，因此更轻、更常见于现代 LLM。
- Norm 不是数值小细节，而是大模型骨架的一部分。

## 子页导航

- [[01_PreNorm与PostNorm谁的梯度路径更稳|PreNorm 与 PostNorm 谁的梯度路径更稳]]
- [[02_LayerNorm与RMSNorm到底在归一化什么|LayerNorm 与 RMSNorm 到底在归一化什么]]

## 最短闭环解释

Transformer 想堆深，就必须回答一个问题：梯度怎么稳定穿过几十层甚至上百层。Pre-Norm 的做法是先把输入归一化，再送进 attention 或 FFN，然后通过残差把原输入直接加回来。这让梯度更容易保留一条清晰主路。

LayerNorm 会对均值和方差都做归一化，RMSNorm 则只控制均方根尺度。现代 LLM 之所以常用 RMSNorm，是因为很多时候“控住尺度”比“强制零均值”更重要，而且这样还能省一点计算和实现复杂度。

所以，Norm 系列问题的核心不是“哪一种公式更优雅”，而是“哪一种骨架在大规模训练里更不容易出问题”。

## 相关链接

- [[02_FFN_GELU_GEGLU_SwiGLU_为什么Attention后还要MLP|FFN 与 GLU]]
- [[03_残差路径_初始化_DeepNorm|残差路径与 DeepNorm]]
- [[04_训练稳定性_梯度传播_数值精度_失败模式|训练稳定性]]
