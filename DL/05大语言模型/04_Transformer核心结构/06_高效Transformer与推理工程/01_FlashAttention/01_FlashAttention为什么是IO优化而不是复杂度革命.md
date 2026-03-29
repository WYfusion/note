---
tags:
  - LLM/训练
  - FlashAttention
created: 2026-03-29
updated: 2026-03-29
---

# FlashAttention 为什么是 IO 优化而不是复杂度革命

## 问题定义

为什么说 FlashAttention 的突破主要在 IO，而不是把 dense attention 的理论复杂度推翻重来？

## 直觉解释

它做的不是少算很多，而是少搬很多。算术关系还在，浪费的内存往返被砍掉了。

## 形式化推导

Q 与 K 的两两关系仍然存在，因此连接模式仍是二次；改动在于不再物化完整的 score/probability 矩阵，而是按块计算并直接累积结果。

## 工程意义

这就是为什么它常能显著提速，却不意味着 attention 复杂度突然变成线性。

## 常见误解

> [!warning] 常见误解
> - “FlashAttention = 线性 attention。” 错。

## 例子或反例

序列翻倍后，FlashAttention 仍然要处理更多 pairwise 关系，只是中间矩阵不再完整落地。

## 相关链接

- [[02_在线Softmax与块计算怎么保证数值正确|在线 Softmax 与块计算怎么保证数值正确]]
- [[03_FlashAttention1_2_3分别优化了什么|FlashAttention 1、2、3 分别优化了什么]]
- [[01_FlashAttention1_2_3_为什么瓶颈是IO不是FLOPs|FlashAttention]]
