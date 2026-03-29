---
tags:
  - LLM/训练
  - FlashAttention
created: 2026-03-29
updated: 2026-03-29
---

# 在线 Softmax 与块计算怎么保证数值正确

## 问题定义

既然 FlashAttention 不再物化完整 logits，在线 softmax 怎么还能保证与标准 attention 一致？

## 直觉解释

关键在于按块维护每一行的最大值和归一化因子，把分段计算重新拼回全局正确结果。

## 形式化推导

在线 softmax 逐块更新每行的 running max 与 running sum，再据此重标定此前统计量，最终得到与整行一次性 softmax 等价的结果。

## 工程意义

这让 FlashAttention 在减少 HBM 往返的同时，保持 exact attention 语义。

## 常见误解

> [!warning] 常见误解
> - “块算就一定是近似。” 不对，关键看归一化统计是否严格维护。

## 例子或反例

若只做块内 softmax 而不做全局重标定，结果就会变成近似；FlashAttention 正是避免了这一点。

## 相关链接

- [[01_FlashAttention为什么是IO优化而不是复杂度革命|FlashAttention 为什么是 IO 优化而不是复杂度革命]]
- [[03_FlashAttention1_2_3分别优化了什么|FlashAttention 1、2、3 分别优化了什么]]
- [[01_FlashAttention1_2_3_为什么瓶颈是IO不是FLOPs|FlashAttention]]
