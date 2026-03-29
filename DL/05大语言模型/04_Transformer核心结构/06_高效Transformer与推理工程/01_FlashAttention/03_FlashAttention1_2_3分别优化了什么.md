---
tags:
  - LLM/训练
  - FlashAttention
created: 2026-03-29
updated: 2026-03-29
---

# FlashAttention 1、2、3 分别优化了什么

## 问题定义

FlashAttention 1、2、3 看起来都在加速 attention，它们各自往前推进了哪一步？

## 直觉解释

FlashAttention-1 先把最明显的中间矩阵 IO 浪费去掉，后续版本再继续优化并行划分、硬件利用率和低精度路径。

## 形式化推导

- FlashAttention-1：在线 softmax + tile 计算。
- FlashAttention-2：更好的工作划分和更高硬件利用率。
- FlashAttention-3：更面向新硬件和低精度特性。

## 工程意义

版本迭代说明：真正的高效实现是持续贴着硬件演化的，而不是一篇论文后就结束。

## 常见误解

> [!warning] 常见误解
> - “后续版本只是小修小补。” 不对，并行策略和硬件映射会显著影响真实吞吐。

## 例子或反例

同样是 dense attention，不同版本 kernel 在同一 GPU 上的真实吞吐差异可能非常明显。

## 相关链接

- [[01_FlashAttention为什么是IO优化而不是复杂度革命|FlashAttention 为什么是 IO 优化而不是复杂度革命]]
- [[02_在线Softmax与块计算怎么保证数值正确|在线 Softmax 与块计算怎么保证数值正确]]
- [[01_FlashAttention1_2_3_为什么瓶颈是IO不是FLOPs|FlashAttention]]
