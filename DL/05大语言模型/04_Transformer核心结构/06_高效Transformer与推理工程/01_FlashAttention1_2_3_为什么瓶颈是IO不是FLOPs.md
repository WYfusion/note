---
tags:
  - LLM/训练
  - FlashAttention
created: 2026-03-28
updated: 2026-03-29
---

# FlashAttention 1/2/3：为什么瓶颈是 IO 而不是 FLOPs

> [!abstract] 摘要
> FlashAttention 没有把 dense attention 的理论复杂度从二次变成一次。它解决的是更现实的问题：标准实现会物化巨大的中间矩阵，导致 HBM 读写而不是算术本身先成为瓶颈。

## 这页先讲清什么

- FlashAttention 为什么主要是 IO 优化。
- 在线 softmax 与块计算如何在不改语义的前提下降低中间矩阵落地。
- FlashAttention 1、2、3 分别把优化推进到了哪一步。

## 关键结论

- FlashAttention 仍是 exact attention，不是近似注意力。
- 它优化的是实现路径和内存流量，而不是理论连接模式。
- 长上下文系统里，FlashAttention 很重要，但不是唯一答案。

## 子页导航

- [[01_FlashAttention为什么是IO优化而不是复杂度革命|FlashAttention 为什么是 IO 优化而不是复杂度革命]]
- [[02_在线Softmax与块计算怎么保证数值正确|在线 Softmax 与块计算怎么保证数值正确]]
- [[03_FlashAttention1_2_3分别优化了什么|FlashAttention 1、2、3 分别优化了什么]]

## 最短闭环解释

标准 attention 往往会先算出 $QK^T$，再把 softmax 概率矩阵也显式存下来。这些中间量一长就是 $L \times L$，显存读写量非常大。FlashAttention 的核心改动是：不再把这些大矩阵完整落到 HBM，而是按块读 Q/K/V，在片上做在线 softmax 并直接累积输出。

这样做没有改变 attention 的数学定义，却显著减少了中间矩阵物化和回写。于是，瓶颈从“算不动”变成了“原来是 IO 浪费太大”。

所以 FlashAttention 的真正意义是让 dense attention 的实现更接近硬件友好极限，而不是发明了一个新的注意力理论。

## 相关链接

- [[02_稀疏_局部_线性_分块注意力_哪些真能替代全注意力|高效注意力变体]]
- [[04_KVCache_Prefill_Decode_PagedAttention|KV Cache]]
- [[03_长上下文工程_Chunking_Streaming_RingAttention|长上下文工程]]
