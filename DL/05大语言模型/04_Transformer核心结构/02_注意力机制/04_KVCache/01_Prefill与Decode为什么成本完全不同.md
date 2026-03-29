---
tags:
  - LLM/架构
  - KVCache
created: 2026-03-29
updated: 2026-03-29
---

# Prefill 与 Decode 为什么成本完全不同

## 问题定义

为什么同一个模型，prefill 阶段像大吞吐矩阵计算，decode 阶段却更像被带宽卡住的串行过程？

## 直觉解释

Prefill 是“整段一起读”，decode 是“每次只写一个字但要翻完整本历史笔记”。

## 形式化推导

Prefill 时，整段长度 $L$ 一起做投影和 attention，可充分并行；decode 时，每一步只有一个新 query，但仍要与历史 $L$ 个 K/V 交互，因此单步计算小、访存读历史重。

## 工程意义

这解释了很多现象：

- Prefill 吃算力。
- Decode 吃带宽和 cache 管理。
- 同一个优化对两阶段收益可能完全不同。

## 常见误解

> [!warning] 常见误解
> - “推理就是把训练缩小。” 不对。prefill 和 decode 的性能画像完全不同。
> - “只要 FLOPs 降了，decode 就一定更快。” 不对，带宽与调度常是主瓶颈。

## 例子或反例

同样生成 100 个 token，长前缀下 decode 往往远比 prefill 更慢，因为每一步都要重新读取越来越长的历史 K/V。

## 相关链接

- [[02_为什么只缓存K和V不缓存Q|为什么只缓存 K 和 V，不缓存 Q]]
- [[03_PagedAttention如何解决KV内存碎片|PagedAttention 如何解决 KV 内存碎片]]
- [[04_KVCache_Prefill_Decode_PagedAttention|KV Cache]]

