---
tags:
  - LLM/架构
  - KVCache
created: 2026-03-29
updated: 2026-03-29
---

# 为什么只缓存 K 和 V，不缓存 Q

## 问题定义

既然 K/V 要缓存，为什么 Q 通常不缓存？是不是缓存更多中间量就更快？

## 直觉解释

K/V 像历史档案，后面每一步都要翻；Q 像当前提问，只在这一刻使用一次。

## 形式化推导

第 $t$ 步的 query 只用于当前 token 与历史 K/V 做匹配；到了第 $t+1$ 步，新的 token 会产生新的 query，旧 query 不再被复用。相反，历史 token 的 K/V 会被未来所有步骤重复读取。

## 工程意义

缓存策略的核心不是“能不能存”，而是“存了是否值得”。Q 复用率低，因此通常不值得额外占缓存；attention score 更大，也往往不值得留。

## 常见误解

> [!warning] 常见误解
> - “所有中间量都缓存起来一定最好。” 不对，很多中间量存储成本高于重算收益。
> - “Q 不缓存是因为技术上做不到。” 不对，是因为复用价值太低。

## 例子或反例

若强行缓存所有历史 Q，不仅占显存，还几乎不带来下一步收益，因为下一步只需要新 token 的 query。

## 相关链接

- [[01_Prefill与Decode为什么成本完全不同|Prefill 与 Decode 为什么成本完全不同]]
- [[03_PagedAttention如何解决KV内存碎片|PagedAttention 如何解决 KV 内存碎片]]
-[[00_KVCache_Prefill_Decode_PagedAttention|KV Cache]]]

