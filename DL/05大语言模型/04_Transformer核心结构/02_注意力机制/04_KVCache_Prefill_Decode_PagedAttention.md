---
tags:
  - LLM/架构
  - KVCache
created: 2026-03-28
updated: 2026-03-29
---

# KV Cache：Prefill、Decode 与 PagedAttention

> [!abstract] 摘要
> 自回归推理之所以慢，并不是模型不会并行矩阵乘法，而是每生成一个新 token，都要重新和整段历史交互。KV cache 的本质是把可复用的历史 Key / Value 留下来，把 decode 阶段的重复前缀计算尽量砍掉。

## 这页先讲清什么

- prefill 与 decode 为什么是两条完全不同的成本路径。
- 为什么缓存 K/V 就够了，而缓存 Q 往往没意义。
- 为什么 PagedAttention 优化的是内存管理，而不是注意力公式。

## 关键结论

- Prefill 更像整段批量计算，decode 更像单步追加但反复读取长历史。
- Q 只服务当前步，K/V 会被未来所有步复用，所以缓存价值完全不同。
- PagedAttention 解决的是长上下文服务中的页式存储、碎片和调度问题。

## 子页导航

-[[01_Prefill与Decode为什么成本完全不同|Prefill 与 Decode 为什么成本完全不同]]]
-[[02_为什么只缓存K和V不缓存Q|为什么只缓存 K 和 V，不缓存 Q]]]
-[[03_PagedAttention如何解决KV内存碎片|PagedAttention 如何解决 KV 内存碎片]]]

## 最短闭环解释

当模型第一次看到前缀时，它需要为整段序列算出所有层的 K/V，这一步叫 prefill。之后每生成一个新 token，旧 token 的 K/V 不会变化，于是可以缓存下来；新步骤只需计算新 token 的 Q/K/V，然后用新 Q 去读取历史 K/V，这就是 decode。

这里的核心不是“缓存一切中间量”，而是只缓存真正会被复用、且重算代价高于存储代价的那部分。Q 只在当前步使用一次，所以通常没必要缓存；attention score 矩阵更大，重算常常反而更划算。

请求一多、上下文一长，KV cache 自身就会成为系统瓶颈：显存占用、碎片、页分配、跨请求复用、带宽读取都会变成问题。PagedAttention 正是在修这层系统路径。

## 相关链接

- [[03_多头注意力为什么有效_MHA_MQA_GQA_MLA|多头注意力]]
-[[01_FlashAttention1_2_3_为什么瓶颈是IO不是FLOPs|FlashAttention]]]
-[[04_推理优化_投机解码_并行解码_接受校正|推理优化]]]

