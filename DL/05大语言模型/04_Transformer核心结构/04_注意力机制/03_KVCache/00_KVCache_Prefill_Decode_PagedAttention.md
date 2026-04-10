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


## 最短闭环解释

当模型第一次看到前缀时，它需要为整段序列算出所有层的 K/V，这一步叫 prefill。之后每生成一个新 token，旧 token 的 K/V 不会变化，于是可以缓存下来；新步骤只需计算新 token 的 Q/K/V，然后用新 Q 去读取历史 K/V，这就是 decode。

> [!note]
> 所以 Prefill 和 Decode 的区别，不是“目标不同”那么简单，而是：
> - 输入形状不同
> - 复用模式不同
> - cache 读写模式不同
> - 用户体验指标不同
>
> 但它们仍然属于**同一条生成链路**。

这里的核心不是“缓存一切中间量”，而是只缓存真正会被复用、且重算代价高于存储代价的那部分。Q 只在当前步使用一次，所以通常没必要缓存；attention score 矩阵更大，重算常常反而更划算。

请求一多、上下文一长，KV cache 自身就会成为系统瓶颈：显存占用、碎片、页分配、跨请求复用、带宽读取都会变成问题。PagedAttention 正是在修这层系统路径。


## 一个最容易误解的点：Prefill 和 Decode 不是对立概念

> [!important]
> 对于**单个自回归生成请求**来说，Prefill 和 Decode 不是“二选一”的两种模式，也不是“推荐谁替代谁”的两条路线。  
> 它们通常是**同一条请求生命周期中的前后两个阶段**。

一个最标准的时间线是：

1. 用户提交 prompt
2. 系统先执行 **Prefill**
3. 产出第一个可采样的 logits
4. 采样出第一个输出 token
5. 系统进入 **Decode**
6. 每步再生成一个新 token，直到停止

所以更准确的理解应该是：

- Prefill 负责“把已有前缀读进去，并建立历史”
- Decode 负责“在已有历史上继续往后生成”

真正需要做权衡的，不是“这次请求到底选 Prefill 还是 Decode”，而是：

- 在**多请求并发**时，调度器要不要优先照顾新请求的 prefill
- 还是优先照顾老请求的 decode

这两个阶段的关系更像：

- 编译器里的“初始化阶段 + 主循环”
- 数据库里的“装载索引 + 反复查询”

而不是两个互相竞争、只能保留一个的方案。

## TTFT 是什么

`TTFT` 是 **Time To First Token**，即：

> 从用户发出请求，到系统把**第一个生成 token** 返回给用户所经历的时间。

### 为什么这个指标重要

因为对用户来说，最先感知到的不是总吞吐，而是：

- 发送请求后要等多久，界面才开始“出字”

所以 TTFT 直接影响“这个模型是否显得卡顿”。

### TTFT 在工程上通常由哪些部分组成

如果写得稍微细一点，可以把它理解为：

$$
\text{TTFT}
\approx
\text{排队等待}
+
\text{Prefill 计算}
+
\text{首次采样与返回}
$$

也就是说，TTFT 往往不只包含纯模型前向，还会包含：

- 请求排队等待调度
- prompt tokenization 等前处理
- Prefill 前向计算
- 第一次采样
- 把第一个 token flush 给客户端的开销

### 为什么 TTFT 常和 Prefill 强相关

因为在标准自回归生成里：

- 在 Prefill 完成前，系统通常拿不到“开始生成”的基础状态
- 因此第一个 token 往往要等到 Prefill 基本完成后才能产生

所以从模型执行角度看：

- **Prefill 更影响 TTFT**
- **Decode 更影响后续每个 token 的间隔**

### 那 ITL 又是什么

和 TTFT 经常一起出现的另一个指标是 `ITL`，即 **Inter-Token Latency**：

> 相邻两个输出 token 之间的时间间隔。

因此可以简单记成：

| 指标 | 关心的问题 |
| --- | --- |
| TTFT | 用户多久看到第一个 token |
| ITL | 之后每个 token 出得快不快 |
| Throughput / TPS | 系统整体每秒能处理多少 token |

## 子页导航

- [[01_Prefill与Decode为什么成本完全不同|Prefill 与 Decode 为什么成本完全不同]]
- [[02_为什么只缓存K和V不缓存Q|为什么只缓存 K 和 V，不缓存 Q]]
- [[03_PagedAttention如何解决KV内存碎片|PagedAttention 如何解决 KV 内存碎片]]

## 相关链接

- [[00_多头注意力为什么有效_MHA_MQA_GQA_MLA|多头注意力]]
- [[00_FlashAttention1_2_3_为什么瓶颈是IO不是FLOPs|FlashAttention]]
- [[00_推理优化_投机解码_并行解码_接受校正|推理优化]]
