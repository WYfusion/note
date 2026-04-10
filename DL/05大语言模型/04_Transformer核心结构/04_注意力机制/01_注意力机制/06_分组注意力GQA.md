---
tags:
  - LLM/Transformer
  - 注意力机制/GQA
  - 推理/KVCache
aliases:
  - Grouped-Query Attention
  - GQA
updated: 2026-03-29
---

# 分组注意力（Grouped-Query Attention, GQA）

> [!abstract]
> GQA 是 MHA 和 [[05_多查询注意力MQA|MQA]] 之间的折中方案：保留多个 Query 头，但让若干个 Query 头共享同一组 Key/Value。

GQA 的思路可以概括为：Query 头数很多，Key/Value 头数较少。  
它不是完全共享，也不是完全独立，而是在两者之间找一个工程上更稳的平衡点。 ^21f310

## 为什么需要 GQA

标准 MHA 表达力强，但 KV cache 重。  
MQA 足够省，但共享一组 $K/V$ 往往压缩得太狠。

GQA 的目标就是：

- 保留大部分多头 Query 的表达能力
- 同时降低推理时的带宽与缓存成本

## 结构直觉

假设总共有 $h$ 个 Query 头，把它们分成 $g$ 个组，每组共享一套 $K/V$。

那么：

- 当 $g = h$ 时，GQA 退化为 MHA
- 当 $g = 1$ 时，GQA 退化为 [[05_多查询注意力MQA|MQA]]

所以 GQA 可以看成一条连续的折中曲线，而不是一个全新范式。

## 数学形式

设输入

$$
X \in \mathbb{R}^{B \times L \times d_{model}}
$$

总 Query 头数为 $h$，组数为 $g$，则每组包含

$$
m = h / g
$$

个 Query 头。

### 1. Query 仍然保持多头

$$
Q_i = XW_i^Q,\quad i \in [1,h]
$$

### 2. Key/Value 只保留 $g$ 组

$$
K_j = XW_j^K,\quad V_j = XW_j^V,\quad j \in [1,g]
$$

### 3. 第 $i$ 个 Query 头只访问自己所属组的 K/V

记组映射函数为 $G(i)$，则有：

$$
A_i = \text{softmax}\left(\frac{Q_i K_{G(i)}^\top}{\sqrt{d_h}}\right)
$$

$$
\text{head}_i = A_i V_{G(i)}
$$

### 4. 最后照常拼接输出

$$
\text{GQA}(X)=\text{Concat}(\text{head}_1,\dots,\text{head}_h)W_O
$$

## 三种方案怎么对比

| 方案 | Query 头数 | K/V 头数 | KV cache 成本 | 表达能力 |
| --- | --- | --- | --- | --- |
| MHA | $h$ | $h$ | 高 | 强 |
| GQA | $h$ | $g$ | 中 | 中高 |
| [[05_多查询注意力MQA|MQA]] | $h$ | $1$ | 最低 | 较弱 |

如果以 MHA 的 KV cache 为基准，那么 GQA 的缓存规模大致缩减为：

$$
\frac{g}{h}
$$

倍。

## 为什么 GQA 往往比 [[05_多查询注意力MQA|MQA]] 更稳

### 1. 共享没有压到极限

[[05_多查询注意力MQA|MQA]] 把所有头的 $K/V$ 都压成一组，GQA 则保留多组，表达退化会更轻。

### 2. Query 仍然保持多样性

不同 Query 头仍然在不同子空间里发问，只是它们分组去读取条件信息。

### 3. 更适合“性能与成本都要”的场景

很多现代 LLM 并不是只追求理论上最省，而是要在质量、显存和吞吐之间平衡。  
GQA 往往比极端的 [[05_多查询注意力MQA|MQA]] 更符合这个目标。

## 使用上的经验判断

- 如果你只关心推理极致省内存，可以优先考虑 [[05_多查询注意力MQA|MQA]]
- 如果你还想保留比较强的表达能力，GQA 往往更稳
- 如果序列不长、推理压力不大，标准 MHA 仍然是最直接的方案
- 如果你希望不只是减少 K/V 头数，而是直接压缩历史 K/V 的表示瓶颈，可以继续看 [[08_多头潜变量注意力MLA|MLA（多头潜变量注意力）]]

## 相关双链

- [[索引_注意力机制]]
- [[02_多头自注意力MHA]]
- [[05_多查询注意力MQA]]
- [[08_多头潜变量注意力MLA|MLA（多头潜变量注意力）]]
- [[03_MQA_GQA_MLA如何做带宽折中|MQA、GQA、MLA 如何做带宽折中]]
- [[00_KVCache_Prefill_Decode_PagedAttention|KV Cache 总览]]
