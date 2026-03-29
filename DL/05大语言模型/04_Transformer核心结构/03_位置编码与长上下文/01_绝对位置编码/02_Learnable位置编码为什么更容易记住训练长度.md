---
tags:
  - LLM/架构
created: 2026-03-29
updated: 2026-03-29
---

# Learnable 位置编码为什么更容易记住训练长度

## 问题定义

为什么 learnable absolute positional embedding 在训练长度附近往往很好用，但一扩窗就更容易失效？

## 直觉解释

它像给每个位置开了一张专属记忆卡，训练时见过的位置都能记得很细，但没见过的位置没有天然规律可外推。

## 形式化推导

Learnable 方案本质上是查表：

$$
\tilde{x}_i = x_i + e_i
$$

其中 $e_i$ 是第 $i$ 个位置的可学习向量。若训练只覆盖到长度 $L_{\text{train}}$，那么超过这个范围的位置向量要么不存在，要么只能靠插值或随机初始化补上。

## 工程意义

这类方案适合固定长度任务、固定分辨率视觉任务，但在长上下文 LLM 中往往不够稳。很多扩窗失败，本质上是模型把位置表记成了训练分布内的编号记忆。

## 常见误解

> [!warning] 常见误解
> - “可学习一定更强。” 不对，它只是更灵活，不代表更能泛化。
> - “插值位置 embedding 就等于解决外推。” 不完整，插值只是补救。

## 例子或反例

ViT 常要在分辨率变化时对 learnable 位置 embedding 做插值，这说明它和训练网格绑定得很紧。

## 相关链接

- [[01_Sinusoidal位置编码的频率直觉|Sinusoidal 位置编码的频率直觉]]
- [[04_长度外推_PositionInterpolation_NTK_YaRN_LongRoPE|长度外推]]
- [[01_绝对位置编码_Sinusoidal_Learnable|绝对位置编码]]
