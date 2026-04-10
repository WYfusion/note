---
tags:
  - LLM/训练
  - RMSNorm
created: 2026-03-29
updated: 2026-03-29
---

# LayerNorm 与 RMSNorm 到底在归一化什么

## 问题定义

LayerNorm 和 RMSNorm 的区别，不只是少了一个均值项，而是对“什么最值得控制”的判断不同。

## 直觉解释

LayerNorm 既把中心挪回去，也把尺度控住；RMSNorm 更像在说：对很多大模型来说，真正关键的是尺度别乱跑。

## 形式化推导

LayerNorm 对每个样本的特征维做去均值和方差归一化；RMSNorm 只按均方根做缩放，不显式去均值。

## 工程意义

RMSNorm 更轻、更简单，在大模型里常足够稳定，因此被广泛采用。

## 常见误解

> [!warning] 常见误解
> - “少做一步一定更差。” 不对，很多时候这一步并非必要。

## 例子或反例

LLaMA、Qwen 一类模型广泛使用 RMSNorm，说明零均值并不是深层训练的唯一关键。

## 相关链接

- [[01_PreNorm与PostNorm谁的梯度路径更稳|PreNorm 与 PostNorm 谁的梯度路径更稳]]
- [[00_AddNorm_PreNorm_PostNorm_LayerNorm_RMSNorm|Norm 与残差骨架]]
- [[00_训练稳定性_梯度传播_数值精度_失败模式|训练稳定性]]
