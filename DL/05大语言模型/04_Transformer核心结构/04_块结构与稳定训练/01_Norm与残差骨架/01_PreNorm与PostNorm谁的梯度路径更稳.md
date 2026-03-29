---
tags:
  - LLM/训练
created: 2026-03-29
updated: 2026-03-29
---

# PreNorm 与 PostNorm 谁的梯度路径更稳

## 问题定义

为什么现代 LLM 通常更偏爱 Pre-Norm，而不是原始 Transformer 的 Post-Norm？

## 直觉解释

Pre-Norm 让残差主路更接近恒等映射，梯度穿层时更容易保留一条清晰主通道。

## 形式化推导

Pre-Norm：

$$
x_{l+1} = x_l + F_l(\operatorname{Norm}(x_l))
$$

Post-Norm：

$$
x_{l+1} = \operatorname{Norm}(x_l + F_l(x_l))
$$

Pre-Norm 对 $x_l$ 明确保留了恒等项，深层时通常更稳。

## 工程意义

Pre-Norm 更适合深层堆叠和大规模训练，因此在现代 LLM 中几乎成了默认骨架。

## 常见误解

> [!warning] 常见误解
> - “Pre-Norm 全面优于 Post-Norm。” 不绝对，只是更稳更常用。

## 例子或反例

原始 Transformer 的 Post-Norm 在层数较浅时可行，但一旦层数大幅增加，训练常更敏感。

## 相关链接

- [[02_LayerNorm与RMSNorm到底在归一化什么|LayerNorm 与 RMSNorm 到底在归一化什么]]
- [[01_AddNorm_PreNorm_PostNorm_LayerNorm_RMSNorm|Norm 与残差骨架]]
- [[03_残差路径_初始化_DeepNorm|残差路径与 DeepNorm]]
