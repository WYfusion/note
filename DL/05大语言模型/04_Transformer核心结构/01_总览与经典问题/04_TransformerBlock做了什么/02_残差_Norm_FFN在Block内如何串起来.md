---
tags:
  - LLM/架构
  - LLM/训练
created: 2026-03-29
updated: 2026-03-29
---

# 残差、Norm、FFN 在 Block 内如何串起来

## 问题定义

为什么现代 Transformer block 往往写成“Norm -> Attention -> 残差 -> Norm -> FFN -> 残差”的骨架？顺序改变为什么会影响训练稳定性？

## 直觉解释

Norm 像是在进入子层前先把尺度校平，残差像给子层保留一条主路，FFN 像对子层输出做深加工。顺序不只是排版问题，而是在定义梯度如何流动。

## 形式化推导

常见 Pre-Norm block 可写成：

$$
x' = x + \operatorname{Attn}(\operatorname{Norm}(x))
$$

$$
y = x' + \operatorname{FFN}(\operatorname{Norm}(x'))
$$

因为残差直接保留了恒等映射项，梯度对上一层至少有一条显式主通路；Norm 放在前面则避免子层直接吃到过于漂移的输入尺度。

## 工程意义

这套骨架不是“最优美的公式分层”，而是深层可训练性筛出来的结果。现代 LLM 之所以常见 Pre-Norm + RMSNorm + Gated FFN，背后都是数值稳定与扩深需求。

## 常见误解

> [!warning] 常见误解
> - “Block 内顺序无所谓，只要组件都在就行。” 不对。顺序直接影响梯度路径和激活统计。
> - “残差只是为了防止信息丢失。” 不完整。它更重要的作用是保留稳定梯度通路。

## 例子或反例

早期 Post-Norm Transformer 在层数很深时更容易训不稳，现代 LLM 大多改成 Pre-Norm，不是流行趋势，而是训练动力学上的选择。

## 相关链接

- [[00_AddNorm_PreNorm_PostNorm_LayerNorm_RMSNorm|Norm 与残差骨架]]
- [[00_残差路径_初始化_DeepNorm|残差路径与 DeepNorm]]
- [[00_一个TransformerBlock到底在做什么_信息混合与特征变换|一个 Transformer Block 在做什么]]

