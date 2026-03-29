---
tags:
  - LLM/架构
  - 长上下文
created: 2026-03-29
updated: 2026-03-29
---

# NTK-aware Scaling 改变了什么频率几何

## 问题定义

NTK-aware scaling 不是简单把位置缩放一下，它在 RoPE 的频率几何上到底改了什么？

## 直觉解释

它试图重新分配不同频段的展开速度，让超长位置不会过快进入相位过密、模型难以分辨的区域。

## 形式化推导

RoPE 的外推问题很大程度来自频率基在长距离下分布失衡。NTK-aware scaling 通过调整底频或缩放方式，让模型在更长位置上保持更接近训练期的核行为。

## 工程意义

它往往比简单插值更贴近 RoPE 的原有几何，因此扩窗后中短上下文性能更容易保住，但调参通常也更讲经验。

## 常见误解

> [!warning] 常见误解
> - “NTK 只是一个神秘缩放系数。” 不对，本质是在修复位置频率分布。

## 例子或反例

若只做粗暴插值，可能把远距离压得太狠；若做更细的频率重标定，模型在中长距离上通常会更平滑。

## 相关链接

- [[01_PositionInterpolation在补什么|Position Interpolation 在补什么]]
- [[03_YaRN与LongRoPE分别解决什么问题|YaRN 与 LongRoPE 分别解决什么问题]]
- [[04_长度外推_PositionInterpolation_NTK_YaRN_LongRoPE|长度外推]]
