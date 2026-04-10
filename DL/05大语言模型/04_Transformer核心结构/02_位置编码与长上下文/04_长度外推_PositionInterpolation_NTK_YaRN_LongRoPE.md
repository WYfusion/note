---
tags:
  - LLM/架构
  - 位置外推
  - 长上下文
created: 2026-03-28
updated: 2026-03-29
---

# 长度外推：Position Interpolation、NTK、YaRN 与 LongRoPE

> [!abstract] 摘要
> 把上下文窗口参数改大，不等于模型真的学会更长上下文。真正的问题是：训练时学到的位置几何，在更长长度上是否仍然稳定、可解释、可利用。

## 这页先讲清什么

- 为什么扩窗首先是位置几何问题，而不是单纯算力问题。
- Position Interpolation、NTK-aware scaling、YaRN、LongRoPE 各自在修什么。
- 为什么扩窗补丁常常伴随继续训练或校准。

## 关键结论

- 许多长上下文退化首先来自位置失真，而不是 attention 公式本身。
- 扩窗补丁的共同目标，是把超长位置重新映射回模型还能理解的频率尺度。
- 真正可用的长上下文，必须同时验证位置补丁、cache、吞吐和任务收益。

## 子页导航

- [[01_PositionInterpolation在补什么|Position Interpolation 在补什么]]
- [[02_NTKAwareScaling改变了什么频率几何|NTK-aware Scaling 改变了什么频率几何]]
- [[03_YaRN与LongRoPE分别解决什么问题|YaRN 与 LongRoPE 分别解决什么问题]]

## 最短闭环解释

位置机制本质上定义了一个几何尺度。模型只在训练长度范围内见过这套几何，超出后，频率分布、相位分辨率、远距离关系都可能进入分布外。于是模型看起来“支持输入更长”，却未必真正能用好更长上下文。

Position Interpolation 的思路是把长位置压回训练范围，NTK-aware scaling 则更像重标定频率底座，YaRN 和 LongRoPE 则在更细粒度上调整不同频段的行为。

所以，扩窗从来不是一行配置，而是一套“位置几何 + 系统代价 + 任务验证”的联合工程。

## 相关链接

- [[03_RoPE_ALiBi_为什么主流模型偏向这两类方案|RoPE 与 ALiBi]]
- [[00_长上下文工程_Chunking_Streaming_RingAttention|长上下文工程]]
- [[00_FlashAttention1_2_3_为什么瓶颈是IO不是FLOPs|FlashAttention]]
