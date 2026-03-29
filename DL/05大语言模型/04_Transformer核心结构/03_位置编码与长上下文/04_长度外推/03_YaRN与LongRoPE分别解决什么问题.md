---
tags:
  - LLM/架构
  - 长上下文
created: 2026-03-29
updated: 2026-03-29
---

# YaRN 与 LongRoPE 分别解决什么问题

## 问题定义

YaRN 和 LongRoPE 都是扩窗补丁，但它们分别更偏向修什么问题？

## 直觉解释

YaRN 更像务实的分段重标定，LongRoPE 更像更细颗粒度的频率重分配。

## 形式化推导

两者都围绕 RoPE 频率失真展开，但 YaRN 更强调分段缩放和稳健扩窗，LongRoPE 更强调在极长范围内重新安排不同频段的角色。

## 工程意义

选择哪种补丁，往往取决于你的目标是“快速把窗口扩到可用”，还是“尽量把更长位置用得更稳、更细”。

## 常见误解

> [!warning] 常见误解
> - “有了这些补丁就不需要继续训练。” 不一定，很多场景仍需要校准或继续训练。

## 例子或反例

同样扩到 128K，上线前既要看是否能输入，也要看长文检索、跨段引用、多轮对话是否真的保持有效。

## 相关链接

- [[01_PositionInterpolation在补什么|Position Interpolation 在补什么]]
- [[02_NTKAwareScaling改变了什么频率几何|NTK-aware Scaling 改变了什么频率几何]]
- [[04_长度外推_PositionInterpolation_NTK_YaRN_LongRoPE|长度外推]]
