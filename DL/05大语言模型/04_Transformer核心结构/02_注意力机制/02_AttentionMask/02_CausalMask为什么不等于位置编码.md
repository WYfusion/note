---
tags:
  - LLM/架构
created: 2026-03-29
updated: 2026-03-29
---

# Causal Mask 为什么不等于位置编码

## 问题定义

自回归模型已经有 causal mask 了，为什么还必须显式加入位置机制？

## 直觉解释

Causal mask 只告诉模型“未来不能看”，却没告诉它“我离你有多远”。它给的是一个偏序约束，而不是完整的序列几何。

## 形式化推导

causal mask 只会把上三角位置的 logits 屏蔽掉，让第 $t$ 个 token 只能访问 $1 \ldots t$。但对允许访问的历史位置来说，mask 并不区分“前 1 步”和“前 100 步”的具体距离。

## 工程意义

这就是为什么 Decoder-only LLM 同时需要：

- causal mask：防止训练作弊。
- 位置编码：表达相对距离、方向和先后。

两者功能完全不同。

## 常见误解

> [!warning] 常见误解
> - “只要是下三角 mask，模型自然就知道顺序。” 不对。它只知道可见集，不知道距离结构。
> - “有 RoPE 就不需要 causal mask。” 也不对。位置几何不负责防止未来泄露。

## 例子或反例

两个历史 token 都对当前 token 可见时，causal mask 无法告诉模型哪个更近、哪个更远；这类信息必须由位置机制额外提供。

## 相关链接

-[[03_RoPE_ALiBi_为什么主流模型偏向这两类方案|RoPE 与 ALiBi]]]
- [[03_CrossAttentionMask如何约束条件信息流|Cross-Attention Mask 如何约束条件信息流]]
- [[02_AttentionMask_Causal_Padding_CrossAttention|Attention Mask]]

