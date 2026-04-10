---
tags:
  - LLM/架构
created: 2026-03-29
updated: 2026-03-29
---

# Padding Mask 为什么不是小细节

## 问题定义

Padding mask 看起来只是把补齐位置屏蔽掉，为什么它在训练和推理里都很关键？

## 直觉解释

padding 不是“弱信息”，而是“无信息”。如果不屏蔽，模型会把这些占位符当成真实上下文，从而污染注意力分布和损失统计。

## 形式化推导

对无效位置加入一个极小值 mask，相当于把该位置的 logits 推到负无穷附近，使 softmax 后权重近似为 0。这样 Value 聚合时，无效 token 不参与任何有效贡献。

## 工程意义

Padding mask 关系到：

- 训练时变长 batch 的正确性。
- 多模态输入中无效帧、空 patch 的处理。
- 推理批服务里不同请求长度混排的稳定性。

## 常见误解

> [!warning] 常见误解
> - “padding 只影响一点点噪声。” 不对。长序列、大 batch 下影响会累积。
> - “推理时只有一个请求就不需要 padding mask。” 不一定。多模态、批处理和缓存场景仍可能需要。

## 例子或反例

两个句子被补齐到同一长度时，若不加 padding mask，短句尾部的一长串 pad 位置可能被模型错误读取，尤其会影响池化表示和分类头。

## 相关链接

- [[02_CausalMask为什么不等于位置编码|Causal Mask 为什么不等于位置编码]]
- [[03_CrossAttentionMask如何约束条件信息流|Cross-Attention Mask 如何约束条件信息流]]
- [[00_AttentionMask_Causal_Padding_CrossAttention|Attention Mask]]

