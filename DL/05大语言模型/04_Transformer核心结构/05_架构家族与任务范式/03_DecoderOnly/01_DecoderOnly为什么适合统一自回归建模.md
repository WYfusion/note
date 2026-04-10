---
tags:
  - LLM/架构
created: 2026-03-29
updated: 2026-03-29
---

# Decoder-Only 为什么适合统一自回归建模

## 问题定义

为什么 Decoder-only 能把大量不同文本任务统一到一个训练目标里？

## 直觉解释

因为任何文本最终都能写成“前缀 -> 下一个 token”的问题。

## 形式化推导

Causal LM 目标天然统一，训练样本构造、推理接口和服务路径都围绕同一个自回归过程展开。

## 工程意义

这极大降低了数据处理、训练、微调和推理系统的复杂度。

## 常见误解

> [!warning] 常见误解
> - “统一接口就代表所有任务都最优。” 不对，只是最易扩展。

## 例子或反例

聊天、代码补全、续写都能自然塞进同一个 Decoder-only 框架。

## 相关链接

- [[02_GPT_LLaMA_Qwen的骨架差异与共性|GPT、LLaMA、Qwen 的骨架差异与共性]]
- [[00_DecoderOnly_GPT_LLaMA_Qwen|Decoder-Only]]
- [[00_KVCache_Prefill_Decode_PagedAttention|KV Cache]]
