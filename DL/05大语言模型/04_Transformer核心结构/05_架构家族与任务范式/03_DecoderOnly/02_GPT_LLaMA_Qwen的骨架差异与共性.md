---
tags:
  - LLM/架构
created: 2026-03-29
updated: 2026-03-29
---

# GPT、LLaMA、Qwen 的骨架差异与共性

## 问题定义

这些主流 Decoder-only 模型真正共享什么，又各自在骨架上改了什么？

## 直觉解释

共性是 causal decoder 主干，差异主要落在位置编码、Norm、FFN、KV 优化和多模态扩展上。

## 形式化推导

GPT 奠定主干，LLaMA 常见 RoPE、RMSNorm、SwiGLU、GQA 趋势，Qwen 则在此基础上扩展更丰富的任务与模态支持。

## 工程意义

现代 LLM 的“创新”很多不是换大骨架，而是在 Decoder-only 主干上不断优化局部结构与系统表现。

## 常见误解

> [!warning] 常见误解
> - “它们只是参数大小不同。” 不对，骨架细节和训练配方差异也会显著影响表现。

## 例子或反例

同样是 Decoder-only，GQA 与否、RoPE 变体与否，都会影响长上下文与推理成本。

## 相关链接

- [[01_DecoderOnly为什么适合统一自回归建模|Decoder-Only 为什么适合统一自回归建模]]
- [[03_DecoderOnly_GPT_LLaMA_Qwen|Decoder-Only]]
- [[03_多头注意力为什么有效_MHA_MQA_GQA_MLA|多头注意力]]
