---
tags:
  - LLM/训练
created: 2026-03-29
updated: 2026-03-29
---

# 为什么 Attention 后还必须有 FFN

## 问题定义

如果 attention 已经完成上下文交互，为什么还必须保留一个大 FFN？

## 直觉解释

attention 像在调度信息，FFN 像在加工信息。只有调度，没有加工，表达深度就不够。

## 形式化推导

attention 输出仍是 value 的加权组合；FFN 通过逐 token 的非线性映射，在通道维度制造新特征。

## 工程意义

FFN 承担了大量参数和容量，是大模型表达力的重要来源。

## 常见误解

> [!warning] 常见误解
> - “删掉 FFN 只是小幅降级。” 不对，模型能力会明显缩水。

## 例子或反例

许多轻量化研究会压缩 FFN，但很少完全删除 FFN，因为 attention 很难完全取代这部分非线性重写。

## 相关链接

- [[02_GELU_GEGLU_SwiGLU的表达差异|GELU、GEGLU、SwiGLU 的表达差异]]
- [[03_FFN维度扩张为什么常设为4倍左右|FFN 维度扩张为什么常设为 4 倍左右]]
- [[00_FFN_GELU_GEGLU_SwiGLU_为什么Attention后还要MLP|FFN 与 GLU]]
