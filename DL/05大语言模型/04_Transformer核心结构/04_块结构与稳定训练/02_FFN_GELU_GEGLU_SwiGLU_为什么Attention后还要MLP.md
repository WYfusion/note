---
tags:
  - LLM/训练
  - SwiGLU
created: 2026-03-28
updated: 2026-03-29
---

# FFN、GELU、GEGLU、SwiGLU：为什么 Attention 后还要 MLP

> [!abstract] 摘要
> attention 解决的是 token 间信息路由，FFN 解决的是每个 token 内部的特征重写。现代模型继续在 FFN 上投入大量参数，并转向门控变体，是因为这里承载了非常重要的表达容量。

## 这页先讲清什么

- 为什么 attention 后必须接 FFN。
- GELU、GEGLU、SwiGLU 分别在什么地方增强表达。
- 为什么 FFN 的中间维度通常远大于主隐藏维。

## 关键结论

- 没有 FFN，模型更像内容重混合器，而不是强非线性表征器。
- 门控 FFN 往往能在相近预算下给出更高表达效率。
- FFN 维度扩张不是浪费，而是在给单 token 表示更大的非线性工作空间。

## 子页导航

- [[01_为什么Attention后还必须有FFN|为什么 Attention 后还必须有 FFN]]
- [[02_GELU_GEGLU_SwiGLU的表达差异|GELU、GEGLU、SwiGLU 的表达差异]]
- [[03_FFN维度扩张为什么常设为4倍左右|FFN 维度扩张为什么常设为 4 倍左右]]

## 最短闭环解释

attention 把别的位置的信息搬过来，但输出本质上还是已有 value 的加权组合。若想让每个 token 在拿到上下文后发生更复杂的重写，就需要 FFN。

经典 FFN 是两层线性加一个非线性激活。现代门控变体如 GEGLU、SwiGLU 的思路是：不是所有通道都应一视同仁通过，模型需要细粒度“开门/关门”的能力。

至于中间维度为什么常做大，是因为模型需要一个更宽的中间空间来制造更复杂的局部特征，再压回主维度。这部分容量在大模型里非常值钱。

## 相关链接

- [[01_AddNorm_PreNorm_PostNorm_LayerNorm_RMSNorm|Norm 与残差骨架]]
- [[03_残差路径_初始化_DeepNorm|残差路径与 DeepNorm]]
- [[04_一个TransformerBlock到底在做什么_信息混合与特征变换|一个 Transformer Block 在做什么]]
