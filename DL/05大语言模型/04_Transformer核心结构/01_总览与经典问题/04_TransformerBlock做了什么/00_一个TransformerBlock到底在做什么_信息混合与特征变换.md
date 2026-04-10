---
tags:
  - LLM/架构
aliases:
  - Transformer Block
created: 2026-03-28
updated: 2026-03-29
---

# 一个 Transformer Block 到底在做什么：信息混合与特征变换

> [!abstract] 摘要
> 一个标准 Transformer Block 不是“只做一次 attention”。它至少要解决两类不同问题：在 token 之间搬运上下文信息，在每个 token 内部重写通道表示。前者主要由 attention 完成，后者主要由 FFN 完成，残差与 Norm 则负责让这一切能稳定堆叠起来。

## 这页先讲清什么

- 为什么 Block 至少包含 attention、FFN、残差和 Norm。
- Token mixing 和 channel mixing 为什么不能互相替代。
- 为什么现代模型的很多参数其实在 FFN，而不是只在 attention。

## 关键结论

- attention 决定“从哪里取信息”，FFN 决定“拿到信息后怎么重写表示”。
- 残差保证有稳定主通路，Norm 保证这条深层主通路不会很快失控。
- 只保留 attention 而删掉 FFN，模型会更像“内容重混合器”，而不是强非线性特征变换器。

## 子页导航

- [[01_TokenMixing与ChannelMixing如何分工|Token Mixing 与 Channel Mixing 如何分工]]
- [[02_残差_Norm_FFN在Block内如何串起来|残差、Norm、FFN 在 Block 内如何串起来]]

## 最短闭环解释

每个 Block 都在做两件事。第一件事是让一个 token 去读取别的 token，于是信息在序列维度上流动，这就是 attention 的职责。第二件事是当这个 token 拿到上下文之后，要在自己的通道空间里产生更复杂的非线性重写，这就是 FFN 的职责。

这两种混合维度完全不同，所以很难互相替代。attention 擅长在位置之间路由信息，但输出本质上仍是 value 的加权组合；FFN 不看别的 token，却能在本位置上制造新的高阶特征。把两者交替堆叠，模型才同时拥有“跨 token 通信”和“单 token 深度变换”。

而残差和 Norm 的任务不是表达语义，而是保证上面这两件事能堆很多层还不炸。于是，Block 的真正结构不是“注意力 + 一个小尾巴”，而是一套分工明确、可深堆叠的信息处理单元。

## 相关链接

- [[00_缩放点积注意力_为什么是点积_为什么要除以根号dk|缩放点积注意力]]
- [[00_AddNorm_PreNorm_PostNorm_LayerNorm_RMSNorm|Norm 与残差骨架]]
- [[00_FFN_GELU_GEGLU_SwiGLU_为什么Attention后还要MLP|FFN 与 GLU]]

