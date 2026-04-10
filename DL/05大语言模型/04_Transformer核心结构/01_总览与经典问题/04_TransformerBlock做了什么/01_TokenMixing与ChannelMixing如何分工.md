---
tags:
  - LLM/架构
created: 2026-03-29
updated: 2026-03-29
---

# Token Mixing 与 Channel Mixing 如何分工

## 问题定义

为什么 attention 负责 token mixing，FFN 负责 channel mixing？这两个词真正指的是什么维度上的运算？

## 直觉解释

token mixing 关注“不同位置之间怎么交换信息”，channel mixing 关注“同一位置内部的特征怎么重写和重组”。

## 形式化推导

对隐藏状态 $X \in \mathbb{R}^{L \times D}$：

- attention 在序列维度上构造权重矩阵 $A \in \mathbb{R}^{L \times L}$，输出近似为 $AX$ 的变体，因此核心是位置之间的混合。
- FFN 对每个位置独立应用同一组 MLP：

$$
\operatorname{FFN}(x_i) = W_2 \sigma(W_1 x_i)
$$

因此它不在 token 之间通信，而是在通道维度内做非线性投影。

## 工程意义

如果只有 token mixing，没有 channel mixing，模型更像在搬运和重排上下文；如果只有 channel mixing，没有 token mixing，每个 token 又像孤岛。Transformer 的表达能力正来自两者交替堆叠。

## 常见误解

> [!warning] 常见误解
> - “attention 已经足够强，所以 FFN 只是装饰。” 错。很多参数和非线性能力都在 FFN。
> - “FFN 不看别的 token，所以不重要。” 不对。深层表示重写大多在这里发生。

## 例子或反例

在代词指代里，attention 先把“它”与前面的实体连起来；拿到这些上下文后，FFN 再把“它”的内部表示改写成更接近“指代书本”的状态。

## 相关链接

- [[02_残差_Norm_FFN在Block内如何串起来|残差、Norm、FFN 在 Block 内如何串起来]]
- [[00_FFN_GELU_GEGLU_SwiGLU_为什么Attention后还要MLP|FFN 与 GLU]]
- [[00_一个TransformerBlock到底在做什么_信息混合与特征变换|一个 Transformer Block 在做什么]]

