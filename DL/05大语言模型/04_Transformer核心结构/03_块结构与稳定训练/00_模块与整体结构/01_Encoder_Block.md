---
tags:
  - LLM/Transformer
  - 模块/EncoderBlock
  - 块结构/PreNorm
aliases:
  - Encoder Block
  - 编码器块
created: 2025-01-18
updated: 2026-03-29
difficulty: 中等
---

# Encoder Block（编码器块）

> [!abstract]
> 一个 Encoder Block 做的事可以概括为两步：先用自注意力在 token 之间混合信息，再用 FFN 在每个位置内部做特征变换。

## 一层内部数据流

现代实现里最常见的是 Pre-LN 形式：

$$
H_1 = X + \text{MHA}(\text{LN}(X))
$$

$$
H_2 = H_1 + \text{FFN}(\text{LN}(H_1))
$$

对应到流程图，就是：

```mermaid
flowchart LR
    X["输入 X"] --> LN1["LayerNorm"]
    LN1 --> SA["Multi-Head Self-Attention"]
    SA --> Add1["残差相加"]
    X --> Add1
    Add1 --> LN2["LayerNorm"]
    LN2 --> FFN["FFN / MLP"]
    FFN --> Add2["残差相加"]
    Add1 --> Add2
```

## 每个子模块分别负责什么

| 子模块 | 作用 | 你可以怎样理解 |
| --- | --- | --- |
| LayerNorm | 稳定数值与梯度 | 先把输入尺度整理好，再交给子层处理 |
| Multi-Head Self-Attention | 跨 token 混合信息 | 决定“每个位置该从别的位置读什么” |
| Residual | 保留原路径、稳定训练 | 给深层网络留一条信息高速路 |
| FFN | 逐位置做非线性变换 | 在每个 token 内部加工特征 |

## 为什么 Encoder Block 是“全局编码器”

### 1. 自注意力负责 token mixing

同一层内，每个位置都可以通过注意力读取全局上下文，因此 Encoder 不只是看局部邻域，而是显式做全局交互。

### 2. FFN 负责 channel mixing

注意力擅长在 token 之间搬运信息，但它并不擅长充分变换单个位置内部的通道表达。  
FFN 的作用就是在每个位置上继续做非线性映射。

### 3. 多层堆叠形成层次化表示

- 浅层更容易学局部搭配、词法模式
- 深层更容易形成抽象语义和全局结构

这也是为什么堆叠多个 Encoder Block 后，输出可以作为高质量上下文表示供下游任务或 Decoder 使用。

## FFN 在 Encoder Block 里通常长什么样

最典型的两层 FFN 写作：

$$
\text{FFN}(x) = \phi(xW_1 + b_1)W_2 + b_2
$$

其中：

- $\phi$ 常用 GELU、SiLU 或门控变体
- 中间层宽度通常取 $d_{ff} \approx 4d_{model}$

> [!tip]
> 可以把注意力理解成“跨位置取信息”，把 FFN 理解成“在当前位置内部重新加工信息”。

## Pre-LN 和 Post-LN 怎么看

| 结构 | 写法位置 | 特点 |
| --- | --- | --- |
| Pre-LN | 先 LN，再进子层 | 深层训练更稳，现代 LLM 更常见 |
| Post-LN | 子层输出后再 LN | 原始 Transformer 常见，但深层更容易梯度不稳 |

这也是为什么现在谈 Encoder Block，如果不特别说明，默认多半是在说 Pre-LN 版本。

## 工程实现里常见的附加部件

- Attention Dropout：作用在注意力权重或子层输出上
- FFN Dropout：作用在激活后或输出投影后
- LayerDrop：按层随机跳过，提升鲁棒性
- RMSNorm：一些大模型会用它替代 LayerNorm
- 残差缩放 / DeepNorm：用于更深网络的稳定训练

## 一句话抓重点

> [!note]
> Encoder Block = `先跨 token 混合信息，再在每个 token 内部加工特征，再依靠残差和归一化把整个过程稳定下来`。

## 相关双链

- [[索引_注意力机制]]
- [[01_自注意力基础]]
- [[02_多头自注意力MHA]]
- [[02_Decoder_Block]]
- [[02_残差_Norm_FFN在Block内如何串起来|残差、Norm、FFN 在 Block 内如何串起来]]
- [[00_AddNorm_PreNorm_PostNorm_LayerNorm_RMSNorm|Add-Norm、PreNorm、PostNorm、LayerNorm、RMSNorm]]
- [[00_FFN_GELU_GEGLU_SwiGLU_为什么Attention后还要MLP|为什么 Attention 后还要 MLP]]
