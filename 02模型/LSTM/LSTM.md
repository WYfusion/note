---
tags:
  - 机器学习
  - 深度学习
  - LSTM
  - 序列模型
  - 门控机制
created: 2025-01-18
modified: 2025-01-18
difficulty: 中高
related:
  - [[RNN]]
  - [[GRU(门控循环单元)]]
  - [[BPTT（Backpropagation Through Time，随时间反向传播）]]
  - [[Transformer]]
---

> [!summary] 核心思想
> LSTM（长短期记忆网络）通过引入记忆单元和门控机制，解决了传统 RNN 的梯度消失问题，能够有效捕获序列中的长距离依赖关系。

# LSTM（长短期记忆网络）

## 概述

长短期记忆网络的设计灵感来自于计算机的逻辑门。LSTM 引入了记忆元（memory cell），或简称为单元（cell），通过精心设计的门控机制控制信息的流动，从而实现长期记忆功能。

### RNN 的基本结构

普通 RNN 的结构非常简单：
- 一个单一的神经网络层
- 一个简单的 tanh 或 sigmoid 激活函数
- 一个状态传递机制

![[循环神经网络的结构.excalidraw|600]]

### LSTM 的改进结构

LSTM 在 RNN 的基础上增加了以下关键结构：

#### 1. 记忆单元（Memory Cell）

这是 LSTM 最重要的创新，相当于一条"高速公路"：
- 信息可以在其中长期保存
- 解决了 RNN 中的梯度消失问题

![lstm-0.svg](https://zh.d2l.ai/_images/lstm-1.svg)

#### 2. 三个门控机制

假设有 $h$ 个隐藏单元，批量大小为 $n$，输入数为 $d$。

- 输入为 $\mathbf{X}_t \in \mathbb{R}^{n \times d}$
- 前一时间步的隐状态为 $\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$

相应地，时间步 $t$ 的门被定义如下：

- **输入门**：$\mathbf{I}_{t} \in \mathbb{R}^{n \times h}$
- **遗忘门**：$\mathbf{F}_t \in \mathbb{R}^{n \times h}$
- **输出门**：$\mathbf{O}_t \in \mathbb{R}^{n \times h}$
- **候选记忆元**：$\tilde{\mathbf{C}}_t \in \mathbb{R}^{n \times h}$
- **记忆元**：$\mathbf{C}_{t-1} \in \mathbb{R}^{n \times h}$
- **隐状态**：$\mathbf{H}_t \in \mathbb{R}^{n \times h}$

### 数学公式

$$
\begin{aligned}
\mathbf{I}_{t} &= \sigma(\mathbf{X}_t\mathbf{W}_{xi} + \mathbf{H}_{t-1}\mathbf{W}_{hi} + \mathbf{b}_i), \\
\mathbf{F}_t &= \sigma(\mathbf{X}_t\mathbf{W}_{xf} + \mathbf{H}_{t-1}\mathbf{W}_{hf} + \mathbf{b}_f), \\
\mathbf{O}_t &= \sigma(\mathbf{X}_t\mathbf{W}_{xo} + \mathbf{H}_{t-1}\mathbf{W}_{ho} + \mathbf{b}_o), \\
\tilde{\mathbf{C}}_t &= \tanh(\mathbf{X}_t\mathbf{W}_{xc} + \mathbf{H}_{t-1}\mathbf{W}_{hc} + \mathbf{b}_c), \\
\mathbf{C}_t &= \mathbf{F}_t \odot \mathbf{C}_{t-1} + \mathbf{I}_t \odot \tilde{\mathbf{C}}_t, \\
\mathbf{H}_t &= \mathbf{O}_t \odot \tanh(\mathbf{C}_t)
\end{aligned}
$$

其中：
- $\mathbf{W}_{xi}, \mathbf{W}_{xf}, \mathbf{W}_{xo}, \mathbf{W}_{xc} \in \mathbb{R}^{d \times h}$ 是输入权重矩阵
- $\mathbf{W}_{hi}, \mathbf{W}_{hf}, \mathbf{W}_{ho}, \mathbf{W}_{hc} \in \mathbb{R}^{h \times h}$ 是隐层权重矩阵
- $\mathbf{b}_i, \mathbf{b}_f, \mathbf{b}_o, \mathbf{b}_c \in \mathbb{R}^{1 \times h}$ 是偏置参数
- $\odot$ 表示逐元素乘积（Hadamard 积）

### 权重矩阵说明

- **输入门权重**：$\mathbf{W}_{xi}, \mathbf{W}_{hi}$
- **遗忘门权重**：$\mathbf{W}_{xf}, \mathbf{W}_{hf}$
- **输出门权重**：$\mathbf{W}_{xo}, \mathbf{W}_{ho}$
- **候选记忆元权重**：$\mathbf{W}_{xc}, \mathbf{W}_{hc}$

![lstm-1.svg](https://zh.d2l.ai/_images/lstm-3.svg)

> [!question] 是否存在"输出权重矩阵"？
>
> **严格来说**，LSTM 没有独立的输出权重矩阵（例如，类似 RNN 中 $\mathbf{W}_{hh}$ 或 $\mathbf{W}_{xh}$ 的显式输出权重）。
>
> **实际实现中**：若需将输出 $\mathbf{H}_t$ 映射到任务特定维度（如分类任务的类别数），通常会**添加额外的全连接层**（即输出权重矩阵）。例如：
> $$ \mathbf{Y}_t = \mathbf{H}_t \mathbf{W}_{hy} + \mathbf{b}_y $$
> 其中 $\mathbf{W}_{hy} \in \mathbb{R}^{h \times m}$ 是任务相关的输出权重矩阵（$m$ 为输出维度）。

---

## 核心特性

### 1. 梯度消失问题的解决

当 **遗忘门始终为 1** 且 **输入门始终为 0** 时，过去的记忆元 $C_{t-1}$ 将随时间被保存并传递到当前时间步。

$$
\mathbf{C}_t = \mathbf{F}_t \odot \mathbf{C}_{t-1} + \mathbf{I}_t \odot \tilde{\mathbf{C}}_t = 1 \cdot \mathbf{C}_{t-1} + 0 \cdot \tilde{\mathbf{C}}_t = \mathbf{C}_{t-1}
$$

这将使得记忆单元梯度可以无损地传递到任意远处的时间步，**彻底避免了梯度消失**：

$$
\frac{\partial \mathbf{C}_t}{\partial \mathbf{C}_k} = \prod_{j=k+1}^{t} \frac{\partial \mathbf{C}_{j}}{\partial \mathbf{C}_{j-1}} = 1
$$

遗忘门和输入门的设计使得模型可以在一定程度上缓解梯度消失。

### 2. 信息流控制

#### 输出门的作用

- **输出门接近 1**：能够有效地将所有记忆信息传递给预测部分
- **输出门接近 0**：只保留记忆元内的所有信息，而不需要更新隐状态

#### 梯度解耦

输出门将隐藏状态的梯度与细胞记忆状态的传播路径分离，提升稳定性。

### 3. 梯度爆炸问题

> [!warning] 注意事项
> LSTM **不能完全解决梯度爆炸**，但相比传统 RNN，其爆炸风险更低。梯度爆炸的发生取决于权重矩阵的谱范数（最大奇异值）。

---

## LSTM 与 RNN 的对比

| **特性** | **RNN** | **LSTM** |
|---------|---------|----------|
| **长期依赖** | 随着序列长度增加，早期信息会逐渐丢失 | 通过记忆单元可以长期保存重要信息 |
| **梯度问题** | 存在严重的梯度消失/爆炸问题 | 记忆单元提供了梯度的"快速通道"，大大缓解了这个问题 |
| **信息流控制** | 信息强制性全部通过 | 可以选择性地保留或遗忘信息 |
| **状态分离** | 只有一个混合状态 | 分离了记忆单元（$C_t$）和隐藏状态（$h_t$），使信息处理更有针对性 |

---

## 相关链接

- [[RNN]] - 循环神经网络基础
- [[GRU(门控循环单元)]] - 简化的 LSTM 变体
- [[BPTT（Backpropagation Through Time，随时间反向传播）]] - 训练算法
- [[Transformer]] - 现代替代架构

## 参考资料

- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780
- D2L.AI: [LSTM](https://zh.d2l.ai/chapter_recurrent-modern/lstm.html)
