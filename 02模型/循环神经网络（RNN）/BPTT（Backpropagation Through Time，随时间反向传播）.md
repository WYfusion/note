---
tags:
  - 机器学习
  - 深度学习
  - BPTT
  - RNN训练
  - 梯度传播
created: 2025-01-18
modified: 2025-01-18
difficulty: 中高
related:
  - [[RNN]]
  - [[LSTM]]
  - [[GRU(门控循环单元)]]
---

> [!summary] 核心思想
> BPTT（Backpropagation Through Time，随时间反向传播）是专门用于训练循环神经网络（RNN）的梯度计算算法，通过将 RNN 在时间轴上展开为前馈网络，实现时序依赖的梯度反向传播。

# BPTT（Backpropagation Through Time，随时间反向传播）

## 概述

BPTT 是传统反向传播（BP）算法在时序数据上的扩展，专门用于训练循环神经网络。其核心思想是将 RNN 在时间轴上展开为前馈网络，通过反向传播计算梯度。BPTT 本质上是 BP 在时序数据上的扩展，但需处理时间依赖性和序列长度带来的计算挑战。

---

## 1. 算法原理

### 1.1 前向传播阶段

RNN 在每个时间步 $t$ 的隐藏状态 $h_t$ 和输出 $y_t$ 由以下公式决定：

$$
h_t = g(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b_h)
$$

其中 $W_{hh}, W_{xh}, W = [W_{hh}, W_{xh}]$ 为权重矩阵，$b_h$ 为偏置项。

#### 时间展开

将 RNN 视为一个展开的前馈网络，每个时间步对应一个"虚拟层"：

![[循环神经网络的结构.excalidraw|80%]]

展开后的网络结构：

$$
\begin{cases}
h_1 = g(U \cdot x_1 + W \cdot h_0) \\
\hat{y}_1 = g(V \cdot h_1)
\end{cases}
\quad
\begin{cases}
h_2 = g(U \cdot x_2 + W \cdot h_1) \\
\hat{y}_2 = g(V \cdot h_2)
\end{cases}
\quad
\begin{cases}
h_3 = g(U \cdot x_3 + W \cdot h_2) \\
\hat{y}_3 = g(V \cdot h_3)
\end{cases}
$$

> [!note] 重要特性
> 所有单独的时刻下，各个节点共享权重矩阵（$U$、$W$、$V$）。隐层 $h$ 的输入含有之前时刻所有隐层，这将直接影响后面计算权重梯度时的链式求导过程。

### 1.2 损失计算

#### 总损失函数

总损失 $L$ 为各时间步损失 $L_t$ 的累加（如交叉熵、均方误差）：

$$
L = \sum_{t=1}^{T} L_t(y_t, \hat{y}_t)
$$

其中 $L_t$ 是损失函数，$\hat{y}_t$ 是期望输出。

### 1.3 反向传播阶段

从时间步 $T$ 到 $1$ **逆序**计算梯度，传播路径包括：

#### 1. 输出层梯度

$$
\frac{\partial L}{\partial y_t}
$$

#### 2. 隐藏层梯度

当前时间步的梯度来自两部分：

$$
\frac{\partial L}{\partial h_t} = \frac{\partial L_t}{\partial h_t} + \frac{\partial L}{\partial h_{t+1}} \cdot \frac{\partial h_{t+1}}{\partial h_t}
$$

- **第一项**：当前步输出损失对 $h_t$ 的直接影响（通过 $y_t$）
- **第二项**：未来时间步的梯度通过 $h_{t+1}$ 反向传播到 $h_t$，体现时间依赖性

---

## 2. 梯度计算详细推导

### 2.1 单时间步梯度

对于 $t=3$ 时刻，损失函数 $L_3$ 对三个权重矩阵的偏导：

#### 1. 输出权重 $V$

$$
\frac{\partial L_3}{\partial V} = \frac{\partial L_3}{\partial \hat{y}_3} \times \frac{\partial \hat{y}_3}{\partial V}
$$

#### 2. 隐层权重 $W$

$$
\begin{aligned}
\frac{\partial L_3}{\partial W} = &\frac{\partial L_3}{\partial \hat{y}_3} \times \frac{\partial \hat{y}_3}{\partial h_3} \times \frac{\partial h_3}{\partial W} \\
&+ \frac{\partial L_3}{\partial \hat{y}_3} \times \frac{\partial \hat{y}_3}{\partial h_3} \times \frac{\partial h_3}{\partial h_2} \times \frac{\partial h_2}{\partial W} \\
&+ \frac{\partial L_3}{\partial \hat{y}_3} \times \frac{\partial \hat{y}_3}{\partial h_3} \times \frac{\partial h_3}{\partial h_2} \times \frac{\partial h_2}{\partial h_1} \times \frac{\partial h_1}{\partial W}
\end{aligned}
$$

#### 3. 输入权重 $U$

$$
\begin{aligned}
\frac{\partial L_3}{\partial U} = &\frac{\partial L_3}{\partial \hat{y}_3} \times \frac{\partial \hat{y}_3}{\partial h_3} \times \frac{\partial h_3}{\partial U} \\
&+ \frac{\partial L_3}{\partial \hat{y}_3} \times \frac{\partial \hat{y}_3}{\partial h_3} \times \frac{\partial h_3}{\partial h_2} \times \frac{\partial h_2}{\partial U} \\
&+ \frac{\partial L_3}{\partial \hat{y}_3} \times \frac{\partial \hat{y}_3}{\partial h_3} \times \frac{\partial h_3}{\partial h_2} \times \frac{\partial h_2}{\partial h_1} \times \frac{\partial h_1}{\partial U}
\end{aligned}
$$

### 2.2 任意时间步梯度

综合有任意 $t$ 时刻，损失函数 $L_t$ 对三个权重矩阵的偏导：

#### 1. 输出权重 $V$

$$
\frac{\partial L_t}{\partial V} = \frac{\partial L_t}{\partial \hat{y}_t} \times \frac{\partial \hat{y}_t}{\partial V}
$$

#### 2. 隐层权重 $W$

$$
\frac{\partial L_t}{\partial W} = \sum_{k=1}^{t} \frac{\partial L_t}{\partial \hat{y}_t} \times \frac{\partial \hat{y}_t}{\partial h_t} \times \frac{\partial h_t}{\partial h_k} \times \frac{\partial h_{k}}{\partial W}
$$

其中隐层状态的梯度累积：

$$
\frac{\partial h_t}{\partial h_k} = \frac{\partial h_t}{\partial h_{t-1}} \times \frac{\partial h_{t-1}}{\partial h_{t-2}} \times \cdots \times \frac{\partial h_{k+1}}{\partial h_{k}} = \prod_{j=k+1}^{t} \frac{\partial h_{j}}{\partial h_{j-1}}
$$

因此式（2）可写为：

$$
\frac{\partial L_t}{\partial W} = \sum_{k=1}^{t} \frac{\partial L_t}{\partial \hat{y}_t} \times \frac{\partial \hat{y}_t}{\partial h_t} \times \left(\prod_{j=k+1}^{t} \frac{\partial h_{j}}{\partial h_{j-1}}\right) \times \frac{\partial h_{k}}{\partial W}
$$

#### 3. 输入权重 $U$

$$
\frac{\partial L_t}{\partial U} = \sum_{k=1}^{t} \frac{\partial L_t}{\partial y_t} \times \frac{\partial \hat{y}_t}{\partial h_t} \times \left(\prod_{j=k+1}^{t} \frac{\partial h_{j}}{\partial h_{j-1}}\right) \times \frac{\partial h_{k}}{\partial U}
$$

### 2.3 总损失梯度

计算总损失函数并结合以上各式可得总的损失函数 $L$ 对于三个权重矩阵的偏导：

$$
\begin{aligned}
\frac{\partial L}{\partial V} &= \sum_{i=1}^{T} \frac{\partial L_t}{\partial V} = \sum_{i=1}^{T} \frac{\partial L_t}{\partial \hat{y}_t} \times \frac{\partial \hat{y}_t}{\partial V} \\
\frac{\partial L}{\partial W} &= \sum_{i=1}^{T} \frac{\partial L_t}{\partial W} = \sum_{i=1}^{T} \sum_{k=1}^{t} \frac{\partial L_t}{\partial \hat{y}_t} \times \frac{\partial \hat{y}_t}{\partial h_t} \times \left(\prod_{j=k+1}^{t} \frac{\partial h_{j}}{\partial h_{j-1}}\right) \times \frac{\partial h_{k}}{\partial W} \\
\frac{\partial L}{\partial U} &= \sum_{i=1}^{T} \frac{\partial L_t}{\partial U} = \sum_{i=1}^{T} \sum_{k=1}^{t} \frac{\partial L_t}{\partial y_t} \times \frac{\partial \hat{y}_t}{\partial h_t} \times \left(\prod_{j=k+1}^{t} \frac{\partial h_{j}}{\partial h_{j-1}}\right) \times \frac{\partial h_{k}}{\partial U}
\end{aligned}
$$

---

## 3. 梯度消失与梯度爆炸问题

### 3.1 累乘部分的关键问题

由于隐层 $h_t$ 的更新公式：

$$
h_t = g(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b_h)
$$

对 $\frac{\partial h_{j}}{\partial h_{j-1}}$ 进行求导运算时：
1. 先对非线性激活函数 $g$ 求导
2. 再对内部 $h_{t-1}$ 求导

这将暴露出权重 $W$ 或 $U$ 的累乘：

$$
\frac{\partial h_{j}}{\partial h_{j-1}} = g'(a_j) \cdot W_{hh}
$$

### 3.2 问题分析

若 $g' \times W$ 或 $g' \times U$ 这一指数级运算的底数：
- **大于 1**：随着时间序列长度 $T$ 的变长会导致**梯度爆炸**
- **小于 1**：随着时间序列长度 $T$ 的变长会导致**梯度消失**

$$
\frac{\partial L}{\partial h_0} \propto \prod_{t=1}^{T} \frac{\partial h_t}{\partial h_{t-1}} = \prod_{t=1}^{T} g'(a_t) \cdot W_{hh}
$$

---

## 4. 算法变体与优化

### 4.1 截断 BPTT（Truncated BPTT）

#### 动机
长序列训练时内存和计算成本过高。

#### 方法
将长序列分割为多个子序列（如长度 $k$），在子序列内部执行 BPTT，忽略跨子序列的梯度传播。

![截断BPTT](https://pfst.cf2.poecdn.net/base/image/beaf31990ae0e4e4543ebd74a0c17f878f43600e1d210b5258d2088fb5d9e2c4?pmaid=312498326)

#### 影响
牺牲部分长期依赖建模能力，但显著降低计算复杂度。

### 4.2 梯度裁剪（Gradient Clipping）

对梯度进行逐元素或全局范数限制，防止梯度爆炸：

$$
g \leftarrow \min\left(1, \frac{\theta}{\|g\|_2}\right) g
$$

其中 $\theta$ 是裁剪阈值，$g$ 是梯度。

### 4.3 结构改进

#### LSTM/GRU
通过门控机制（遗忘门、输入门）选择性传递梯度，缓解梯度消失：

- **LSTM**：记忆单元 $C_t$ 提供了梯度的"高速公路"
- **GRU**：更新门和重置门控制信息流动

---

## 5. 学习资源

[BPTT 学习视频](https://www.bilibili.com/video/BV1fF411P72y/?spm_id_from=333.337.search-card.all.click&vd_source=1574bd9421ca96a2f458f57838315ec6)

---

## 总结

BPTT 是 RNN 训练的基石算法，通过时间展开和反向传播实现时序依赖建模。但其计算复杂度和梯度问题促使了 LSTM、Transformer 等结构的演进。理解 BPTT 的机制有助于设计更高效的序列模型和优化策略。

---

## 相关链接

- [[RNN]] - 循环神经网络基础
- [[LSTM]] - 长短期记忆网络
- [[GRU(门控循环单元)]] - 门控循环单元
- [[Transformer]] - 现代序列建模架构

## 参考资料

- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press
