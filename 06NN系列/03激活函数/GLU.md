# GLU 激活函数 (Gated Linear Unit)

## 1. 定义与公式
**GLU (Gated Linear Unit)** 是一种引入了**门控机制 (Gating Mechanism)** 的神经网络层，最早由 Dauphin 等人在 2017 年的论文《Language Modeling with Gated Convolutional Networks》中提出。

它的核心思想是：输入被分为两部分，一部分经过 Sigmoid 激活函数变成 $(0, 1)$ 之间的“门控系数”，用来控制另一部分（线性变换后）的信息通过量。

**公式**：
$$ \text{GLU}(x) = \sigma(xW + b) \otimes (xV + c) $$
其中：
-   $x$ 是输入向量。
-   $W, V$ 是两个独立的权重矩阵。
-   $b, c$ 是偏置项。
-   $\sigma$ 是 Sigmoid 激活函数。
-   $\otimes$ 表示逐元素乘法 (Element-wise product / Hadamard product)。

简单来说，GLU 学习决定哪些信息应该保留（门控值为 1），哪些应该被抑制（门控值为 0）。

## 2. 导函数
令 $g = xW+b$ (门控路径), $l = xV+c$ (线性路径)。
输出 $y = \sigma(g) \odot l$。

对输入 $x$ 的梯度涉及两部分：
1.  **通过线性路径 $l$ 的梯度**：受门控 $\sigma(g)$ 的缩放。
    $$ \frac{\partial y}{\partial l} = \sigma(g) $$
    这表明如果门控打开（接近 1），梯度可以无损地通过线性路径反向传播，这有助于缓解梯度消失问题。
2.  **通过门控路径 $g$ 的梯度**：
    $$ \frac{\partial y}{\partial g} = \sigma'(g) \odot l = \sigma(g)(1-\sigma(g)) \odot l $$

## 3. 优缺点分析

### 优点
1.  **缓解梯度消失**：线性路径 $xV+c$ 提供了一个让梯度流动的“高速公路”，类似于 LSTM 中的线性单元，使得训练深层网络更加容易。
2.  **选择性特征提取**：模型可以学习动态地选择通过哪些特征，这在自然语言处理（NLP）等序列建模任务中非常有效。

### 缺点
1.  **参数量加倍**：相比于普通的线性层 + 激活函数，GLU 需要两个权重矩阵 ($W$ 和 $V$)，导致参数量和计算量增加了一倍。
