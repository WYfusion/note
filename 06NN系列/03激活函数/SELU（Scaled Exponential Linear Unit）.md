# SELU 激活函数 (Scaled ELU)

## 1. 定义与公式
**SELU** 是 ELU 的缩放版本，专门为 **SNN (Self-Normalizing Neural Networks)** 设计。它通过特定的缩放因子 $\lambda$ 和 $\alpha$，使得网络在深层传播时能够自动保持输出的均值为 0，方差为 1（自归一化）。

**公式**：
$$
f(x) = \lambda \begin{cases}
x, & x \geq 0 \\
\alpha (e^x - 1), & x < 0
\end{cases}
$$
**固定参数**（经过数学推导得出）：
-   $\lambda \approx 1.0507$
-   $\alpha \approx 1.67326$

**图像**：
![[SELU.png|600]]

## 2. 导函数

**导数公式**：
$$
f'(x) = \begin{cases}
\lambda, & x \geq 0 \\
\lambda \alpha e^x, & x < 0
\end{cases}
$$

**导数图像**：
![[SELU_derivative.png|600]]

## 3. 优缺点分析

### 优点
1.  **自归一化 (Self-Normalizing)**：这是 SELU 最大的特点。在全连接层中，如果输入服从标准正态分布，经过 SELU 后输出仍近似服从标准正态分布。这使得深层网络可以不使用 Batch Normalization (BN) 也能稳定训练。
2.  **避免梯度消失/爆炸**：由于自归一化属性，梯度在传播过程中更稳定。

### 缺点
1.  **适用范围限制**：自归一化性质主要在**全连接层 (Dense Layers)** 中严格成立。在卷积神经网络 (CNN) 或包含 Skip Connection 的网络中，效果可能不如 ReLU + BN。
2.  **参数固定**：$\lambda$ 和 $\alpha$ 是固定的，不能随意更改，否则会破坏自归一化属性。
3.  **计算成本**：涉及指数运算。