# ELU 激活函数 (Exponential Linear Unit)

## 1. 定义与公式
**ELU** 旨在融合 ReLU 的线性特性和 Sigmoid/Tanh 的左侧软饱和特性，同时使输出均值接近于 0。

**公式**：
$$
f(x) = \begin{cases}
x, & x \geq 0 \\
\alpha (e^x - 1), & x < 0
\end{cases}
$$
其中 $\alpha$ 是一个超参数，通常取 $\alpha=1$。

**图像**：
![[ELU.png|600]]

## 2. 导函数

**导数公式**：
$$
f'(x) = \begin{cases}
1, & x \geq 0 \\
f(x) + \alpha, & x < 0
\end{cases}
$$
或者写为：
$$
f'(x) = \begin{cases}
1, & x \geq 0 \\
\alpha e^x, & x < 0
\end{cases}
$$

**导数图像**：
![[ELU_derivative.png|600]]

## 3. 优缺点分析

### 优点
1.  **零中心化 (Zero-Centered)**：负区间的指数形式使得输出均值更接近 0，有助于加快收敛速度。
2.  **抗噪声能力**：负区间的软饱和特性（趋向于 $-\alpha$）使得 ELU 对输入噪声更鲁棒。
3.  **缓解神经元死亡**：在负区间有非零梯度，避免了 Dead ReLU 问题。

### 缺点
1.  **计算成本**：负区间涉及指数运算 $e^x$，计算量比 ReLU 大。
2.  **梯度爆炸风险**：虽然缓解了梯度消失，但如果 $\alpha$ 设置不当或网络极深，仍需注意梯度控制（不过通常比 Sigmoid 好）。