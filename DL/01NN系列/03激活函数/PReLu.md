# PReLU 激活函数 (Parametric ReLU)

## 1. 定义与公式
**PReLU (Parametric Rectified Linear Unit)** 是 Leaky ReLU 的改进版。它将负区间的斜率 $\alpha$ 作为一个**可学习的参数**，而不是固定的超参数。

**公式**：
$$
f(x) = \begin{cases}
x, & x \geq 0 \\
\alpha x, & x < 0
\end{cases}
$$
其中 $\alpha$ 是通过反向传播学习得到的参数。
![[Pasted image 20251230105150.png|800]]
## 2. 导函数

**对输入 $x$ 的导数**：
$$
\frac{\partial f(x)}{\partial x} = \begin{cases}
1, & x \geq 0 \\
\alpha, & x < 0
\end{cases}
$$

**对参数 $\alpha$ 的导数**（用于更新 $\alpha$）：
$$
\frac{\partial f(x)}{\partial \alpha} = \begin{cases}
0, & x \geq 0 \\
x, & x < 0
\end{cases}
$$
这意味着只有当输入为负时，$\alpha$ 才会得到更新。
![[Pasted image 20251230105200.png|800]]

## 3. 优缺点分析

### 优点
1.  **自适应性**：模型可以根据数据自动学习最佳的负区间斜率，避免了人工选择 $\alpha$ 的盲目性。
2.  **性能提升**：在 ImageNet 等大型数据集上，PReLU 证明了比 ReLU 和 Leaky ReLU 有更好的性能（何凯明等，2015）。
3.  **计算代价小**：增加的参数量和计算量非常小（每个通道仅增加一个参数）。

### 缺点
1.  **过拟合风险**：由于引入了额外的参数，在小数据集上可能会增加过拟合的风险（尽管风险较小）。
2.  **实现稍复杂**：相比无参的 ReLU，需要维护参数 $\alpha$ 的更新。

## 4. 变种
-   **Channel-wise PReLU**：每个通道共享一个 $\alpha$（最常用）。
-   **Element-wise PReLU**：每个神经元有一个独立的 $\alpha$。

