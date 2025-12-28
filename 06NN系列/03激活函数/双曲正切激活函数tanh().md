# Tanh 激活函数 (Hyperbolic Tangent)

## 1. 定义与公式
Tanh 函数是双曲正切函数，它是 Sigmoid 函数的缩放和平移版本，将输入映射到 $(-1, 1)$ 区间。

**公式**：
$$ \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $$

**与 Sigmoid 的关系**：
$$ \tanh(x) = 2\sigma(2x) - 1 $$

**图像**：
![[Tanh.png|500]]

## 2. 导函数

**导数公式**：
$$ \frac{d}{dx}\tanh(x) = 1 - \tanh^2(x) $$

**推导**：
$$
\begin{aligned}
(\tanh x)' &= \left( \frac{\sinh x}{\cosh x} \right)' \\
&= \frac{(\sinh x)'\cosh x - \sinh x(\cosh x)'}{\cosh^2 x} \\
&= \frac{\cosh^2 x - \sinh^2 x}{\cosh^2 x} \\
&= 1 - \frac{\sinh^2 x}{\cosh^2 x} \\
&= 1 - \tanh^2 x
\end{aligned}
$$

**导数图像**：
![[Tanh_derivative.png|500]]

**导数性质**：
- 导数范围：$(0, 1]$。
- 当 $x=0$ 时，导数取得最大值 $1$。
- 当 $|x|$ 很大时，导数趋近于 0。

## 3. 优缺点分析

### 优点
1.  **零中心化 (Zero-Centered)**：输出均值为 0，解决了 Sigmoid 函数输出恒正导致的收敛震荡问题，通常比 Sigmoid 收敛更快。
2.  **梯度更强**：在原点附近，Tanh 的导数为 1，而 Sigmoid 仅为 0.25，这使得 Tanh 在训练初期梯度更强。

### 缺点
1.  **梯度消失**：与 Sigmoid 类似，当输入很大或很小时，导数趋近于 0，仍然存在梯度消失问题。
2.  **计算成本**：涉及幂运算，计算开销较大。