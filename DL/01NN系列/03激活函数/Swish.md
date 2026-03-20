# Swish 激活函数

## 1. 定义与公式
**Swish** 是 Google Brain 团队通过自动搜索技术发现的自门控激活函数。它在深层模型中通常优于 ReLU。

**公式**：
$$ \text{Swish}(x) = x \cdot \sigma(\beta x) = \frac{x}{1 + e^{-\beta x}} $$
其中 $\sigma$ 是 [[Sigmoid]] 函数，$\beta$ 是一个可学习参数或固定常数。
当 $\beta=1$ 时，Swish 变为 **SiLU (Sigmoid Linear Unit)**。

**图像** ($\beta=1$)：
![[Swish.png|600]]

## 2. 导函数

**导数公式**：
$$ f'(x) = \beta \cdot f(x) + \sigma(\beta x)(1 - \beta \cdot f(x)) $$
当 $\beta=1$ 时：
$$ f'(x) = f(x) + \sigma(x)(1 - f(x)) $$

**导数图像**：
![[Swish_derivative.png|600]]

## 3. 优缺点分析

### 优点
1.  **无上界 (Unbounded above)**：避免了 Sigmoid/Tanh 在正区间的梯度饱和问题。
2.  **有下界 (Bounded below)**：负区间趋向于 0（但不是直接截断），有助于正则化。
3.  **平滑非单调 (Smooth & Non-monotonic)**：这是 Swish 最显著的特点。在 $x < 0$ 的某个区域，函数值先下降后上升。这种非单调性被认为有助于梯度的流动和表达能力的提升。
4.  **性能优越**：在 MobileNet, ResNet 等模型上，替换 ReLU 后通常能获得准确率提升。

### 缺点
1.  **计算成本**：涉及 Sigmoid 运算，比 ReLU 慢。
2.  **不稳定性**：由于非单调性，在某些特定初始化或极深网络中，可能导致训练不稳定（虽然较少见）。