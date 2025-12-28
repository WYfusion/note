 # GELU 激活函数 (Gaussian Error Linear Unit)

## 1. 定义与公式
**GELU** 结合了 ReLU（线性特性）、Dropout（随机性）和 Zoneout 的思想。它通过将输入乘以其服从高斯分布的累积分布函数 (CDF) 来实现激活。GELU 在 BERT、GPT 等 Transformer 模型中被广泛使用。

**数学定义**：
$$ \text{GELU}(x) = x \cdot \Phi(x) $$
其中 $\Phi(x)$ 是标准正态分布的累积分布函数：
$$ \Phi(x) = P(X \le x) = \frac{1}{2} \left[ 1 + \text{erf}\left( \frac{x}{\sqrt{2}} \right) \right] $$

**近似计算公式**（为了加速计算）：
$$ \text{GELU}(x) \approx 0.5x \left( 1 + \tanh\left[ \sqrt{\frac{2}{\pi}} (x + 0.044715x^3) \right] \right) $$

**图像**：
![[Figure_1.png|500]]

## 2. 导函数

**导数公式**：
$$ \frac{d}{dx}\text{GELU}(x) = \Phi(x) + x \cdot \phi(x) $$
其中 $\phi(x)$ 是标准正态分布的概率密度函数 (PDF)：
$$ \phi(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}} $$

**导数图像**：
![[GELU_derivative.png|500]]

## 3. 优缺点分析

### 优点
1.  **平滑性**：GELU 是处处连续且光滑的（在 $x=0$ 处可导），这有助于优化算法的梯度下降。
2.  **概率解释**：可以看作是对神经元输入的“随机正则化”，输入越小，被丢弃（归零）的概率越大；输入越大，保留的概率越大。
3.  **高性能**：在 NLP 和 CV 的许多 SOTA 模型（如 BERT, ViT）中表现优于 ReLU 和 ELU。

### 缺点
1.  **计算复杂度**：涉及 $\tanh$ 或 $\text{erf}$ 运算，比 ReLU 慢。
2.  **缺乏稀疏性**：不像 ReLU 那样在负区间完全为 0，GELU 在负区间有微小的值，无法产生真正的稀疏激活。
