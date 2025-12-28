# Mish 激活函数

## 1. 定义与公式
**Mish** 是一个自正则化的非单调激活函数，灵感来源于 Swish 和 GELU。它在 YOLOv4 等模型中表现出色。

**公式**：
$$ \text{Mish}(x) = x \cdot \tanh(\text{softplus}(x)) = x \cdot \tanh(\ln(1 + e^x)) $$

**图像**：
![[Mish.png|600]]

## 2. 导函数

**导数公式**：
令 $\omega = \text{softplus}(x) = \ln(1+e^x)$，则：
$$ \text{Mish}'(x) = \tanh(\omega) + x \cdot \text{sech}^2(\omega) \cdot \sigma(x) $$
其中 $\sigma(x)$ 是 Sigmoid 函数。

**导数图像**：
![[Mish_derivative.png|600]]

## 3. 优缺点分析

### 优点
1.  **平滑非单调**：与 Swish 类似，Mish 也是平滑且非单调的，允许负梯度流入，有助于信息的深度传播。
2.  **无上界有下界**：正区间线性，负区间近似为 0。
3.  **连续可导**：处处光滑，避免了 ReLU 在零点的奇异性。
4.  **鲁棒性**：实验表明 Mish 对输入噪声和初始化更具鲁棒性。

### 缺点
1.  **计算昂贵**：涉及 $\tanh, \ln, e^x$ 等多种复杂运算，是目前常见激活函数中计算量较大的之一。