在研究问题时，当变量属于**正态分布的随机变量**，服从**样本独立假定**时，我们会选择LM-线性(回归)模型
### 普通线性模型
$$y=\beta_0+\beta_1*x+\varepsilon$$
也可以是
$$Y=\mathbf{X}\beta+\varepsilon$$
也就是**固定效应+随机误差**，且因变量必须要满足**正态性、独立性以及方差齐次性**。
### 多元线性回归模型
$$f(\mathbf{x})=\mathbf{k}^\mathrm{T}\mathbf{x}+b$$
$\mathbf{x}$为输入向量，包含多个特征（自变量）
$f(\mathbf{x})$为模型的输出或响应（预测的目标变量）
$\mathbf{k^T}$ 为特征权重
$b$为是模型的截距或偏置

我们的目标是通过学习$\mathbf{k^T}$ 和 $b$ 使得$f(\mathbf{x})$尽可能的接近真实观测值 $\mathbf{y}$ 。
为了方便计算和编程，我们可以将 $b$ 吸收进 $\mathbf{x}$ 和  $\mathbf{k}$ 中去，使得 $y=\mathbf{\hat k^T}\mathbf{\hat x}$