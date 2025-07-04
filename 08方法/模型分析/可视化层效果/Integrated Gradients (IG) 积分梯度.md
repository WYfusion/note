**Integrated Gradients（积分梯度）** 是一种基于梯度的深度学习模型可解释性方法，旨在量化输入特征对模型预测结果的贡献程度。其核心思想是通过计算输入从**基线**（Baseline）到实际值的路径上梯度的积分，解决传统梯度方法中的**梯度饱和问题**（如ReLU函数的负半轴梯度消失），从而更准确地反映特征重要性。
#### 核心公式
对于输入 `x` 和基线 `x′`，特征 `i` 的积分梯度贡献值为：$$\phi_i^{I G}(x)=(x_i-x_i^{\prime})\int_0^1\frac{\partial F(x^{\prime}+\alpha(x-x^{\prime}))}{\partial x_i}d\alpha$$
其中，`F` 为模型输出函数，`α` 为插值系数，积分路径通常为线性插值（从基线到输入的直线路径）
### 实现流程
#### 1. **选择基线（Baseline）**
- **基线的作用**：作为特征贡献的“参考起点”，通常选择使模型输出接近零的输入（如全黑图像、零向量等）。例如，图像任务可选全黑图，文本任务可选零嵌入向量。
- **基线的影响**：不同基线可能影响解释结果，需根据任务调整。例如，对抗样本或均值图像也可作为基线。
#### 2. **定义积分路径**
- **路径选择**：通常采用线性插值路径，即生成一系列中间输入$x_{\alpha}=x^{\prime}+\alpha(x-x^{\prime})$，其中 $α∈[0,1]$ 控制插值比例。
- **离散化积分**：实际计算中将积分近似为有限步数 `m` 的累加，例如：$\phi_{i}^{IG}(x)\:\approx\:(x_{i}-x_{i}^{'})\:\sum\limits_{k=1}^{m}\:\frac{\partial F(x^{'}+\frac{k}{m}(x-x^{'}))}{\partial x_{i}}\:\cdot\:\frac{1}{m}$
#### 3. **计算梯度并积分**
- **梯度计算**：对每个中间输入 xαxα​ 进行前向传播和反向传播，获取模型输出对输入的梯度$\frac{\partial F(x_\alpha)}{\partial x}$
#### 4. **可视化与解释**
- **生成热力图**：将各特征的贡献值归一化后映射为热力图（如图像中的显著区域或文本中的关键词权重）。
- **后处理**：可能结合平滑处理（如SmoothGrad）或与其他方法（如Grad-CAM）融合，提升可视化效果。
### 核心优势与公理
1. **敏感性（Sensitivity）**：若输入与基线在某一特征不同且导致预测变化，该特征的贡献值非零，避免传统梯度方法在饱和区的失效。
2. **实现不变性（Implementation Invariance）**：功能等效的不同模型实现（如不同网络结构）对同一输入的解释结果一致。
3. **完整性（Completeness）**：所有特征贡献值的总和等于模型预测从基线到输入的差值，即$\sum\phi_i^{IG}=F(x)-F(x^{\prime})$