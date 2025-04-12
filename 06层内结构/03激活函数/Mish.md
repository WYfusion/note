函数：$Mish(x)=x\cdot\tanh(\ln(1+e^x))$
**函数范围**：约 $(−0.31,+∞)$，负输入被平滑抑制。
![[Mish.png|600]]
导函数：$Mish^{\prime}(x)=\tanh(\ln(1+e^x))+x\cdot\mathrm{sech}^2(\ln(1+e^x))\cdot\frac{e^x}{1+e^x}$
**导数范围**：$(0,1]$，连续且无饱和区域
![[Mish_derivative.png|600]]

### 特点
平滑、高精度，适合复杂任务