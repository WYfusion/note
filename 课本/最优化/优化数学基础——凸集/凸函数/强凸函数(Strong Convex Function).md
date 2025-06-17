![[Pasted image 20250607204257.png]]
**强凸函数定义2**：若存在常数 $m > 0$，对任意 $\boldsymbol{x}, \boldsymbol{y} \in \text{dom}f$ 及 $\theta \in (0,1)$，有$f(\theta\boldsymbol{x} + (1-\theta)\boldsymbol{y}) \leq \theta f(\boldsymbol{x}) + (1-\theta)f(\boldsymbol{y}) - \frac{m}{2}\theta(1-\theta)\|\boldsymbol{x}-\boldsymbol{y}\|^2,$ 则称 $f(\boldsymbol{x})$ 为强凸函数。

### 分析：
1. **凸组合形式**： $\theta\boldsymbol{x} + (1-\theta)\boldsymbol{y}$ 是标准凸组合，确保自变量的线性组合正确。
2. **强凸性修正项**：右边减去 $\frac{m}{2}\theta(1-\theta)\|\boldsymbol{x}-\boldsymbol{y}\|^2$，其中 $\theta(1-\theta) > 0$（$\theta \in (0,1)$），表明强凸函数比普通凸函数 “更凸”（函数值在凸组合处更小，偏离线性组合的程度由 m 控制）。
3. **与凸函数的关系**：普通凸函数对应 $m = 0$，强凸函数通过正参数 m 强化凸性，符合经典定义（如二次可微时 $\nabla^2 f \geq mI$ 等价于该不等式）。