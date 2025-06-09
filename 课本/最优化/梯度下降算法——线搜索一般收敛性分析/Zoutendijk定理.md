![[Pasted image 20250609222434.png]]

#### 1. 定理条件与符号
- **迭代格式**：$x^{k+1} = x^k + \alpha_k d^k$，其中 $d^k$ 是下降方向（$-\nabla f(x^k)^T d^k > 0$这是下降方向的**固有属性**），$\alpha_k$ 满足 **Wolfe 准则**（充分下降：$f(x^{k+1}) \leq f(x^k) + c_1 \alpha_k \nabla f(x^k)^T d^k$；曲率条件：$\nabla f(x^{k+1})^T d^k \geq c_2 \nabla f(x^k)^T d^k$，$0 < c_1 < c_2 < 1$）。
- **函数性质**：f 下有界（$\inf f(x) > -\infty$），梯度 L-Lipschitz 连续（$\|\nabla f(x) - \nabla f(y)\| \leq L\|x - y\|$）。[[梯度Lipschitz连续]]
- **夹角余弦**：$\cos\theta_k = \frac{-\nabla f(x^k)^T d^k}{\|\nabla f(x^k)\|\|d^k\|}$（因 $d^k$ 是下降方向，$\cos\theta_k > 0$，表示负梯度与搜索方向的接近程度）。
#### 2. 推导过程
- **步骤 1：利用梯度 Lipschitz 连续性** 对 $x^{k+1} = x^k + \alpha_k d^k$，有：$\|\nabla f(x^{k+1}) - \nabla f(x^k)\| \leq L\alpha_k \|d^k\| \implies \|\nabla f(x^{k+1})\| \geq \|\nabla f(x^k)\| - L\alpha_k \|d^k\|$ （三角不等式，两边平方后整理得能量不等式基础）。
- **步骤 2：结合 Wolfe 的充分下降条件** 由 $f(x^{k+1}) \leq f(x^k) + c_1 \alpha_k \nabla f(x^k)^T d^k$，得：$\alpha_k (-\nabla f(x^k)^T d^k) \leq \frac{1}{c_1}(f(x^k) - f(x^{k+1}))$ 因 f 下有界，级数 $\sum (f(x^k) - f(x^{k+1}))$ 收敛，故 $\sum \alpha_k (-\nabla f(x^k)^T d^k) < \infty$。
- **步骤 3：代入夹角余弦** 令 $-\nabla f(x^k)^T d^k = \|\nabla f(x^k)\|\|d^k\| \cos\theta_k$，则：$\sum \alpha_k \|\nabla f(x^k)\|\|d^k\| \cos\theta_k < \infty$ 由 Wolfe 准则，步长 $\alpha_k \geq \alpha_{\text{min}} > 0$（曲率条件保证步长下界），且 $\|d^k\| \geq \delta > 0$（下降方向非零），故：$\sum \cos\theta_k \|\nabla f(x^k)\| < \infty$
- **步骤 4：利用曲率条件推导平方项** 由曲率条件 $\nabla f(x^{k+1})^T d^k \geq c_2 \nabla f(x^k)^T d^k$，结合梯度 Lipschitz，可得：$\|\nabla f(x^{k+1})\|^2 \geq \|\nabla f(x^k)\|^2 - 2L\alpha_k (-\nabla f(x^k)^T d^k)$ 累加后利用级数收敛性，最终得：$\sum_\limits{k=0}^\infty \cos^2\theta_k \|\nabla f(x^k)\|^2 < \infty$
#### 3. 定理含义
- **级数收敛**：表明 $\cos^2\theta_k \|\nabla f(x^k)\|^2 \to 0$。若算法保证 $\cos\theta_k \not\to 0$（如最速下降法 $\cos\theta_k = 1$，牛顿法 $\cos\theta_k \to 1$），则 $\|\nabla f(x^k)\| \to 0$，即收敛到临界点。
- **应用场景**：证明线搜索算法（如最速下降法、牛顿法）的收敛性，当搜索方向始终与负梯度 “充分接近”（$\cos\theta_k$ 不趋于 0）时，梯度必趋于零。
#### 4. 关键公式总结
- **夹角余弦**：$\cos\theta_k = \frac{-\nabla f(x^k)^T d^k}{\|\nabla f(x^k)\|\|d^k\|}$
- **Zoutendijk 条件**：$\sum_{k=0}^\infty \cos^2\theta_k \|\nabla f(x^k)\|^2 < +\infty$
- **推导依赖**：Wolfe 准则、梯度 Lipschitz 连续性、函数下有界性，通过能量不等式和级数收敛性严格证明。
### 总结
Zoutendijk 定理通过分析搜索方向与负梯度的夹角，建立了线搜索算法的收敛性判据。其核心是利用 Wolfe 准则保证步长和方向的有效性，结合梯度的 Lipschitz 性质，推导出梯度范数与夹角余弦的平方乘积级数收敛，为算法收敛性提供了理论支撑。对于实际优化算法（如最速下降、牛顿法），该定理确保了在合理方向选择下，梯度必趋于零，从而收敛到临界点。