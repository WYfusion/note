![[Pasted image 20250604195442.png|800]]

从最优化角度证明 BFGS 算法的更新公式为上述优化问题的解，步骤如下：
BFGS (Broyden-Fletcher-Goldfarb-Shanno) 方法是拟牛顿法的一种经典实现，通过迭代更新海瑟矩阵的近似逆矩阵 $H_k \approx \nabla^2 f(x_k)^{-1}$。下面从基本原理出发，详细推导 BFGS 公式的完整过程。

#### 1. 拟牛顿条件与目标

对于二次函数 $f(x) = \frac{1}{2}x^T H x + b^T x + c$，海瑟矩阵 H 满足： $\nabla f(x_{k+1}) - \nabla f(x_k) = H(x_{k+1} - x_k)$ 定义 $s_k = x_{k+1} - x_k$ 和 $y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$，则拟牛顿条件为： $B_{k+1} s_k = y_k \quad \text{或} \quad H_{k+1} y_k = s_k$ 其中 $B_{k+1} \approx H$ 是海瑟矩阵的近似，$H_{k+1} \approx H^{-1}$ 是海瑟逆矩阵的近似。
#### 2. 对称秩二更新（SR2）框架
假设从 $H_k$ 到 $H_{k+1}$ 的更新形式为： $H_{k+1} = H_k + P_k + Q_k$ 其中 $P_k$ 和 $Q_k$ 是秩一矩阵，即 $P_k = \alpha u u^T$，$Q_k = \beta v v^T$。代入拟牛顿条件 $H_{k+1} y_k = s_k$ 得： $(H_k + \alpha u u^T + \beta v v^T)y_k = s_k$ 整理得： $\alpha (u^T y_k) u + \beta (v^T y_k) v = s_k - H_k y_k$ 选择 $u = s_k$ 和 $v = H_k y_k$，则： $\alpha (s_k^T y_k) s_k + \beta (y_k^T H_k y_k) H_k y_k = s_k - H_k y_k$ 解得： $\alpha = \frac{1}{s_k^T y_k}, \quad \beta = -\frac{1}{y_k^T H_k y_k}$ 得到 DFP 公式： $H_{k+1}^{DFP} = H_k + \frac{s_k s_k^T}{s_k^T y_k} - \frac{H_k y_k y_k^T H_k}{y_k^T H_k y_k}$
#### 3. BFGS 公式的推导
BFGS 公式通过对 DFP 公式的对偶变换得到。考虑海瑟矩阵近似 $B_k \approx H$ 的更新： $B_{k+1} = B_k + \frac{y_k y_k^T}{y_k^T s_k} - \frac{B_k s_k s_k^T B_k}{s_k^T B_k s_k}$ 利用 Sherman-Morrison 公式对 $B_{k+1}$ 求逆，得到 $H_{k+1} = B_{k+1}^{-1}$ 的表达式。
设 $A = B_k$，$u = \frac{y_k}{\sqrt{y_k^T s_k}}$，$v = \frac{y_k}{\sqrt{y_k^T s_k}}$，则： $B_{k+1} = A + u u^T - \frac{A s_k s_k^T A}{s_k^T A s_k}$ 应用两次 Sherman-Morrison 公式： $H_{k+1} = (I - \rho_k s_k y_k^T)H_k(I - \rho_k y_k s_k^T) + \rho_k s_k s_k^T$ 其中 $\rho_k = \frac{1}{y_k^T s_k}$。展开后得到 BFGS 公式： $H_{k+1} = H_k - \frac{H_k y_k y_k^T H_k}{y_k^T H_k y_k} + \frac{s_k s_k^T}{y_k^T s_k}$
#### 4. BFGS 公式的另一种推导
从目标函数出发，寻找满足拟牛顿条件且最小化 $\|B_{k+1} - B_k\|_W$ 的对称矩阵 $B_{k+1}$，其中 $\| \cdot \|_W$ 是加权范数。通过求解优化问题： $\min_{B} \|B - B_k\|_W \quad \text{s.t.} \quad B s_k = y_k, \quad B = B^T$ 得到 BFGS 更新公式。类似地，对逆矩阵 $H_{k+1}$ 求解： $\min_{H} \|H - H_k\|_{W^{-1}} \quad \text{s.t.} \quad H y_k = s_k, \quad H = H^T$ 同样得到 BFGS 公式。
#### 5. BFGS 公式的最终形式
综合上述推导，BFGS 更新公式为： $H_{k+1} = \left(I - \frac{s_k y_k^T}{y_k^T s_k}\right)H_k\left(I - \frac{y_k s_k^T}{y_k^T s_k}\right) + \frac{s_k s_k^T}{y_k^T s_k}$ 展开后等价于： $H_{k+1} = H_k + \frac{s_k s_k^T}{y_k^T s_k} - \frac{H_k y_k y_k^T H_k + y_k y_k^T H_k}{y_k^T s_k} + \frac{(y_k^T H_k y_k)}{(y_k^T s_k)^2}s_k s_k^T$
#### 6. BFGS 方法的算法流程
1. **初始化**：给定初始点 $x_0$，初始海瑟逆近似 $H_0 = I$（单位矩阵）
2. **迭代**：对于 $k = 0, 1, 2, \dots$
    - 计算搜索方向：$p_k = -H_k \nabla f(x_k)$
    - 线搜索：确定步长 $\alpha_k$ 满足 Wolfe 条件
    - 更新点：$x_{k+1} = x_k + \alpha_k p_k$
    - 计算 $s_k = x_{k+1} - x_k$ 和 $y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$
    - 更新 $H_{k+1}$ 使用 BFGS 公式
3. **终止条件**：当 $\|\nabla f(x_k)\| < \epsilon$ 时停止
#### 7. BFGS 方法的性质
1. **拟牛顿条件**：$H_{k+1} y_k = s_k$
2. **对称性**：$H_{k+1} = H_{k+1}^T$
3. **正定性保持**：若 $H_k$ 正定且 $y_k^T s_k > 0$，则 $H_{k+1}$ 正定
4. **超线性收敛**：在适当条件下，BFGS 方法具有超线性收敛速度
5. **二次终止性**：对于二次函数，BFGS 方法在有限步内收敛到极小点
BFGS 方法通过迭代更新海瑟矩阵的近似逆，结合线搜索技术，在不需要显式计算海瑟矩阵的情况下，实现了接近牛顿法的收敛速度，是无约束优化问题中最有效的方法之一。