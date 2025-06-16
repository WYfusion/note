$g$是$\nabla f(x^k)$的含义
![[Pasted image 20250610161111.png|800]]
![[Pasted image 20250610161906.png|800]]
![[Pasted image 20250610162037.png|800]]
![[Pasted image 20250610162049.png|800]]
![[Pasted image 20250610162058.png|800]]


#### 例子
![[Pasted image 20250616204521.png]]
- **$\eta$（接受阈值）**：当实际下降量与预测下降量的比值 $\rho \geq \eta$ 时，认为子问题近似可靠，**接受当前步长并扩大信赖域**，另外$\eta$通常是不变的
- **$\mu$（拒绝阈值）**：当 $\rho \leq \mu$ 时，认为子问题近似不可靠，**拒绝当前步长并缩小信赖域**。
- $\gamma_1$**（缩小倍数）**：缩小信赖域：$\Delta_{k+1} = \gamma_1 \Delta_k$
- **$\gamma_2$（扩大倍数）**：当满足 $\rho \geq \eta$ 时，信赖域半径按 $\Delta_{k+1} = \gamma_2 \Delta_k$ 扩大。
$\Delta_{k+1} = \Delta_k$条件$\mu < \rho < \eta$接受步长，但保持半径不变

### 信赖域方法求解步骤
1. **信赖域原理**： 信赖域方法通过求解二次子问题近似原函数，调整信赖域半径以平衡局部近似精度。子问题为：$$\min_{\boldsymbol{d}} \, q(\boldsymbol{d}) = f(\boldsymbol{x}^k) + \nabla f(\boldsymbol{x}^k)^T \boldsymbol{d} + \frac{1}{2} \boldsymbol{d}^T B^k \boldsymbol{d}, \quad \|\boldsymbol{d}\| \leq \Delta_k$$ 其中 $B^k$ 为海瑟矩阵（此处函数二次可分离，海瑟矩阵对角正定），$\Delta_k$ 为信赖域半径。通过计算实际与预测下降量的比值 $\rho$ 调整 $\Delta_k$：
    - $\rho \geq \eta$（接受步长，扩大半径），
    - $\rho < \mu$（拒绝步长，缩小半径）。
2. **第一次迭代（$k=1$）**：
    - **初始点**：$\boldsymbol{x}^{(1)} = [0, 0]^T$，$f=5$，$\nabla f = [0, -4]^T$，$B = \text{diag}(2, 2)$，$\Delta_1=1$。
    - **子问题**：$q(\boldsymbol{d}) = 5 - 4d_2 + d_1^2 + d_2^2$，无约束极小点 $[0, 2]^T$ 超出信赖域（$\|\boldsymbol{d}\|=2>1$），故在边界 $d_1^2 + d_2^2=1$ 取 $\boldsymbol{d}^{(1)} = [0, 1]^T$（$d_2=1$ 最小化 $q(\boldsymbol{d})$）。
    - **更新**：
        - **预测下降量**：$q(\boldsymbol{0}) = f(\boldsymbol{x}^{(1)}) = 5, \quad q(\boldsymbol{d}^{(1)}) = 5 - 4 \cdot 1 + 0 + 1 = 2,$  ,预测下降量为 $5 - 2 = 3$
        - **更新迭代点**：$\boldsymbol{x}^{(2)} = \boldsymbol{x}^{(1)} + \boldsymbol{d}^{(1)} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} + \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$  ，注意，这里不用考虑步长的事情。信赖域求解方法是自适应的，边界上的解（超出时）或无约束解（在内时），无需额外步长参数（如 $\alpha$），而是通过调整信赖域半径 $\Delta_k$ 控制搜索范围。
        - 验证原函数值：$f(\boldsymbol{x}^{(2)}) = 0^4 + 0^2 + 1^2 - 4 \cdot 1 + 5 = 2$，$f(\boldsymbol{x}^{(1)}) = 0^4 + 0^2 + 0^2 - 4 \cdot 0 + 5 = 5$ ，下降了3
    - 实际与预测下降量均为 3，$\rho=1 \geq 3/4$，接受步长，$\boldsymbol{x}^{(2)} = [0, 1]^T$，$\Delta_2=2$（扩大半径）。
3. **第二次迭代（$k=2$）**：
    - **当前点**：$\boldsymbol{x}^{(2)} = [0, 1]^T$，$f=2$，$\nabla f = [0, -2]^T$，$\Delta_2=2$。
    - **子问题**：$q(\boldsymbol{d}) = 2 - 2d_2 + d_1^2 + d_2^2$，无约束极小点 $[0, 1]^T$ 在信赖域内（$\|\boldsymbol{d}\|=1 \leq 2$），直接取此点。
    - **更新**：
        - **预测下降量**：$q(\boldsymbol{0}) = f(\boldsymbol{x}^{(2)}) = 2, \quad q(\boldsymbol{d}^{(2)}) = 2 - 2 \cdot 1 + 0 + 1 = 1,$ 预测下降量为 $2 - 1 = 1$。
        - **更新迭代点**：$\boldsymbol{x}^{(3)} = \boldsymbol{x}^{(2)} + \boldsymbol{d}^{(2)} = \begin{bmatrix} 0 \\ 1 \end{bmatrix} + \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 0 \\ 2 \end{bmatrix}.$ （信赖域方法中，$\boldsymbol{d}^{(2)}$ 已由子问题自适应确定步长，无需额外参数。）
        - 验证原函数值：$f(\boldsymbol{x}^{(3)}) = 0^4 + 0^2 + (2 - 2)^2 + 1 = 1.$下降了1
    - 实际与预测下降量均为 1，$\rho = \frac{1}{1} = 1 \geq \frac{3}{4}$，接受步长，更新 $\boldsymbol{x}^{(3)} = \begin{bmatrix} 0 \\ 2 \end{bmatrix}$。
- **收敛性验证**：$\nabla f(\boldsymbol{x}^{(3)}) = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$，海瑟矩阵正定，故 $\boldsymbol{x}^{(3)}$ 为全局最优解。
### 最终答案

$\boxed{\begin{bmatrix} 0 \\ 2 \end{bmatrix}}$
