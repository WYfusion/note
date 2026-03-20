# LMS自适应算法

## 1. 引言

最小均方（Least Mean Squares, LMS）算法是由Widrow和Hoff于1960年提出的一种自适应滤波算法。它是最简单、最广泛使用的自适应算法之一，以其计算简单和良好的收敛性能著称。

**核心思想**：使用瞬时梯度估计代替真实梯度，通过随机梯度下降迭代更新滤波器系数。

**应用领域**：
- 系统辨识
- 回声消除
- 噪声消除
- 信道均衡
- 自适应波束形成

---

## 2. 问题建模

### 2.1 自适应滤波器结构

考虑一个FIR（有限冲激响应）自适应滤波器：

```
x(n) ──┬──[z⁻¹]──┬──[z⁻¹]──┬── ... ──┬──[z⁻¹]──┐
       │         │         │         │         │
       ↓         ↓         ↓         ↓         ↓
      w₀        w₁        w₂       ...      w_{M-1}
       │         │         │         │         │
       └────┬────┴────┬────┴────┬────┴────┬────┘
            │         │         │         │
            └─────────┴─────────┴─────────┘
                          │
                          ↓ Σ
                        y(n) ──→ (+) ←── d(n)
                                  │
                                  ↓
                                e(n) = d(n) - y(n)
```

**滤波器输出**：
$$y(n) = \sum_{i=0}^{M-1} w_i(n) x(n-i) = \mathbf{w}^T(n)\mathbf{x}(n)$$

其中：
- $\mathbf{w}(n) = [w_0(n), w_1(n), ..., w_{M-1}(n)]^T$：滤波器系数向量
- $\mathbf{x}(n) = [x(n), x(n-1), ..., x(n-M+1)]^T$：输入信号向量
- $M$：滤波器阶数

### 2.2 误差信号

定义误差信号为期望信号与滤波器输出之差：
$$e(n) = d(n) - y(n) = d(n) - \mathbf{w}^T(n)\mathbf{x}(n)$$

---

## 3. 最优Wiener解

### 3.1 均方误差代价函数

定义均方误差（MSE）代价函数：
$$J(\mathbf{w}) = E[e^2(n)] = E[(d(n) - \mathbf{w}^T\mathbf{x}(n))^2]$$

展开：
$$\begin{aligned}
J(\mathbf{w}) &= E[d^2(n)] - 2\mathbf{w}^T E[d(n)\mathbf{x}(n)] + \mathbf{w}^T E[\mathbf{x}(n)\mathbf{x}^T(n)]\mathbf{w} \\
&= \sigma_d^2 - 2\mathbf{w}^T\mathbf{p} + \mathbf{w}^T\mathbf{R}\mathbf{w}
\end{aligned}$$

其中：
- $\mathbf{R} = E[\mathbf{x}(n)\mathbf{x}^T(n)]$：输入信号的自相关矩阵
- $\mathbf{p} = E[d(n)\mathbf{x}(n)]$：输入与期望信号的互相关向量

### 3.2 Wiener-Hopf方程

对$J(\mathbf{w})$求梯度并令其为零：
$$\nabla_\mathbf{w} J = -2\mathbf{p} + 2\mathbf{R}\mathbf{w} = \mathbf{0}$$

得到**Wiener-Hopf方程**：
$$\boxed{\mathbf{R}\mathbf{w}_{opt} = \mathbf{p}}$$

**最优Wiener解**：
$$\boxed{\mathbf{w}_{opt} = \mathbf{R}^{-1}\mathbf{p}}$$

### 3.3 MSE曲面特性

MSE代价函数$J(\mathbf{w})$是关于$\mathbf{w}$的二次函数，形成一个**超抛物面**：
- 唯一全局最小值点：$\mathbf{w}_{opt}$
- 最小MSE：$J_{min} = \sigma_d^2 - \mathbf{p}^T\mathbf{R}^{-1}\mathbf{p}$

---

## 4. LMS算法推导

### 4.1 最速下降法

最速下降法的迭代公式：
$$\mathbf{w}(n+1) = \mathbf{w}(n) - \frac{\mu}{2}\nabla_\mathbf{w} J(n)$$

其中$\mu$为步长参数。

真实梯度：
$$\nabla_\mathbf{w} J = -2\mathbf{p} + 2\mathbf{R}\mathbf{w} = -2E[e(n)\mathbf{x}(n)]$$

### 4.2 瞬时梯度估计

**关键近似**：用瞬时值代替期望值
$$\hat{\nabla}_\mathbf{w} J(n) = -2e(n)\mathbf{x}(n)$$

这是一个**无偏估计**：
$$E[\hat{\nabla}_\mathbf{w} J(n)] = -2E[e(n)\mathbf{x}(n)] = \nabla_\mathbf{w} J$$

### 4.3 LMS更新公式

将瞬时梯度代入最速下降法：
$$\mathbf{w}(n+1) = \mathbf{w}(n) - \frac{\mu}{2}(-2e(n)\mathbf{x}(n))$$

得到**LMS算法**：
$$\boxed{\mathbf{w}(n+1) = \mathbf{w}(n) + \mu \cdot e(n) \cdot \mathbf{x}(n)}$$

---

## 5. LMS算法总结

### 5.1 算法流程

```
初始化: w(0) = 0 (或小随机值)

对于每个时刻 n = 0, 1, 2, ...：
    1. 计算滤波器输出:
       y(n) = w^T(n) · x(n)
    
    2. 计算误差:
       e(n) = d(n) - y(n)
    
    3. 更新滤波器系数:
       w(n+1) = w(n) + μ · e(n) · x(n)
```

### 5.2 计算复杂度

| 操作     | 乘法次数   | 加法次数  |
| ------ | ------ | ----- |
| 滤波器输出  | $M$    | $M-1$ |
| 误差计算   | $0$    | $1$   |
| 系数更新   | $M+1$  | $M$   |
| **总计** | $2M+1$ | $2M$  |

每次迭代复杂度：$O(M)$

---

## 6. 收敛性分析

### 6.1 均值收敛

定义权值误差向量：
$$\mathbf{v}(n) = \mathbf{w}(n) - \mathbf{w}_{opt}$$

在一定假设下，权值误差的期望满足：
$$E[\mathbf{v}(n+1)] = (\mathbf{I} - \mu\mathbf{R})E[\mathbf{v}(n)]$$

**均值收敛条件**：矩阵$(\mathbf{I} - \mu\mathbf{R})$的所有特征值模小于1。

设$\mathbf{R}$的特征值为$\lambda_1, \lambda_2, ..., \lambda_M$，则需要：
$$|1 - \mu\lambda_i| < 1, \quad \forall i$$

即：
$$0 < \mu < \frac{2}{\lambda_{max}}$$

### 6.2 步长选择

**稳定性条件**：
$$\boxed{0 < \mu < \frac{2}{\lambda_{max}}}$$

由于$\text{tr}(\mathbf{R}) = \sum_{i=1}^M \lambda_i \geq \lambda_{max}$，实用条件为：
$$0 < \mu < \frac{2}{\text{tr}(\mathbf{R})} = \frac{2}{M \cdot \sigma_x^2}$$

其中$\sigma_x^2$为输入信号功率。

**步长的影响**：
- $\mu$大：收敛快，但稳态误差大，可能不稳定
- $\mu$小：收敛慢，但稳态误差小，更稳定

### 6.3 收敛速度

收敛时间常数与$\mathbf{R}$的特征值分布有关：
$$\tau_{mse} \approx \frac{1}{4\mu\lambda_{min}}$$

**特征值扩散**（Eigenvalue Spread）：
$$\chi = \frac{\lambda_{max}}{\lambda_{min}}$$

$\chi$越大，收敛越慢。

### 6.4 稳态超量MSE

LMS算法的稳态MSE比最优Wiener解略大：
$$J(\infty) = J_{min}(1 + \mu \cdot \text{tr}(\mathbf{R}))$$

**超量MSE**（Excess MSE）：
$$J_{ex} = J(\infty) - J_{min} = \mu \cdot J_{min} \cdot \text{tr}(\mathbf{R})$$

**失调系数**（Misadjustment）：
$$\mathcal{M} = \frac{J_{ex}}{J_{min}} = \mu \cdot \text{tr}(\mathbf{R}) \approx \mu \cdot M \cdot \sigma_x^2$$

---

## 7. LMS变体

### 7.1 归一化LMS（NLMS）

为解决输入信号功率变化导致的收敛问题，引入归一化：
$$\mathbf{w}(n+1) = \mathbf{w}(n) + \frac{\mu}{\|\mathbf{x}(n)\|^2 + \epsilon} e(n)\mathbf{x}(n)$$

其中$\epsilon$是小正数，防止除零。

**优点**：
- 收敛速度与输入功率无关
- 步长自动调整
- 稳定性更好

### 7.2 带泄漏的LMS（Leaky LMS）

$$\mathbf{w}(n+1) = (1-\mu\gamma)\mathbf{w}(n) + \mu \cdot e(n) \cdot \mathbf{x}(n)$$

其中$0 < \gamma \ll 1$为泄漏因子。

**优点**：防止系数漂移，提高数值稳定性。

### 7.3 符号LMS算法

为降低计算复杂度：

**Sign-Error LMS**：
$$\mathbf{w}(n+1) = \mathbf{w}(n) + \mu \cdot \text{sign}(e(n)) \cdot \mathbf{x}(n)$$

**Sign-Data LMS**：
$$\mathbf{w}(n+1) = \mathbf{w}(n) + \mu \cdot e(n) \cdot \text{sign}(\mathbf{x}(n))$$

**Sign-Sign LMS**：
$$\mathbf{w}(n+1) = \mathbf{w}(n) + \mu \cdot \text{sign}(e(n)) \cdot \text{sign}(\mathbf{x}(n))$$

---

## 8. LMS与Wiener滤波器的比较

| 特性 | Wiener滤波器 | LMS算法 |
|------|--------------|---------|
| 计算方式 | 批处理 | 在线/递归 |
| 需要统计量 | $\mathbf{R}, \mathbf{p}$ | 不需要 |
| 计算复杂度 | $O(M^3)$（矩阵求逆） | $O(M)$（每次迭代） |
| 适应性 | 静态 | 可跟踪时变系统 |
| 最优性 | 全局最优 | 渐近最优 |

---

## 9. 参考文献

1. Widrow, B., & Hoff, M. E. (1960). Adaptive switching circuits. *IRE WESCON Convention Record*, 4, 96-104.
2. Haykin, S. (2002). *Adaptive Filter Theory* (4th ed.). Prentice Hall.
3. Sayed, A. H. (2008). *Adaptive Filters*. Wiley-IEEE Press.
