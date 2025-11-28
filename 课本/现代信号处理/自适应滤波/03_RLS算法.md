# RLS递归最小二乘算法

## 1. 引言

递归最小二乘（Recursive Least Squares, RLS）算法是一种基于最小二乘准则的自适应滤波算法。与LMS相比，RLS具有更快的收敛速度，但计算复杂度更高。

**核心思想**：递归地最小化加权误差平方和，通过矩阵求逆引理避免直接矩阵求逆。

**特点**：
- 收敛速度快（与输入信号相关性无关）
- 计算复杂度$O(M^2)$
- 需要更多存储空间
- 数值稳定性需要关注

---

## 2. 问题建模

### 2.1 最小二乘准则

考虑到时刻$n$的加权最小二乘代价函数：
$$J(n) = \sum_{i=1}^{n} \lambda^{n-i} |e(i)|^2 = \sum_{i=1}^{n} \lambda^{n-i} |d(i) - \mathbf{w}^T\mathbf{x}(i)|^2$$

其中：
- $\lambda \in (0, 1]$：**遗忘因子**（forgetting factor）
- $\lambda^{n-i}$：对过去数据的指数加权，越早的数据权重越小
- $\lambda = 1$：标准最小二乘（所有数据等权重）
- $\lambda < 1$：指数加权最小二乘（适应时变系统）

### 2.2 遗忘因子的作用

有效数据窗口长度：
$$N_{eff} = \frac{1}{1-\lambda}$$

| $\lambda$ | 有效窗口长度 | 特点 |
|-----------|--------------|------|
| 1.0 | $\infty$ | 无限记忆，适合平稳系统 |
| 0.99 | 100 | 长记忆 |
| 0.95 | 20 | 中等记忆 |
| 0.9 | 10 | 短记忆，快速跟踪 |

---

## 3. RLS算法推导

### 3.1 正规方程

对$J(n)$关于$\mathbf{w}$求导并令其为零：
$$\nabla_\mathbf{w} J(n) = -2\sum_{i=1}^{n} \lambda^{n-i} \mathbf{x}(i)[d(i) - \mathbf{w}^T\mathbf{x}(i)] = \mathbf{0}$$

整理得**正规方程**：
$$\boldsymbol{\Phi}(n)\mathbf{w}(n) = \boldsymbol{\theta}(n)$$

其中：
- **加权自相关矩阵**：$\boldsymbol{\Phi}(n) = \sum_{i=1}^{n} \lambda^{n-i} \mathbf{x}(i)\mathbf{x}^T(i)$
- **加权互相关向量**：$\boldsymbol{\theta}(n) = \sum_{i=1}^{n} \lambda^{n-i} d(i)\mathbf{x}(i)$

### 3.2 递归关系

**自相关矩阵的递归**：
$$\begin{aligned}
\boldsymbol{\Phi}(n) &= \sum_{i=1}^{n} \lambda^{n-i} \mathbf{x}(i)\mathbf{x}^T(i) \\
&= \lambda \sum_{i=1}^{n-1} \lambda^{n-1-i} \mathbf{x}(i)\mathbf{x}^T(i) + \mathbf{x}(n)\mathbf{x}^T(n) \\
&= \lambda \boldsymbol{\Phi}(n-1) + \mathbf{x}(n)\mathbf{x}^T(n)
\end{aligned}$$

**互相关向量的递归**：
$$\boldsymbol{\theta}(n) = \lambda \boldsymbol{\theta}(n-1) + d(n)\mathbf{x}(n)$$

### 3.3 矩阵求逆引理

直接求$\boldsymbol{\Phi}^{-1}(n)$计算量大。使用**矩阵求逆引理**（Woodbury恒等式）：

若$\mathbf{A}^{-1}$已知，则：
$$(\mathbf{A} + \mathbf{u}\mathbf{v}^T)^{-1} = \mathbf{A}^{-1} - \frac{\mathbf{A}^{-1}\mathbf{u}\mathbf{v}^T\mathbf{A}^{-1}}{1 + \mathbf{v}^T\mathbf{A}^{-1}\mathbf{u}}$$

定义$\mathbf{P}(n) = \boldsymbol{\Phi}^{-1}(n)$，应用矩阵求逆引理：
$$\mathbf{P}(n) = (\lambda \boldsymbol{\Phi}(n-1) + \mathbf{x}(n)\mathbf{x}^T(n))^{-1}$$

令$\mathbf{A} = \lambda \boldsymbol{\Phi}(n-1)$，$\mathbf{u} = \mathbf{v} = \mathbf{x}(n)$：
$$\mathbf{P}(n) = \frac{1}{\lambda}\left[\mathbf{P}(n-1) - \frac{\mathbf{P}(n-1)\mathbf{x}(n)\mathbf{x}^T(n)\mathbf{P}(n-1)}{\lambda + \mathbf{x}^T(n)\mathbf{P}(n-1)\mathbf{x}(n)}\right]$$

### 3.4 增益向量

定义**增益向量**：
$$\mathbf{k}(n) = \frac{\mathbf{P}(n-1)\mathbf{x}(n)}{\lambda + \mathbf{x}^T(n)\mathbf{P}(n-1)\mathbf{x}(n)}$$

则$\mathbf{P}(n)$的更新简化为：
$$\mathbf{P}(n) = \frac{1}{\lambda}[\mathbf{P}(n-1) - \mathbf{k}(n)\mathbf{x}^T(n)\mathbf{P}(n-1)]$$

或等价地：
$$\mathbf{P}(n) = \frac{1}{\lambda}[\mathbf{I} - \mathbf{k}(n)\mathbf{x}^T(n)]\mathbf{P}(n-1)$$

### 3.5 权值更新

**先验误差**（使用旧权值）：
$$\xi(n) = d(n) - \mathbf{w}^T(n-1)\mathbf{x}(n)$$

**权值更新**：
$$\mathbf{w}(n) = \mathbf{w}(n-1) + \mathbf{k}(n)\xi(n)$$

**推导**：
$$\begin{aligned}
\mathbf{w}(n) &= \mathbf{P}(n)\boldsymbol{\theta}(n) \\
&= \mathbf{P}(n)[\lambda\boldsymbol{\theta}(n-1) + d(n)\mathbf{x}(n)] \\
&= \mathbf{P}(n)\lambda\boldsymbol{\Phi}(n-1)\mathbf{w}(n-1) + \mathbf{P}(n)d(n)\mathbf{x}(n)
\end{aligned}$$

利用$\mathbf{P}(n)\boldsymbol{\Phi}(n) = \mathbf{I}$和相关恒等式，可得上述更新公式。

---

## 4. RLS算法总结

### 4.1 算法流程

```
初始化:
    w(0) = 0
    P(0) = δ⁻¹I  (δ为小正数，如0.01)

对于 n = 1, 2, 3, ...：
    1. 计算增益向量:
       k(n) = P(n-1)x(n) / [λ + x^T(n)P(n-1)x(n)]
    
    2. 计算先验误差:
       ξ(n) = d(n) - w^T(n-1)x(n)
    
    3. 更新权值:
       w(n) = w(n-1) + k(n)ξ(n)
    
    4. 更新逆相关矩阵:
       P(n) = λ⁻¹[P(n-1) - k(n)x^T(n)P(n-1)]
```

### 4.2 公式汇总

| 步骤 | 公式 |
|------|------|
| 增益向量 | $\mathbf{k}(n) = \frac{\mathbf{P}(n-1)\mathbf{x}(n)}{\lambda + \mathbf{x}^T(n)\mathbf{P}(n-1)\mathbf{x}(n)}$ |
| 先验误差 | $\xi(n) = d(n) - \mathbf{w}^T(n-1)\mathbf{x}(n)$ |
| 权值更新 | $\mathbf{w}(n) = \mathbf{w}(n-1) + \mathbf{k}(n)\xi(n)$ |
| 逆矩阵更新 | $\mathbf{P}(n) = \lambda^{-1}[\mathbf{P}(n-1) - \mathbf{k}(n)\mathbf{x}^T(n)\mathbf{P}(n-1)]$ |

### 4.3 计算复杂度

| 操作 | 乘法次数 |
|------|----------|
| $\mathbf{P}(n-1)\mathbf{x}(n)$ | $M^2$ |
| $\mathbf{x}^T(n)\mathbf{P}(n-1)\mathbf{x}(n)$ | $M$ |
| $\mathbf{k}(n)$ | $M$ |
| $\mathbf{w}(n)$ | $M$ |
| $\mathbf{P}(n)$ | $M^2$ |
| **总计** | $\approx 2M^2 + 4M$ |

每次迭代复杂度：$O(M^2)$

---

## 5. 收敛性分析

### 5.1 收敛速度

RLS的收敛速度与输入信号的自相关矩阵特征值分布**无关**，这是其相对于LMS的主要优势。

**收敛时间**：通常在$2M$到$3M$次迭代内收敛（$M$为滤波器阶数）。

### 5.2 稳态性能

在平稳环境下（$\lambda = 1$），RLS渐近达到最优Wiener解：
$$\lim_{n \to \infty} \mathbf{w}(n) = \mathbf{w}_{opt}$$

### 5.3 跟踪能力

对于时变系统，选择$\lambda < 1$可以跟踪系统变化：
- $\lambda$越小，跟踪越快，但稳态误差越大
- $\lambda$越大，稳态误差越小，但跟踪越慢

---

## 6. 数值稳定性

### 6.1 问题

RLS算法可能出现数值问题：
- $\mathbf{P}(n)$可能失去对称性
- $\mathbf{P}(n)$可能失去正定性
- 有限精度运算导致误差累积

### 6.2 改进方法

**1. 对称化**：
$$\mathbf{P}(n) \leftarrow \frac{1}{2}[\mathbf{P}(n) + \mathbf{P}^T(n)]$$

**2. 平方根RLS**：
维护$\mathbf{P}(n)$的Cholesky分解$\mathbf{P}(n) = \mathbf{S}(n)\mathbf{S}^T(n)$

**3. QR分解RLS**：
使用QR分解代替矩阵求逆

**4. 正则化**：
$$\mathbf{P}(n) = \lambda^{-1}[\mathbf{P}(n-1) - \mathbf{k}(n)\mathbf{x}^T(n)\mathbf{P}(n-1)] + \epsilon\mathbf{I}$$

---

## 7. RLS与LMS比较

| 特性 | LMS | RLS |
|------|-----|-----|
| 计算复杂度 | $O(M)$ | $O(M^2)$ |
| 存储需求 | $O(M)$ | $O(M^2)$ |
| 收敛速度 | 慢，依赖特征值分布 | 快，与特征值无关 |
| 稳态误差 | 较大 | 较小 |
| 数值稳定性 | 好 | 需要注意 |
| 跟踪能力 | 一般 | 好（通过$\lambda$调节） |
| 实现难度 | 简单 | 较复杂 |

### 选择建议

- **选择LMS**：计算资源有限、滤波器阶数大、对收敛速度要求不高
- **选择RLS**：需要快速收敛、输入信号相关性强、跟踪时变系统

---

## 8. RLS变体

### 8.1 滑动窗口RLS

只使用最近$L$个数据点：
$$J(n) = \sum_{i=n-L+1}^{n} |e(i)|^2$$

### 8.2 快速RLS算法

利用输入向量的移位结构，将复杂度降至$O(M)$：
- 快速横向RLS（FTF）
- 格型RLS

### 8.3 正则化RLS

添加正则化项防止过拟合：
$$J(n) = \sum_{i=1}^{n} \lambda^{n-i} |e(i)|^2 + \delta\|\mathbf{w}\|^2$$

---

## 9. 参考文献

1. Haykin, S. (2002). *Adaptive Filter Theory* (4th ed.). Prentice Hall.
2. Sayed, A. H. (2008). *Adaptive Filters*. Wiley-IEEE Press.
3. Ljung, L., & Söderström, T. (1983). *Theory and Practice of Recursive Identification*. MIT Press.
