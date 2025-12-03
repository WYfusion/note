# JADE 算法

## 1. 算法概述

JADE (Joint Approximate Diagonalization of Eigenmatrices) 是一种基于**四阶累积量联合对角化**的 ICA 算法，由 Cardoso 和 Souloumiac 于 1993 年提出。

### 1.1 核心思想

- 利用四阶累积量（与峭度相关）来度量非高斯性
- 通过联合对角化多个累积量矩阵来估计解混矩阵
- 代数方法，无需迭代优化

### 1.2 优势

- 无需选择非线性函数
- 无需设置学习率
- 收敛性有理论保证
- 对初始化不敏感

---

## 2. 四阶累积量

### 2.1 定义

四阶累积量（cumulant）定义为：

$$\text{cum}(x_i, x_j, x_k, x_l) = E[x_i x_j x_k x_l] - E[x_i x_j]E[x_k x_l] - E[x_i x_k]E[x_j x_l] - E[x_i x_l]E[x_j x_k]$$

对于零均值信号，简化为：
$$\text{cum}(x_i, x_j, x_k, x_l) = E[x_i x_j x_k x_l] - R_{ij}R_{kl} - R_{ik}R_{jl} - R_{il}R_{jk}$$

其中 $R_{ij} = E[x_i x_j]$ 是协方差。

### 2.2 累积量的性质

**性质1：多线性**
$$\text{cum}(ax_1 + by_1, x_2, x_3, x_4) = a\cdot\text{cum}(x_1, x_2, x_3, x_4) + b\cdot\text{cum}(y_1, x_2, x_3, x_4)$$

**性质2：对称性**
$$\text{cum}(x_i, x_j, x_k, x_l) = \text{cum}(x_{\pi(i)}, x_{\pi(j)}, x_{\pi(k)}, x_{\pi(l)})$$
对任意排列 $\pi$。

**性质3：独立变量的累积量**

若 $x$ 和 $y$ 独立，则：
$$\text{cum}(x+y, x+y, x+y, x+y) = \text{cum}(x,x,x,x) + \text{cum}(y,y,y,y)$$

**性质4：高斯变量的四阶累积量为零**
$$\text{cum}(g_1, g_2, g_3, g_4) = 0 \quad \text{若 } g_i \text{ 是高斯}$$

### 2.3 累积量张量

定义四阶累积量张量：
$$C_{ijkl} = \text{cum}(x_i, x_j, x_k, x_l)$$

这是一个四维张量，具有对称性。

---

## 3. 累积量矩阵

### 3.1 定义

对于任意矩阵 $\mathbf{M}$，定义累积量矩阵：

$$[\mathbf{Q}_{\mathbf{x}}(\mathbf{M})]_{ij} = \sum_{k,l} C_{ijkl} M_{kl}$$

或等价地：
$$\mathbf{Q}_{\mathbf{x}}(\mathbf{M}) = E[\mathbf{x}\mathbf{x}^T \cdot (\mathbf{x}^T\mathbf{M}\mathbf{x})] - \mathbf{R}\text{tr}(\mathbf{M}\mathbf{R}) - \mathbf{R}\mathbf{M}\mathbf{R} - \mathbf{R}\mathbf{M}^T\mathbf{R}$$

其中 $\mathbf{R} = E[\mathbf{x}\mathbf{x}^T]$ 是协方差矩阵。

### 3.2 白化数据的累积量矩阵

对于白化数据 $\mathbf{z}$（$E[\mathbf{z}\mathbf{z}^T] = \mathbf{I}$）：

$$\mathbf{Q}_{\mathbf{z}}(\mathbf{M}) = E[\mathbf{z}\mathbf{z}^T \cdot (\mathbf{z}^T\mathbf{M}\mathbf{z})] - \text{tr}(\mathbf{M})\mathbf{I} - \mathbf{M} - \mathbf{M}^T$$

### 3.3 关键性质

**性质：线性变换下的累积量矩阵**

若 $\mathbf{z} = \mathbf{B}\mathbf{s}$，则：
$$\mathbf{Q}_{\mathbf{z}}(\mathbf{M}) = \mathbf{B}\mathbf{Q}_{\mathbf{s}}(\mathbf{B}^T\mathbf{M}\mathbf{B})\mathbf{B}^T$$

---

## 4. JADE 算法推导

### 4.1 独立源的累积量矩阵

假设源信号 $\mathbf{s}$ 相互独立且方差为1，则：

$$\mathbf{Q}_{\mathbf{s}}(\mathbf{M}) = \text{diag}(\kappa_1 M_{11}, \kappa_2 M_{22}, \ldots, \kappa_N M_{NN})$$

其中 $\kappa_i = \text{cum}(s_i, s_i, s_i, s_i)$ 是第 $i$ 个源的峭度。

**证明**：由于源独立，交叉累积量为零：
$$\text{cum}(s_i, s_j, s_k, s_l) = 0 \quad \text{若 } i,j,k,l \text{ 不全相等}$$

因此只有对角元素非零。

### 4.2 观测数据的累积量矩阵

对于白化观测 $\mathbf{z} = \mathbf{U}\mathbf{s}$（$\mathbf{U}$ 是正交矩阵）：

$$\mathbf{Q}_{\mathbf{z}}(\mathbf{M}) = \mathbf{U}\mathbf{Q}_{\mathbf{s}}(\mathbf{U}^T\mathbf{M}\mathbf{U})\mathbf{U}^T$$

设 $\tilde{\mathbf{M}} = \mathbf{U}^T\mathbf{M}\mathbf{U}$，则：

$$\mathbf{Q}_{\mathbf{z}}(\mathbf{M}) = \mathbf{U}\text{diag}(\kappa_1 \tilde{M}_{11}, \ldots, \kappa_N \tilde{M}_{NN})\mathbf{U}^T$$

### 4.3 特征矩阵

选择 $\mathbf{M} = \mathbf{e}_i\mathbf{e}_j^T$（只有 $(i,j)$ 位置为1的矩阵），得到：

$$\mathbf{Q}_{\mathbf{z}}(\mathbf{e}_i\mathbf{e}_j^T) = \mathbf{U}\text{diag}(\kappa_1 U_{1i}U_{1j}, \ldots, \kappa_N U_{Ni}U_{Nj})\mathbf{U}^T$$

这些矩阵可以被 $\mathbf{U}$ **同时对角化**！

### 4.4 联合对角化

**目标**：找到正交矩阵 $\mathbf{U}$，使得一组矩阵 $\{\mathbf{Q}_1, \mathbf{Q}_2, \ldots, \mathbf{Q}_K\}$ 同时近似对角化。

**代价函数**：
$$\mathcal{J}(\mathbf{U}) = \sum_{k=1}^{K} \text{off}(\mathbf{U}^T\mathbf{Q}_k\mathbf{U})$$

其中 $\text{off}(\mathbf{A}) = \sum_{i \neq j} |A_{ij}|^2$ 是非对角元素的平方和。

**等价形式**（最大化对角元素）：
$$\mathcal{J}(\mathbf{U}) = \sum_{k=1}^{K} \|\text{diag}(\mathbf{U}^T\mathbf{Q}_k\mathbf{U})\|^2$$

---

## 5. Jacobi 旋转优化

### 5.1 Givens 旋转

使用一系列 Givens 旋转来逐步对角化：

$$\mathbf{G}_{pq}(\theta) = \begin{pmatrix} 
\mathbf{I} & & & \\
& \cos\theta & \cdots & \sin\theta \\
& \vdots & \ddots & \vdots \\
& -\sin\theta & \cdots & \cos\theta \\
& & & & \mathbf{I}
\end{pmatrix}$$

只在 $(p,q)$ 位置进行旋转。

### 5.2 最优旋转角度

对于一对索引 $(p,q)$，最优旋转角度 $\theta$ 满足：

$$\tan(4\theta) = \frac{2\sum_k (G_k^{pq} + G_k^{qp})(G_k^{pp} - G_k^{qq})}{\sum_k [(G_k^{pp} - G_k^{qq})^2 - (G_k^{pq} + G_k^{qp})^2]}$$

其中 $G_k^{ij}$ 是当前变换后矩阵 $\mathbf{Q}_k$ 的元素。

### 5.3 简化计算

定义：
$$h_k = G_k^{pp} - G_k^{qq}, \quad g_k = G_k^{pq} + G_k^{qp}$$

则：
$$\tan(4\theta) = \frac{2\sum_k g_k h_k}{\sum_k (h_k^2 - g_k^2)}$$

---

## 6. JADE 算法流程

### 6.1 完整算法

```
输入: 观测数据 X ∈ R^(M×T), 源数量 N
输出: 解混矩阵 W, 估计源 S

1. 预处理
   a. 中心化: X = X - mean(X)
   b. 白化: Z = V·X, 使 E[ZZ^T] = I

2. 构建累积量矩阵集合
   for i = 1 to N:
     for j = i to N:
       计算 Q_{ij} = Q_z(e_i·e_j^T)
     end
   end
   得到 K = N(N+1)/2 个矩阵

3. 联合对角化
   初始化 U = I
   repeat
     for p = 1 to N-1:
       for q = p+1 to N:
         计算最优旋转角度 θ
         更新 U = U·G_{pq}(θ)
         更新所有 Q_k = G_{pq}^T·Q_k·G_{pq}
       end
     end
   until 收敛 (off-diagonal 元素足够小)

4. 计算解混矩阵
   W = U^T·V

5. 估计源信号
   S = W·X

6. return W, S
```

### 6.2 计算复杂度

- 累积量矩阵计算：$O(N^4 T)$
- 联合对角化：$O(N^4 K)$ 每次迭代
- 总复杂度：$O(N^4 T + N^6 \cdot \text{iterations})$

### 6.3 实际优化

**减少矩阵数量**：
- 只使用特征值分解后的主要特征矩阵
- 典型选择 $K = N^2$ 或更少

**高效累积量计算**：
$$\mathbf{Q}_{\mathbf{z}}(\mathbf{M}) \approx \frac{1}{T}\sum_{t=1}^{T} \mathbf{z}_t\mathbf{z}_t^T (\mathbf{z}_t^T\mathbf{M}\mathbf{z}_t) - \text{tr}(\mathbf{M})\mathbf{I} - \mathbf{M} - \mathbf{M}^T$$

---

## 7. 数值示例

### 7.1 二维情况

设有两个独立源 $s_1, s_2$，峭度分别为 $\kappa_1, \kappa_2$。

白化后的累积量矩阵：

$$\mathbf{Q}(\mathbf{e}_1\mathbf{e}_1^T) = \mathbf{U}\begin{pmatrix} \kappa_1 U_{11}^2 & 0 \\ 0 & \kappa_2 U_{21}^2 \end{pmatrix}\mathbf{U}^T$$

$$\mathbf{Q}(\mathbf{e}_2\mathbf{e}_2^T) = \mathbf{U}\begin{pmatrix} \kappa_1 U_{12}^2 & 0 \\ 0 & \kappa_2 U_{22}^2 \end{pmatrix}\mathbf{U}^T$$

$$\mathbf{Q}(\mathbf{e}_1\mathbf{e}_2^T + \mathbf{e}_2\mathbf{e}_1^T) = \mathbf{U}\begin{pmatrix} 2\kappa_1 U_{11}U_{12} & 0 \\ 0 & 2\kappa_2 U_{21}U_{22} \end{pmatrix}\mathbf{U}^T$$

联合对角化这三个矩阵即可恢复 $\mathbf{U}$。

### 7.2 旋转角度计算

对于二维情况，设 $\mathbf{U} = \begin{pmatrix} \cos\phi & -\sin\phi \\ \sin\phi & \cos\phi \end{pmatrix}$。

最优角度 $\phi$ 可以通过求解：
$$\tan(4\phi) = \frac{2(\kappa_1 - \kappa_2)\sin(2\phi)\cos(2\phi)}{(\kappa_1 - \kappa_2)\cos^2(2\phi) - (\kappa_1 - \kappa_2)\sin^2(2\phi)}$$

简化为：
$$\tan(4\phi) = \tan(4\phi) \cdot \frac{\kappa_1 - \kappa_2}{\kappa_1 - \kappa_2}$$

当 $\kappa_1 \neq \kappa_2$ 时有唯一解。

---

## 8. JADE 的变体

### 8.1 JADE-TD (Time-Delayed)

使用时延协方差矩阵代替累积量矩阵：
$$\mathbf{R}(\tau) = E[\mathbf{x}(t)\mathbf{x}(t-\tau)^T]$$

适用于时间结构明显的信号。

### 8.2 JADE-OPAC

使用特征值分解预选择最重要的累积量矩阵，减少计算量。

### 8.3 复数 JADE

扩展到复数域，用于频域 BSS：
$$\text{cum}(z_i, z_j^*, z_k, z_l^*) = E[z_i z_j^* z_k z_l^*] - E[z_i z_j^*]E[z_k z_l^*] - E[z_i z_l^*]E[z_k z_j^*]$$
