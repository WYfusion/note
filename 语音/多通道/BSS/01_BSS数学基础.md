# BSS 数学基础

## 1. 混合模型

### 1.1 瞬时混合模型 (Instantaneous Mixing)

最简单的BSS模型，假设信号在同一时刻瞬时混合：

$$\mathbf{x}(t) = \mathbf{A}\mathbf{s}(t) + \mathbf{n}(t)$$

其中：
- $\mathbf{x}(t) \in \mathbb{R}^M$ - M个麦克风的观测信号
- $\mathbf{s}(t) \in \mathbb{R}^N$ - N个源信号
- $\mathbf{A} \in \mathbb{R}^{M \times N}$ - 混合矩阵
- $\mathbf{n}(t)$ - 噪声

**目标**：找到解混矩阵 $\mathbf{W}$，使得：
$$\hat{\mathbf{s}}(t) = \mathbf{W}\mathbf{x}(t) \approx \mathbf{P}\mathbf{D}\mathbf{s}(t)$$

其中 $\mathbf{P}$ 是排列矩阵，$\mathbf{D}$ 是对角缩放矩阵。

### 1.2 卷积混合模型 (Convolutive Mixing)

实际声学环境中，信号经过房间反射形成卷积混合：

$$x_i(t) = \sum_{j=1}^{N} \sum_{\tau=0}^{L-1} a_{ij}(\tau) s_j(t-\tau) + n_i(t)$$

或矩阵形式：
$$\mathbf{x}(t) = \sum_{\tau=0}^{L-1} \mathbf{A}(\tau) \mathbf{s}(t-\tau)$$

其中 $\mathbf{A}(\tau)$ 是第 $\tau$ 个延迟的混合矩阵，$L$ 是滤波器长度。

### 1.3 频域模型

通过短时傅里叶变换(STFT)，卷积变为乘法：

$$\mathbf{X}(f,t) = \mathbf{A}(f)\mathbf{S}(f,t)$$

其中：
- $f$ - 频率索引
- $t$ - 时间帧索引
- $\mathbf{A}(f) \in \mathbb{C}^{M \times N}$ - 频率 $f$ 处的混合矩阵

**优势**：每个频点独立处理，将卷积问题转化为多个瞬时混合问题。

---

## 2. 统计独立性

### 2.1 独立性定义

随机变量 $s_1, s_2, \ldots, s_N$ 相互独立，当且仅当其联合概率密度函数等于边缘密度函数的乘积：

$$p(s_1, s_2, \ldots, s_N) = \prod_{i=1}^{N} p_i(s_i)$$

### 2.2 独立性与不相关性

**不相关 (Uncorrelated)**：
$$E[s_i s_j] = E[s_i]E[s_j], \quad \forall i \neq j$$

**独立 (Independent)**：
$$E[g(s_i)h(s_j)] = E[g(s_i)]E[h(s_j)], \quad \forall g, h$$

> **关键区别**：独立性比不相关性更强。不相关只涉及二阶统计量，而独立性涉及所有阶统计量。

### 2.3 高斯分布的特殊性

对于高斯分布，不相关等价于独立。这也是为什么ICA要求源信号**最多只有一个是高斯分布**。

**证明**：设 $(s_1, s_2)$ 服从联合高斯分布，协方差矩阵为：
$$\mathbf{\Sigma} = \begin{pmatrix} \sigma_1^2 & \rho\sigma_1\sigma_2 \\ \rho\sigma_1\sigma_2 & \sigma_2^2 \end{pmatrix}$$

当 $\rho = 0$（不相关）时，$\mathbf{\Sigma}$ 为对角阵，联合密度可分解为边缘密度的乘积，即独立。

---

## 3. 高阶统计量

### 3.1 累积量 (Cumulants)

累积量是描述概率分布的统计量，与矩(moments)相关但不同。

**一阶累积量**（均值）：
$$\kappa_1 = E[x] = \mu$$

**二阶累积量**（方差）：
$$\kappa_2 = E[(x-\mu)^2] = \sigma^2$$

**三阶累积量**（偏度相关）：
$$\kappa_3 = E[(x-\mu)^3]$$

**四阶累积量**（峭度相关）：
$$\kappa_4 = E[(x-\mu)^4] - 3\sigma^4$$

### 3.2 峭度 (Kurtosis)

峭度衡量分布的"尖峰"程度：

$$\text{Kurt}(x) = \frac{E[(x-\mu)^4]}{\sigma^4} - 3 = \frac{\kappa_4}{\kappa_2^2}$$

**分布类型**：
- $\text{Kurt} = 0$：高斯分布（中峭度）
- $\text{Kurt} > 0$：超高斯分布（尖峰，如拉普拉斯分布）
- $\text{Kurt} < 0$：亚高斯分布（平坦，如均匀分布）

**语音信号**：通常是超高斯分布，峭度为正。

### 3.3 累积量的重要性质

**性质1：独立随机变量的累积量可加**
$$\kappa_n(x + y) = \kappa_n(x) + \kappa_n(y), \quad \text{若 } x, y \text{ 独立}$$

**性质2：高斯分布的高阶累积量为零**
$$\kappa_n = 0, \quad \forall n \geq 3 \text{ (高斯分布)}$$

**性质3：缩放性质**
$$\kappa_n(ax) = a^n \kappa_n(x)$$

---

## 4. 信息论基础

### 4.1 熵 (Entropy)

连续随机变量 $x$ 的微分熵：

$$H(x) = -\int p(x) \log p(x) dx$$

**性质**：在所有方差相同的分布中，高斯分布的熵最大。

### 4.2 互信息 (Mutual Information)

衡量两个随机变量之间的依赖程度：

$$I(x; y) = H(x) + H(y) - H(x, y)$$

$$I(x; y) = \int\int p(x,y) \log \frac{p(x,y)}{p(x)p(y)} dx dy$$

**性质**：$I(x; y) \geq 0$，等号成立当且仅当 $x, y$ 独立。

### 4.3 KL散度 (Kullback-Leibler Divergence)

衡量两个分布之间的"距离"：

$$D_{KL}(p \| q) = \int p(x) \log \frac{p(x)}{q(x)} dx$$

**性质**：
- $D_{KL}(p \| q) \geq 0$
- $D_{KL}(p \| q) = 0$ 当且仅当 $p = q$
- 不对称：$D_{KL}(p \| q) \neq D_{KL}(q \| p)$

### 4.4 负熵 (Negentropy)

衡量分布与高斯分布的偏离程度：

$$J(x) = H(x_{\text{gauss}}) - H(x)$$

其中 $x_{\text{gauss}}$ 是与 $x$ 具有相同方差的高斯随机变量。

**性质**：
- $J(x) \geq 0$
- $J(x) = 0$ 当且仅当 $x$ 是高斯分布
- 负熵是非高斯性的良好度量

**近似计算**：
$$J(x) \approx \frac{1}{12}E[x^3]^2 + \frac{1}{48}\text{Kurt}(x)^2$$

或使用非多项式函数：
$$J(x) \approx [E\{G(x)\} - E\{G(\nu)\}]^2$$

其中 $\nu \sim \mathcal{N}(0,1)$，$G$ 是非二次函数，如：
- $G_1(u) = \frac{1}{a}\log\cosh(au)$
- $G_2(u) = -\exp(-u^2/2)$

---

## 5. 预处理步骤

### 5.1 中心化 (Centering)

去除均值：
$$\tilde{\mathbf{x}} = \mathbf{x} - E[\mathbf{x}]$$

### 5.2 白化 (Whitening)

使数据不相关且方差为1：

**步骤**：
1. 计算协方差矩阵：$\mathbf{C} = E[\tilde{\mathbf{x}}\tilde{\mathbf{x}}^T]$
2. 特征值分解：$\mathbf{C} = \mathbf{E}\mathbf{D}\mathbf{E}^T$
3. 白化变换：$\mathbf{z} = \mathbf{D}^{-1/2}\mathbf{E}^T\tilde{\mathbf{x}}$

**白化后的性质**：
$$E[\mathbf{z}\mathbf{z}^T] = \mathbf{I}$$

**白化的好处**：
1. 降低问题维度（从 $M \times N$ 到 $N \times N$）
2. 解混矩阵变为正交矩阵
3. 加速收敛

### 5.3 白化后的ICA模型

白化后：
$$\mathbf{z} = \mathbf{V}\mathbf{x} = \mathbf{V}\mathbf{A}\mathbf{s} = \tilde{\mathbf{A}}\mathbf{s}$$

其中 $\tilde{\mathbf{A}} = \mathbf{V}\mathbf{A}$ 是正交矩阵（因为 $E[\mathbf{z}\mathbf{z}^T] = \tilde{\mathbf{A}}E[\mathbf{s}\mathbf{s}^T]\tilde{\mathbf{A}}^T = \tilde{\mathbf{A}}\tilde{\mathbf{A}}^T = \mathbf{I}$）。

因此，解混矩阵 $\mathbf{W}$ 也是正交矩阵，只需在正交矩阵空间中搜索。
