# ICA 独立成分分析

## 1. ICA 问题定义

### 1.1 基本模型

观测信号是源信号的线性混合：
$$\mathbf{x} = \mathbf{A}\mathbf{s}$$

**目标**：仅从观测 $\mathbf{x}$ 估计源信号 $\mathbf{s}$ 和混合矩阵 $\mathbf{A}$。

### 1.2 ICA 的基本假设

1. **源信号相互独立**：$p(\mathbf{s}) = \prod_{i=1}^{N} p_i(s_i)$
2. **最多一个源是高斯分布**
3. **混合矩阵 $\mathbf{A}$ 是方阵且可逆**（确定性情况）

### 1.3 ICA 的固有模糊性

1. **幅度模糊性**：无法确定 $s_i$ 的方差
2. **排列模糊性**：无法确定源的顺序
3. **符号模糊性**：无法确定 $s_i$ 的符号

---

## 2. FastICA 算法

### 2.1 算法思想

FastICA 基于**最大化非高斯性**原理。根据中心极限定理，独立随机变量的线性组合比原始变量更接近高斯分布。因此，最大化输出的非高斯性可以恢复独立源。

### 2.2 目标函数

使用负熵作为非高斯性度量：
$$J(y) = [E\{G(y)\} - E\{G(\nu)\}]^2$$

其中 $\nu \sim \mathcal{N}(0,1)$，$G$ 是非二次函数。

**常用的 $G$ 函数**：

| 函数 | $G(u)$ | $g(u) = G'(u)$ | $g'(u)$ |
|------|--------|----------------|---------|
| logcosh | $\frac{1}{a}\log\cosh(au)$ | $\tanh(au)$ | $a(1-\tanh^2(au))$ |
| exp | $-\exp(-u^2/2)$ | $u\exp(-u^2/2)$ | $(1-u^2)\exp(-u^2/2)$ |
| kurtosis | $u^4/4$ | $u^3$ | $3u^2$ |

### 2.3 优化问题

寻找投影方向 $\mathbf{w}$，使得 $y = \mathbf{w}^T\mathbf{z}$ 的非高斯性最大：

$$\max_{\mathbf{w}} J(\mathbf{w}^T\mathbf{z}) \quad \text{s.t.} \quad \|\mathbf{w}\| = 1$$

### 2.4 定点迭代推导

使用拉格朗日乘数法：
$$\mathcal{L}(\mathbf{w}, \lambda) = E\{G(\mathbf{w}^T\mathbf{z})\} - \lambda(\|\mathbf{w}\|^2 - 1)$$

对 $\mathbf{w}$ 求导并令其为零：
$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = E\{\mathbf{z}g(\mathbf{w}^T\mathbf{z})\} - 2\lambda\mathbf{w} = 0$$

使用牛顿法求解，得到更新公式：

$$\mathbf{w}^+ = E\{\mathbf{z}g(\mathbf{w}^T\mathbf{z})\} - E\{g'(\mathbf{w}^T\mathbf{z})\}\mathbf{w}$$

**归一化**：
$$\mathbf{w} \leftarrow \frac{\mathbf{w}^+}{\|\mathbf{w}^+\|}$$

### 2.5 FastICA 算法流程

**单个独立成分提取**：

```
输入: 白化数据 Z, 非线性函数 g, g'
输出: 解混向量 w

1. 随机初始化 w，归一化 ||w|| = 1
2. repeat
     w_new = E{z·g(w^T z)} - E{g'(w^T z)}·w
     w_new = w_new / ||w_new||
   until |w_new^T w| ≈ 1 (收敛)
3. return w
```

**多个独立成分提取（去相关）**：

提取第 $p$ 个成分时，需要与前 $p-1$ 个成分正交：

$$\mathbf{w}_p \leftarrow \mathbf{w}_p - \sum_{j=1}^{p-1}(\mathbf{w}_p^T\mathbf{w}_j)\mathbf{w}_j$$

**对称正交化**（同时提取所有成分）：

$$\mathbf{W} \leftarrow (\mathbf{W}\mathbf{W}^T)^{-1/2}\mathbf{W}$$

### 2.6 完整 FastICA 算法

```
输入: 观测数据 X ∈ R^(M×T), 源数量 N
输出: 解混矩阵 W, 估计源 S

1. 中心化: X = X - mean(X)
2. 白化: Z = V·X, 其中 V 使 E[ZZ^T] = I
3. 初始化 W ∈ R^(N×N) 为随机正交矩阵
4. repeat
     for p = 1 to N:
       w_p = E{z·g(w_p^T z)} - E{g'(w_p^T z)}·w_p
     end
     对称正交化: W = (WW^T)^(-1/2) W
   until W 收敛
5. S = W·Z
6. return W, S
```

### 2.7 收敛性分析

FastICA 具有**三次收敛速度**（cubic convergence），比梯度下降快得多。

设 $\mathbf{w}^*$ 是最优解，误差 $\epsilon_k = \|\mathbf{w}_k - \mathbf{w}^*\|$，则：
$$\epsilon_{k+1} = O(\epsilon_k^3)$$

---

## 3. InfoMax 算法

### 3.1 算法思想

InfoMax 基于**最大化输出熵**原理。通过最大化神经网络输出的信息量来实现分离。

### 3.2 模型结构

```
x → W → u = Wx → g(·) → y = g(u)
```

其中 $g(\cdot)$ 是非线性激活函数（通常是 sigmoid）。

### 3.3 目标函数

最大化输出 $\mathbf{y}$ 的熵：
$$H(\mathbf{y}) = H(\mathbf{u}) + E[\log|\det \mathbf{J}|]$$

其中 $\mathbf{J}$ 是 $g$ 的雅可比矩阵。

对于逐元素非线性 $y_i = g(u_i)$：
$$H(\mathbf{y}) = H(\mathbf{u}) + \sum_i E[\log|g'(u_i)|]$$

### 3.4 与 ICA 的等价性

**关键定理**：当 $g$ 是源分布 CDF 的导数时，最大化 $H(\mathbf{y})$ 等价于最小化输出的互信息。

对于超高斯源（如语音），$g(u) = \tanh(u)$ 是合适的选择。

### 3.5 梯度推导

目标函数：
$$\mathcal{L}(\mathbf{W}) = \log|\det\mathbf{W}| + \sum_i E[\log g'(\mathbf{w}_i^T\mathbf{x})]$$

对 $\mathbf{W}$ 求梯度：
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = (\mathbf{W}^T)^{-1} + E[\boldsymbol{\phi}(\mathbf{u})\mathbf{x}^T]$$

其中 $\boldsymbol{\phi}(\mathbf{u}) = [\phi(u_1), \ldots, \phi(u_N)]^T$，$\phi(u) = \frac{g''(u)}{g'(u)}$。

对于 $g(u) = \tanh(u)$：
$$\phi(u) = -2\tanh(u)$$

### 3.6 自然梯度

标准梯度在参数空间中不是最优的。**自然梯度**考虑了参数空间的黎曼几何结构：

$$\tilde{\nabla}_{\mathbf{W}}\mathcal{L} = \frac{\partial \mathcal{L}}{\partial \mathbf{W}}\mathbf{W}^T\mathbf{W}$$

**自然梯度更新**：
$$\Delta\mathbf{W} = \eta[\mathbf{I} + \boldsymbol{\phi}(\mathbf{u})\mathbf{u}^T]\mathbf{W}$$

其中 $\mathbf{u} = \mathbf{W}\mathbf{x}$。

### 3.7 InfoMax 算法流程

```
输入: 观测数据 X, 学习率 η
输出: 解混矩阵 W

1. 中心化 X
2. 初始化 W 为单位矩阵或随机矩阵
3. for each mini-batch {x_1, ..., x_B}:
     u = W·x  (对每个样本)
     ΔW = η·[I + φ(u)·u^T]·W  (自然梯度)
     W = W + ΔW
   end
4. 重复步骤3直到收敛
5. return W
```

### 3.8 扩展 InfoMax (Extended InfoMax)

标准 InfoMax 假设源是超高斯分布。**扩展 InfoMax** 可以处理超高斯和亚高斯混合：

$$\phi_i(u_i) = -2\tanh(u_i) + k_i u_i$$

其中 $k_i$ 根据估计的峭度自适应选择：
- $k_i = 1$：超高斯
- $k_i = -1$：亚高斯

---

## 4. 频域 ICA

### 4.1 频域模型

对于卷积混合，在频域中：
$$\mathbf{X}(f,t) = \mathbf{A}(f)\mathbf{S}(f,t)$$

每个频点独立应用 ICA：
$$\hat{\mathbf{S}}(f,t) = \mathbf{W}(f)\mathbf{X}(f,t)$$

### 4.2 排列问题

不同频点的 ICA 独立进行，导致输出顺序可能不一致。

**示例**：
- 频点 $f_1$：$[\hat{s}_1, \hat{s}_2]$ 对应 $[s_1, s_2]$
- 频点 $f_2$：$[\hat{s}_1, \hat{s}_2]$ 对应 $[s_2, s_1]$（排列错误）

### 4.3 排列问题解决方法

**方法1：基于包络相关性**

假设同一源在不同频点的包络相关：
$$\rho_{ij}(f_1, f_2) = \text{corr}(|\hat{S}_i(f_1,t)|, |\hat{S}_j(f_2,t)|)$$

通过最大化相邻频点的包络相关性来对齐。

**方法2：基于 DOA**

利用解混矩阵估计 DOA，相同源的 DOA 应该一致：
$$\theta_i(f) = \arcsin\left(\frac{c \cdot \angle[W^{-1}(f)]_{1i}}{2\pi f d}\right)$$

**方法3：IVA（见后续章节）**

---

## 5. 复数域 ICA

### 5.1 复数信号模型

频域信号是复数：
$$\mathbf{X}(f,t) \in \mathbb{C}^M$$

### 5.2 复数非线性函数

对于复数 $u = |u|e^{j\theta}$，常用的非线性函数：

$$g(u) = \frac{u}{|u|} \cdot \tilde{g}(|u|)$$

其中 $\tilde{g}$ 是实数非线性函数，如 $\tilde{g}(r) = \tanh(r)$。

### 5.3 复数自然梯度

$$\Delta\mathbf{W} = \eta[\mathbf{I} - E\{\boldsymbol{\phi}(\mathbf{u})\mathbf{u}^H\}]\mathbf{W}$$

其中 $\boldsymbol{\phi}(\mathbf{u}) = [g(u_1)/u_1, \ldots, g(u_N)/u_N]^T$。

---

## 6. 算法比较

| 特性 | FastICA | InfoMax |
|------|---------|---------|
| 收敛速度 | 三次收敛（快） | 线性收敛（慢） |
| 批处理/在线 | 批处理 | 可在线 |
| 内存需求 | 较高 | 较低 |
| 超参数 | 非线性函数选择 | 学习率 |
| 适用场景 | 离线处理 | 实时/大数据 |
