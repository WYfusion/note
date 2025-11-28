# ILRMA 独立低秩矩阵分析

## 1. 算法概述

### 1.1 背景

**ILRMA (Independent Low-Rank Matrix Analysis)** 是一种结合了 IVA 和 NMF (Non-negative Matrix Factorization) 的 BSS 算法，由 Kitamura 等人于 2016 年提出。

### 1.2 核心思想

- **IVA**：假设源向量服从球形分布，频点间通过范数耦合
- **ILRMA**：假设每个源的功率谱具有**低秩结构**，用 NMF 建模

### 1.3 优势

- 更精确的源模型（低秩假设符合语音/音乐特性）
- 自动解决排列问题
- 分离质量通常优于 IVA

---

## 2. 数学模型

### 2.1 频域混合模型

$$\mathbf{X}(f,t) = \mathbf{A}(f)\mathbf{S}(f,t)$$

分离：
$$\hat{\mathbf{S}}(f,t) = \mathbf{W}(f)\mathbf{X}(f,t)$$

### 2.2 源模型

假设分离后的源信号服从**时变高斯分布**：

$$\hat{S}_n(f,t) \sim \mathcal{N}_c(0, \lambda_{nft})$$

其中 $\lambda_{nft}$ 是第 $n$ 个源在频点 $f$、时刻 $t$ 的方差。

### 2.3 低秩假设

**关键假设**：方差矩阵 $\boldsymbol{\Lambda}_n = [\lambda_{nft}]_{F \times T}$ 具有低秩结构：

$$\lambda_{nft} = \sum_{k=1}^{K} b_{nfk} \cdot h_{nkt}$$

或矩阵形式：
$$\boldsymbol{\Lambda}_n = \mathbf{B}_n \mathbf{H}_n$$

其中：
- $\mathbf{B}_n \in \mathbb{R}_+^{F \times K}$：基矩阵（频谱模板）
- $\mathbf{H}_n \in \mathbb{R}_+^{K \times T}$：激活矩阵（时间激活）
- $K$：秩（基的数量）

### 2.4 物理意义

- **基矩阵 $\mathbf{B}_n$**：源 $n$ 的频谱特征模板
- **激活矩阵 $\mathbf{H}_n$**：每个模板随时间的激活强度
- **低秩**：语音/音乐的频谱可以用少量模板的组合表示

---

## 3. 目标函数

### 3.1 负对数似然

假设各源独立，各时频点独立：

$$\mathcal{L} = \sum_{f,t,n} \left( \log\lambda_{nft} + \frac{|\hat{S}_n(f,t)|^2}{\lambda_{nft}} \right) - 2T\sum_f \log|\det\mathbf{W}(f)|$$

### 3.2 代入 NMF 模型

$$\mathcal{L} = \sum_{f,t,n} \left( \log\sum_k b_{nfk}h_{nkt} + \frac{|\hat{S}_n(f,t)|^2}{\sum_k b_{nfk}h_{nkt}} \right) - 2T\sum_f \log|\det\mathbf{W}(f)|$$

### 3.3 优化变量

需要优化：
- 解混矩阵 $\mathbf{W}(f)$，$f = 1, \ldots, F$
- 基矩阵 $\mathbf{B}_n$，$n = 1, \ldots, N$
- 激活矩阵 $\mathbf{H}_n$，$n = 1, \ldots, N$

---

## 4. 优化算法

### 4.1 交替优化策略

ILRMA 使用**交替优化**：
1. 固定 $\mathbf{B}_n, \mathbf{H}_n$，更新 $\mathbf{W}(f)$
2. 固定 $\mathbf{W}(f)$，更新 $\mathbf{B}_n, \mathbf{H}_n$

### 4.2 更新 W(f) - IP 方法

类似 AuxIVA，使用辅助函数方法。

定义加权协方差：
$$\mathbf{V}_n(f) = \frac{1}{T}\sum_{t=1}^{T} \frac{\mathbf{x}(f,t)\mathbf{x}^H(f,t)}{\lambda_{nft}}$$

**IP 更新**：
$$\mathbf{w}_n(f) \leftarrow (\mathbf{W}(f)\mathbf{V}_n(f))^{-1}\mathbf{e}_n$$
$$\mathbf{w}_n(f) \leftarrow \frac{\mathbf{w}_n(f)}{\sqrt{\mathbf{w}_n^H(f)\mathbf{V}_n(f)\mathbf{w}_n(f)}}$$

### 4.3 更新 B, H - 乘法更新规则

使用 NMF 的乘法更新规则（保证非负性）。

定义功率谱：
$$P_{nft} = |\hat{S}_n(f,t)|^2$$

**更新 $\mathbf{H}_n$**：
$$h_{nkt} \leftarrow h_{nkt} \sqrt{\frac{\sum_f b_{nfk} P_{nft} / \lambda_{nft}^2}{\sum_f b_{nfk} / \lambda_{nft}}}$$

**更新 $\mathbf{B}_n$**：
$$b_{nfk} \leftarrow b_{nfk} \sqrt{\frac{\sum_t h_{nkt} P_{nft} / \lambda_{nft}^2}{\sum_t h_{nkt} / \lambda_{nft}}}$$

### 4.4 推导乘法更新规则

对于 $h_{nkt}$，目标函数对其的梯度：

$$\frac{\partial \mathcal{L}}{\partial h_{nkt}} = \sum_f \left( \frac{b_{nfk}}{\lambda_{nft}} - \frac{b_{nfk} P_{nft}}{\lambda_{nft}^2} \right)$$

令梯度为零：
$$\sum_f \frac{b_{nfk}}{\lambda_{nft}} = \sum_f \frac{b_{nfk} P_{nft}}{\lambda_{nft}^2}$$

乘法更新规则保证非负性并满足 KKT 条件。

---

## 5. ILRMA 算法流程

### 5.1 完整算法

```
输入: X(f,t) ∈ C^(M×T), 源数 N, 基数 K, 迭代次数
输出: W(f), B_n, H_n, Ŝ(f,t)

1. 初始化
   W(f) = I, ∀f
   B_n = rand(F, K), ∀n  (随机正数)
   H_n = rand(K, T), ∀n  (随机正数)

2. for iter = 1 to max_iter:
     
     // 计算分离信号
     Ŝ(f,t) = W(f)·X(f,t), ∀f,t
     
     // 计算功率谱
     P_n(f,t) = |Ŝ_n(f,t)|², ∀n,f,t
     
     // 计算方差模型
     λ_n(f,t) = Σ_k b_{nfk}·h_{nkt}, ∀n,f,t
     
     // 更新 NMF 参数
     for n = 1 to N:
       // 更新 H_n
       for k = 1 to K:
         for t = 1 to T:
           numer = Σ_f b_{nfk}·P_n(f,t)/λ_n(f,t)²
           denom = Σ_f b_{nfk}/λ_n(f,t)
           h_{nkt} *= sqrt(numer/denom)
         end
       end
       
       // 重新计算 λ
       λ_n(f,t) = Σ_k b_{nfk}·h_{nkt}
       
       // 更新 B_n
       for f = 1 to F:
         for k = 1 to K:
           numer = Σ_t h_{nkt}·P_n(f,t)/λ_n(f,t)²
           denom = Σ_t h_{nkt}/λ_n(f,t)
           b_{nfk} *= sqrt(numer/denom)
         end
       end
       
       // 重新计算 λ
       λ_n(f,t) = Σ_k b_{nfk}·h_{nkt}
     end
     
     // 更新解混矩阵 W(f)
     for n = 1 to N:
       for f = 1 to F:
         V_n(f) = (1/T) Σ_t x(f,t)·x^H(f,t)/λ_n(f,t)
         w_n(f) = (W(f)·V_n(f))^(-1)·e_n
         w_n(f) /= sqrt(w_n^H(f)·V_n(f)·w_n(f))
       end
     end
     
   end

3. // 投影回原始尺度
   for n = 1 to N:
     for f = 1 to F:
       Ŝ_n(f,:) *= (W^(-1)(f))_{:,n}·W(f)_{n,:}·X(f,:)
     end
   end

4. return W(f), B_n, H_n, Ŝ(f,t)
```

### 5.2 投影回原始尺度

分离后的信号存在尺度模糊性。使用**最小失真原则**：

$$\hat{S}_n^{\text{proj}}(f,t) = \mathbf{a}_n(f) \cdot \hat{S}_n(f,t)$$

其中：
$$\mathbf{a}_n(f) = (\mathbf{W}^{-1}(f))_{:,n}$$

---

## 6. ILRMA 与 IVA 的关系

### 6.1 源模型对比

| 方法 | 源模型 | 方差结构 |
|------|--------|----------|
| IVA | 球形拉普拉斯 | $\lambda_{nft} = \|\mathbf{s}_n(t)\|$ |
| ILRMA | 时变高斯 | $\lambda_{nft} = \sum_k b_{nfk}h_{nkt}$ |

### 6.2 特殊情况

当 $K = T$，$\mathbf{B}_n = \mathbf{I}$，$h_{nkt} = \delta_{kt}$ 时，ILRMA 退化为 IVA-G。

### 6.3 优势对比

| 特性 | IVA | ILRMA |
|------|-----|-------|
| 源模型精度 | 一般 | 高 |
| 参数数量 | 少 | 多 |
| 计算复杂度 | 低 | 高 |
| 分离质量 | 好 | 更好 |
| 适用场景 | 实时 | 离线 |

---

## 7. 参数选择

### 7.1 基数 K 的选择

- **K 太小**：无法充分表示源的频谱变化
- **K 太大**：过拟合，计算量增加

**经验值**：
- 语音：$K = 2 \sim 10$
- 音乐：$K = 10 \sim 50$

### 7.2 初始化策略

**随机初始化**：
$$b_{nfk} \sim \text{Uniform}(0, 1)$$
$$h_{nkt} \sim \text{Uniform}(0, 1)$$

**基于 PCA 初始化**：
使用观测信号的 PCA 初始化 $\mathbf{B}_n$。

### 7.3 收敛判据

$$\frac{|\mathcal{L}^{(k+1)} - \mathcal{L}^{(k)}|}{|\mathcal{L}^{(k)}|} < \epsilon$$

典型 $\epsilon = 10^{-6}$。

---

## 8. ILRMA 变体

### 8.1 t-ILRMA

使用学生 t 分布代替高斯分布，对异常值更鲁棒：

$$\hat{S}_n(f,t) \sim t_\nu(0, \lambda_{nft})$$

### 8.2 ILRMA-T

结合时间相关性：

$$\lambda_{nft} = \sum_k b_{nfk} \sum_{\tau} c_{k\tau} h_{nk,t-\tau}$$

### 8.3 Consistent ILRMA

添加一致性约束，确保分离信号可以通过 STFT 逆变换重建。

### 8.4 FastILRMA

使用近似计算加速：
- 子采样更新
- 低秩近似协方差

---

## 9. 计算复杂度

### 9.1 每次迭代复杂度

| 步骤 | 复杂度 |
|------|--------|
| 分离信号计算 | $O(FN^2T)$ |
| NMF 更新 | $O(FNKT)$ |
| 协方差计算 | $O(FN^2T)$ |
| IP 更新 | $O(FN^3)$ |
| **总计** | $O(FN^2T + FNKT)$ |

### 9.2 与 IVA 对比

ILRMA 比 IVA 多了 NMF 更新步骤，复杂度增加 $O(FNKT)$。

---

## 10. Python 实现

```python
import numpy as np

def ilrma(X, n_sources, n_bases=10, n_iter=100, eps=1e-8):
    """
    ILRMA 算法
    
    参数:
        X: 观测信号 [F, M, T]
        n_sources: 源数量 N
        n_bases: NMF 基数量 K
        n_iter: 迭代次数
    
    返回:
        S: 分离信号 [F, N, T]
        W: 解混矩阵 [F, N, M]
        B: 基矩阵 [N, F, K]
        H: 激活矩阵 [N, K, T]
    """
    F, M, T = X.shape
    N = n_sources
    K = n_bases
    
    # 初始化
    W = np.stack([np.eye(N, M, dtype=complex) for _ in range(F)])
    B = np.random.rand(N, F, K) + eps
    H = np.random.rand(N, K, T) + eps
    
    for iteration in range(n_iter):
        # 分离信号
        S = np.einsum('fnm,fmt->fnt', W, X)
        
        # 功率谱
        P = np.abs(S)**2  # [F, N, T]
        P = P.transpose(1, 0, 2)  # [N, F, T]
        
        # 方差模型 λ = B @ H
        Lambda = np.einsum('nfk,nkt->nft', B, H) + eps  # [N, F, T]
        
        # 更新 NMF 参数
        for n in range(N):
            # 更新 H
            numer_H = np.einsum('fk,ft->kt', B[n], P[n] / Lambda[n]**2)
            denom_H = np.einsum('fk,ft->kt', B[n], 1.0 / Lambda[n])
            H[n] *= np.sqrt(numer_H / (denom_H + eps))
            
            # 重新计算 Lambda
            Lambda[n] = B[n] @ H[n] + eps
            
            # 更新 B
            numer_B = np.einsum('kt,ft->fk', H[n], P[n] / Lambda[n]**2)
            denom_B = np.einsum('kt,ft->fk', H[n], 1.0 / Lambda[n])
            B[n] *= np.sqrt(numer_B / (denom_B + eps))
            
            # 重新计算 Lambda
            Lambda[n] = B[n] @ H[n] + eps
        
        # 更新 W (IP 方法)
        Lambda = Lambda.transpose(1, 0, 2)  # [F, N, T]
        for n in range(N):
            for f in range(F):
                # 加权协方差
                weight = 1.0 / Lambda[f, n, :]  # [T]
                V = np.einsum('mt,t,nt->mn', X[f], weight, X[f].conj()) / T
                
                # IP 更新
                WV = W[f] @ V
                w = np.linalg.solve(WV, np.eye(N)[:, n])
                w /= np.sqrt(w.conj() @ V @ w + eps)
                W[f, n, :] = w.conj()
    
    # 最终分离
    S = np.einsum('fnm,fmt->fnt', W, X)
    
    return S, W, B, H
```
