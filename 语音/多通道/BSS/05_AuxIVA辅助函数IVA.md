# AuxIVA 辅助函数 IVA

## 1. 动机与背景

### 1.1 标准 IVA 的问题

标准 IVA 使用自然梯度下降，存在以下问题：
- 收敛速度慢
- 需要调节学习率
- 可能陷入局部最优

### 1.2 AuxIVA 的优势

**AuxIVA (Auxiliary-function-based IVA)** 使用辅助函数技术：
- **无需学习率**：每步更新都保证代价函数下降
- **快速收敛**：通常 10-50 次迭代即可收敛
- **稳定性好**：不会发散

---

## 2. 辅助函数方法

### 2.1 基本原理

对于难以直接优化的目标函数 $\mathcal{J}(\theta)$，构造辅助函数 $\mathcal{Q}(\theta, \theta')$ 满足：

1. **上界条件**：$\mathcal{Q}(\theta, \theta') \geq \mathcal{J}(\theta), \quad \forall \theta, \theta'$
2. **相切条件**：$\mathcal{Q}(\theta, \theta) = \mathcal{J}(\theta)$

### 2.2 优化策略

**迭代更新**：
$$\theta^{(k+1)} = \arg\min_\theta \mathcal{Q}(\theta, \theta^{(k)})$$

**保证单调下降**：
$$\mathcal{J}(\theta^{(k+1)}) \leq \mathcal{Q}(\theta^{(k+1)}, \theta^{(k)}) \leq \mathcal{Q}(\theta^{(k)}, \theta^{(k)}) = \mathcal{J}(\theta^{(k)})$$

---

## 3. AuxIVA 推导

### 3.1 IVA 代价函数回顾

$$\mathcal{J} = \sum_{t=1}^{T} \sum_{n=1}^{N} G(\|\mathbf{s}_n(t)\|) - T\sum_{f=1}^{F} \log|\det\mathbf{W}(f)|$$

其中 $G(r) = r$（球形拉普拉斯）或其他对比函数。

### 3.2 关键不等式

对于凹函数 $G(r)$，有：

$$G(r) \leq G(r') + G'(r')(r - r')$$

对于 $G(r) = r$：
$$r \leq r' + \frac{r^2 - r'^2}{2r'} = \frac{r^2}{2r'} + \frac{r'}{2}$$

### 3.3 辅助函数构造

定义辅助变量 $r_n^{(k)}(t) = \|\mathbf{s}_n^{(k)}(t)\|$（上一次迭代的范数）。

辅助函数：
$$\mathcal{Q} = \sum_{t,n} \frac{\|\mathbf{s}_n(t)\|^2}{2r_n^{(k)}(t)} + \text{const} - T\sum_f \log|\det\mathbf{W}(f)|$$

展开 $\|\mathbf{s}_n(t)\|^2$：
$$\|\mathbf{s}_n(t)\|^2 = \sum_f |\hat{S}_n(f,t)|^2 = \sum_f |\mathbf{w}_n^H(f)\mathbf{x}(f,t)|^2$$

其中 $\mathbf{w}_n(f)$ 是 $\mathbf{W}(f)$ 的第 $n$ 行（转置）。

### 3.4 加权协方差矩阵

定义加权协方差矩阵：
$$\mathbf{V}_n(f) = \frac{1}{T}\sum_{t=1}^{T} \frac{1}{r_n^{(k)}(t)} \mathbf{x}(f,t)\mathbf{x}^H(f,t)$$

则辅助函数变为：
$$\mathcal{Q} = \frac{T}{2}\sum_{f,n} \mathbf{w}_n^H(f)\mathbf{V}_n(f)\mathbf{w}_n(f) - T\sum_f \log|\det\mathbf{W}(f)| + \text{const}$$

### 3.5 最优解推导

对 $\mathbf{w}_n(f)$ 求导并令其为零，同时考虑约束。

使用 **IP (Iterative Projection)** 方法，逐个更新 $\mathbf{w}_n(f)$：

**步骤1**：计算
$$\mathbf{w}_n(f) \leftarrow (\mathbf{W}(f)\mathbf{V}_n(f))^{-1}\mathbf{e}_n$$

其中 $\mathbf{e}_n = [0,\ldots,0,1,0,\ldots,0]^T$（第 $n$ 个位置为1）。

**步骤2**：归一化
$$\mathbf{w}_n(f) \leftarrow \frac{\mathbf{w}_n(f)}{\sqrt{\mathbf{w}_n^H(f)\mathbf{V}_n(f)\mathbf{w}_n(f)}}$$

---

## 4. AuxIVA 算法

### 4.1 IP (Iterative Projection) 版本

```
输入: X(f,t) ∈ C^(M×T), f=1,...,F
输出: W(f), Ŝ(f,t)

1. 初始化
   for f = 1 to F:
     W(f) = I
   end

2. 迭代
   for iter = 1 to max_iter:
     
     // 计算分离信号
     for f = 1 to F:
       Ŝ(f,:) = W(f)·X(f,:)
     end
     
     // 计算源向量范数
     for n = 1 to N:
       r_n(t) = sqrt(Σ_f |Ŝ_n(f,t)|²), ∀t
     end
     
     // 更新每个源的解混向量
     for n = 1 to N:
       // 计算加权协方差
       for f = 1 to F:
         V_n(f) = (1/T) Σ_t [x(f,t)·x^H(f,t) / r_n(t)]
       end
       
       // IP 更新
       for f = 1 to F:
         w_n(f) = (W(f)·V_n(f))^(-1)·e_n
         w_n(f) = w_n(f) / sqrt(w_n^H(f)·V_n(f)·w_n(f))
         更新 W(f) 的第 n 行
       end
     end
     
   end

3. return W(f), Ŝ(f,t)
```

### 4.2 ISS (Iterative Source Steering) 版本

ISS 是一种更高效的更新方式，避免矩阵求逆。

**更新公式**：
$$\mathbf{w}_n(f) \leftarrow \mathbf{w}_n(f) - \frac{\mathbf{v}_n(f)}{\mathbf{V}_{nn}(f)}$$

其中：
$$\mathbf{v}_n(f) = \frac{1}{T}\sum_t \frac{\hat{S}_n(f,t)}{r_n(t)}\mathbf{x}(f,t)$$

**归一化**：
$$\mathbf{w}_n(f) \leftarrow \frac{\mathbf{w}_n(f)}{\sqrt{\mathbf{w}_n^H(f)\mathbf{V}_n(f)\mathbf{w}_n(f)}}$$

### 4.3 ISS 算法流程

```
输入: X(f,t), 最大迭代次数
输出: W(f), Ŝ(f,t)

1. 初始化 W(f) = I, ∀f

2. for iter = 1 to max_iter:
     
     // 分离
     Ŝ(f,t) = W(f)·X(f,t), ∀f,t
     
     // 范数
     r_n(t) = ||s_n(t)||, ∀n,t
     
     // ISS 更新
     for n = 1 to N:
       for f = 1 to F:
         // 计算 v_n(f)
         v_n(f) = (1/T) Σ_t [Ŝ_n(f,t)/r_n(t)]·x(f,t)
         
         // 计算 V_nn(f)
         V_nn(f) = (1/T) Σ_t |Ŝ_n(f,t)|²/r_n(t)
         
         // 更新
         w_n(f) -= v_n(f) / V_nn(f)
         
         // 归一化（简化版）
         w_n(f) /= sqrt(w_n^H(f)·V_n(f)·w_n(f))
       end
     end
     
   end

3. return W(f), Ŝ(f,t)
```

---

## 5. 收敛性分析

### 5.1 单调收敛保证

由辅助函数性质，每次迭代代价函数单调下降：
$$\mathcal{J}(\mathbf{W}^{(k+1)}) \leq \mathcal{J}(\mathbf{W}^{(k)})$$

### 5.2 收敛速度

实验表明，AuxIVA 通常在 **10-50 次迭代**内收敛，比自然梯度 IVA 快 5-10 倍。

### 5.3 收敛判据

**方法1**：代价函数变化
$$\frac{|\mathcal{J}^{(k+1)} - \mathcal{J}^{(k)}|}{|\mathcal{J}^{(k)}|} < \epsilon$$

**方法2**：解混矩阵变化
$$\max_f \|\mathbf{W}^{(k+1)}(f) - \mathbf{W}^{(k)}(f)\|_F < \epsilon$$

---

## 6. 不同对比函数

### 6.1 球形拉普拉斯 (Laplace)

$$G(r) = r$$
$$G'(r) = 1$$

权重：$\frac{1}{r_n(t)}$

### 6.2 球形高斯 (Gauss)

$$G(r) = r^2$$
$$G'(r) = 2r$$

权重：$\frac{1}{1}$（常数，退化为 PCA）

### 6.3 广义高斯

$$G(r) = r^\beta, \quad 0 < \beta \leq 2$$

权重：$\frac{\beta}{r_n^{2-\beta}(t)}$

### 6.4 对比函数选择

| 对比函数 | 适用信号 | 特点 |
|----------|----------|------|
| Laplace ($\beta=1$) | 语音、稀疏信号 | 最常用 |
| Gauss ($\beta=2$) | 高斯信号 | 退化为 PCA |
| $\beta < 1$ | 超稀疏信号 | 更强稀疏假设 |

---

## 7. 实现优化

### 7.1 批处理 vs 在线

**批处理**：使用所有数据计算协方差
$$\mathbf{V}_n(f) = \frac{1}{T}\sum_{t=1}^{T} \frac{\mathbf{x}(f,t)\mathbf{x}^H(f,t)}{r_n(t)}$$

**在线/块处理**：使用滑动窗口
$$\mathbf{V}_n^{(k)}(f) = \alpha\mathbf{V}_n^{(k-1)}(f) + (1-\alpha)\frac{\mathbf{x}(f,t)\mathbf{x}^H(f,t)}{r_n(t)}$$

### 7.2 数值稳定性

**问题**：$r_n(t) \approx 0$ 时除法不稳定。

**解决**：
$$r_n(t) \leftarrow \max(r_n(t), \epsilon)$$

或使用软阈值：
$$r_n(t) \leftarrow \sqrt{r_n^2(t) + \epsilon^2}$$

### 7.3 并行化

- 不同频点的更新可以并行
- 不同源的协方差计算可以并行
- GPU 加速矩阵运算

---

## 8. AuxIVA 变体

### 8.1 AuxIVA-IP

使用 Iterative Projection 更新，需要矩阵求逆。

**复杂度**：$O(FN^3T)$ 每次迭代

### 8.2 AuxIVA-ISS

使用 Iterative Source Steering，避免矩阵求逆。

**复杂度**：$O(FN^2T)$ 每次迭代

### 8.3 AuxIVA-IPA

Iterative Projection with Adjustment，结合两者优点。

### 8.4 FastAuxIVA

使用近似计算加速：
- 子采样时间帧
- 低秩近似协方差矩阵

---

## 9. 与其他方法对比

| 方法 | 收敛速度 | 每迭代复杂度 | 需要学习率 | 稳定性 |
|------|----------|--------------|------------|--------|
| 自然梯度 IVA | 慢 | $O(FN^2T)$ | 是 | 一般 |
| AuxIVA-IP | 快 | $O(FN^3T)$ | 否 | 好 |
| AuxIVA-ISS | 快 | $O(FN^2T)$ | 否 | 好 |
| FastIVA | 中等 | $O(FN^2T)$ | 是 | 一般 |

---

## 10. Python 实现示例

```python
import numpy as np

def auxiva_iss(X, n_iter=50, eps=1e-8):
    """
    AuxIVA-ISS 算法
    
    参数:
        X: 观测信号 [F, M, T] (频点, 麦克风, 时间帧)
        n_iter: 迭代次数
        eps: 数值稳定性常数
    
    返回:
        S: 分离信号 [F, N, T]
        W: 解混矩阵 [F, N, M]
    """
    F, M, T = X.shape
    N = M  # 假设源数等于麦克风数
    
    # 初始化
    W = np.stack([np.eye(N, M, dtype=complex) for _ in range(F)])
    
    for iteration in range(n_iter):
        # 分离信号
        S = np.einsum('fnm,fmt->fnt', W, X)  # [F, N, T]
        
        # 计算源向量范数
        r = np.sqrt(np.sum(np.abs(S)**2, axis=0) + eps)  # [N, T]
        
        # ISS 更新
        for n in range(N):
            # 加权
            weight = 1.0 / r[n]  # [T]
            
            for f in range(F):
                # v_n(f) = (1/T) * sum_t [S_n(f,t)/r_n(t)] * x(f,t)
                v = np.mean(S[f, n, :] / r[n] * X[f, :, :], axis=1)  # [M]
                
                # V_nn(f) = (1/T) * sum_t |S_n(f,t)|^2 / r_n(t)
                V_nn = np.mean(np.abs(S[f, n, :])**2 / r[n])
                
                # 更新
                W[f, n, :] -= v / (V_nn + eps)
                
                # 归一化
                S_new = W[f, n, :] @ X[f]
                scale = np.sqrt(np.mean(np.abs(S_new)**2 / r[n]))
                W[f, n, :] /= (scale + eps)
    
    # 最终分离
    S = np.einsum('fnm,fmt->fnt', W, X)
    
    return S, W
```
