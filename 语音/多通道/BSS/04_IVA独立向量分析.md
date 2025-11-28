# IVA 独立向量分析

## 1. 从 ICA 到 IVA

### 1.1 频域 ICA 的排列问题

在频域 BSS 中，每个频点独立应用 ICA：
$$\hat{\mathbf{S}}(f,t) = \mathbf{W}(f)\mathbf{X}(f,t)$$

**问题**：不同频点的分离结果顺序可能不一致。

**示例**：
```
频点 f1: [ŝ₁, ŝ₂] → [s₁, s₂]  ✓
频点 f2: [ŝ₁, ŝ₂] → [s₂, s₁]  ✗ (排列错误)
频点 f3: [ŝ₁, ŝ₂] → [s₁, s₂]  ✓
```

重建时域信号时，频谱会混乱。

### 1.2 IVA 的核心思想

**IVA (Independent Vector Analysis)** 将每个源视为一个**向量**（跨所有频点），而不是独立的标量。

定义源向量：
$$\mathbf{s}_n = [S_n(f_1,t), S_n(f_2,t), \ldots, S_n(f_F,t)]^T \in \mathbb{C}^F$$

**IVA 假设**：
- 不同源的向量 $\mathbf{s}_1, \mathbf{s}_2, \ldots, \mathbf{s}_N$ 相互独立
- 同一源的不同频点 $S_n(f_1,t), S_n(f_2,t), \ldots$ 可以相关

这种建模自然地解决了排列问题！

---

## 2. IVA 数学模型

### 2.1 多频点联合模型

对于所有频点：
$$\mathbf{X}(f,t) = \mathbf{A}(f)\mathbf{S}(f,t), \quad f = 1, \ldots, F$$

分离：
$$\hat{\mathbf{S}}(f,t) = \mathbf{W}(f)\mathbf{X}(f,t)$$

### 2.2 源向量定义

第 $n$ 个源在时刻 $t$ 的向量：
$$\mathbf{s}_n(t) = [\hat{S}_n(f_1,t), \hat{S}_n(f_2,t), \ldots, \hat{S}_n(f_F,t)]^T$$

### 2.3 独立性假设

**向量间独立**：
$$p(\mathbf{s}_1(t), \mathbf{s}_2(t), \ldots, \mathbf{s}_N(t)) = \prod_{n=1}^{N} p_n(\mathbf{s}_n(t))$$

**向量内依赖**：$p_n(\mathbf{s}_n)$ 是多变量分布，允许频点间相关。

---

## 3. IVA 目标函数

### 3.1 最大似然推导

假设源向量服从分布 $p_n(\mathbf{s}_n)$，观测的对数似然为：

$$\mathcal{L} = \sum_{t=1}^{T} \sum_{n=1}^{N} \log p_n(\mathbf{s}_n(t)) + T\sum_{f=1}^{F} \log|\det\mathbf{W}(f)|$$

### 3.2 源模型选择

**球形拉普拉斯分布**（最常用）：

$$p_n(\mathbf{s}_n) \propto \exp\left(-\|\mathbf{s}_n\|\right) = \exp\left(-\sqrt{\sum_{f=1}^{F}|S_n(f)|^2}\right)$$

这个分布：
- 假设频点间的依赖通过共同的幅度 $\|\mathbf{s}_n\|$ 建模
- 超高斯特性适合语音信号

**多变量高斯分布**：
$$p_n(\mathbf{s}_n) = \mathcal{N}(\mathbf{0}, \mathbf{\Sigma}_n)$$

允许更复杂的频点间相关结构。

### 3.3 代价函数

使用球形拉普拉斯分布，代价函数为：

$$\mathcal{J} = \sum_{t=1}^{T} \sum_{n=1}^{N} \|\mathbf{s}_n(t)\| - T\sum_{f=1}^{F} \log|\det\mathbf{W}(f)|$$

其中：
$$\|\mathbf{s}_n(t)\| = \sqrt{\sum_{f=1}^{F}|\hat{S}_n(f,t)|^2}$$

---

## 4. 自然梯度优化

### 4.1 梯度计算

对 $\mathbf{W}(f)$ 求梯度：

$$\frac{\partial \mathcal{J}}{\partial \mathbf{W}^*(f)} = \sum_{t=1}^{T} \boldsymbol{\phi}(f,t)\mathbf{X}^H(f,t) - T(\mathbf{W}^H(f))^{-1}$$

其中非线性函数：
$$\phi_n(f,t) = \frac{\partial \|\mathbf{s}_n(t)\|}{\partial \hat{S}_n^*(f,t)} = \frac{\hat{S}_n(f,t)}{\|\mathbf{s}_n(t)\|}$$

### 4.2 自然梯度

应用自然梯度变换：

$$\Delta\mathbf{W}(f) = \left[\mathbf{I} - E\{\boldsymbol{\phi}(f,t)\hat{\mathbf{S}}^H(f,t)\}\right]\mathbf{W}(f)$$

展开：
$$\Delta\mathbf{W}(f) = \left[\mathbf{I} - \frac{1}{T}\sum_{t=1}^{T}\boldsymbol{\phi}(f,t)\hat{\mathbf{S}}^H(f,t)\right]\mathbf{W}(f)$$

其中：
$$\boldsymbol{\phi}(f,t) = \left[\frac{\hat{S}_1(f,t)}{\|\mathbf{s}_1(t)\|}, \ldots, \frac{\hat{S}_N(f,t)}{\|\mathbf{s}_N(t)\|}\right]^T$$

### 4.3 更新规则

$$\mathbf{W}(f) \leftarrow \mathbf{W}(f) + \eta \cdot \Delta\mathbf{W}(f)$$

---

## 5. IVA 算法流程

### 5.1 标准 IVA 算法

```
输入: 多频点观测 X(f,t), f=1,...,F, t=1,...,T
      学习率 η, 收敛阈值 ε
输出: 解混矩阵 W(f), 分离信号 S(f,t)

1. 初始化
   for f = 1 to F:
     W(f) = I  (单位矩阵)
   end

2. 迭代优化
   repeat
     // 计算分离信号
     for f = 1 to F:
       Ŝ(f,t) = W(f)·X(f,t), ∀t
     end
     
     // 计算源向量范数
     for n = 1 to N:
       for t = 1 to T:
         r_n(t) = sqrt(Σ_f |Ŝ_n(f,t)|²)
       end
     end
     
     // 计算非线性函数
     for f = 1 to F:
       for t = 1 to T:
         φ_n(f,t) = Ŝ_n(f,t) / r_n(t), ∀n
       end
     end
     
     // 更新解混矩阵
     for f = 1 to F:
       ΔW(f) = [I - (1/T)Σ_t φ(f,t)·Ŝ^H(f,t)]·W(f)
       W(f) = W(f) + η·ΔW(f)
     end
     
   until 收敛

3. return W(f), Ŝ(f,t)
```

### 5.2 收敛判断

常用收敛准则：
$$\max_f \|\mathbf{W}^{(k+1)}(f) - \mathbf{W}^{(k)}(f)\|_F < \epsilon$$

或代价函数变化：
$$|\mathcal{J}^{(k+1)} - \mathcal{J}^{(k)}| < \epsilon$$

---

## 6. IVA 的理论分析

### 6.1 为什么 IVA 解决排列问题？

**关键洞察**：源向量范数 $\|\mathbf{s}_n(t)\|$ 将所有频点耦合在一起。

如果某个频点的排列错误，会导致：
- 错误源的频点被混入 $\mathbf{s}_n$
- $\|\mathbf{s}_n(t)\|$ 的统计特性改变
- 代价函数增大

因此，最小化代价函数自然倾向于正确的排列。

### 6.2 与 ICA 的关系

当频点间完全独立时（$p_n(\mathbf{s}_n) = \prod_f p_{n,f}(S_n(f))$），IVA 退化为独立的频域 ICA。

### 6.3 源模型的影响

| 源模型 | 频点依赖 | 适用场景 |
|--------|----------|----------|
| 球形拉普拉斯 | 通过范数耦合 | 语音、音乐 |
| 多变量高斯 | 协方差矩阵 | 一般信号 |
| 学生t分布 | 重尾+耦合 | 鲁棒分离 |

---

## 7. IVA 的变体

### 7.1 IVA-G (Gaussian IVA)

使用时变高斯模型：
$$p_n(\mathbf{s}_n(t)) = \mathcal{N}(\mathbf{0}, v_n(t)\mathbf{I})$$

其中 $v_n(t)$ 是时变方差。

**优势**：更灵活的源模型。

### 7.2 IVA-L (Laplace IVA)

标准球形拉普拉斯模型，即前述的 IVA。

### 7.3 OverIVA

处理过定情况（麦克风数 > 源数）：
$$\mathbf{X}(f,t) \in \mathbb{C}^M, \quad M > N$$

使用降维或正则化技术。

---

## 8. 实现细节

### 8.1 初始化策略

**方法1：单位矩阵**
$$\mathbf{W}(f) = \mathbf{I}$$

**方法2：PCA 初始化**
$$\mathbf{W}(f) = \mathbf{V}(f)^{-1}$$
其中 $\mathbf{V}(f)$ 是白化矩阵。

**方法3：频域 ICA + 排列对齐**
先用 ICA 分离，再用启发式方法对齐。

### 8.2 学习率选择

- 固定学习率：$\eta = 0.1 \sim 0.5$
- 自适应学习率：根据代价函数下降调整
- 线搜索：每步找最优步长

### 8.3 数值稳定性

**问题**：当 $\|\mathbf{s}_n(t)\| \approx 0$ 时，$\phi_n(f,t)$ 不稳定。

**解决**：添加小常数
$$\phi_n(f,t) = \frac{\hat{S}_n(f,t)}{\|\mathbf{s}_n(t)\| + \epsilon}$$

典型 $\epsilon = 10^{-8}$。

### 8.4 计算复杂度

每次迭代：
- 分离信号计算：$O(FN^2T)$
- 范数计算：$O(FNT)$
- 梯度更新：$O(FN^2T)$

总复杂度：$O(FN^2T \cdot \text{iterations})$

---

## 9. IVA vs ICA 对比

| 特性 | 频域 ICA | IVA |
|------|----------|-----|
| 排列问题 | 需要后处理 | 自动解决 |
| 频点处理 | 独立 | 联合 |
| 计算复杂度 | 较低 | 较高 |
| 源模型 | 单变量 | 多变量 |
| 收敛速度 | 快 | 较慢 |
| 分离质量 | 依赖排列对齐 | 通常更好 |

---

## 10. 应用示例

### 10.1 语音分离

```python
# 伪代码示例
def iva_separation(mixed_signals, n_sources, n_fft=1024):
    # STFT
    X = stft(mixed_signals, n_fft)  # [F, M, T]
    
    # 初始化
    W = [np.eye(n_sources) for _ in range(n_freq)]
    
    # IVA 迭代
    for iteration in range(max_iter):
        # 分离
        S = [W[f] @ X[f] for f in range(n_freq)]
        
        # 计算范数
        r = np.sqrt(np.sum(np.abs(S)**2, axis=0))  # [N, T]
        
        # 更新
        for f in range(n_freq):
            phi = S[f] / (r + eps)
            grad = np.eye(n_sources) - np.mean(phi @ S[f].conj().T, axis=-1)
            W[f] += learning_rate * grad @ W[f]
    
    # iSTFT
    separated = [istft(S[:, n, :]) for n in range(n_sources)]
    return separated
```
