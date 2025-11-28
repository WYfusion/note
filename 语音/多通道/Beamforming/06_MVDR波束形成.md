# MVDR 波束形成

## 1. 概述

**MVDR (Minimum Variance Distortionless Response)** 波束形成器，也称为 **Capon 波束形成器**，是最经典的自适应波束形成方法之一。

### 1.1 核心思想

在保持目标方向信号无失真通过的约束下，最小化输出功率（方差），从而自适应地抑制噪声和干扰。

### 1.2 物理意义

- **无失真约束**：保证目标信号不被衰减或失真
- **最小方差**：最小化输出中的噪声和干扰成分
- **自适应性**：根据噪声统计特性自动调整权重

---

## 2. 数学推导

### 2.1 优化问题

$$\min_{\mathbf{w}} \mathbf{w}^H \mathbf{R}_X \mathbf{w} \quad \text{s.t.} \quad \mathbf{w}^H \mathbf{d} = 1$$

其中：
- $\mathbf{R}_X = E[\mathbf{X}\mathbf{X}^H]$：输入信号协方差矩阵
- $\mathbf{d} = \mathbf{d}(\theta_0, f)$：目标方向导向矢量
- 约束 $\mathbf{w}^H \mathbf{d} = 1$ 保证无失真

### 2.2 拉格朗日乘数法

构造拉格朗日函数：

$$\mathcal{L}(\mathbf{w}, \lambda) = \mathbf{w}^H \mathbf{R}_X \mathbf{w} + \lambda(\mathbf{w}^H \mathbf{d} - 1) + \lambda^*(\mathbf{d}^H \mathbf{w} - 1)$$

对 $\mathbf{w}^*$ 求导并令其为零：

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}^*} = \mathbf{R}_X \mathbf{w} + \lambda \mathbf{d} = 0$$

解得：

$$\mathbf{w} = -\lambda \mathbf{R}_X^{-1} \mathbf{d}$$

### 2.3 确定拉格朗日乘数

将上式代入约束条件：

$$\mathbf{w}^H \mathbf{d} = -\lambda^* \mathbf{d}^H \mathbf{R}_X^{-1} \mathbf{d} = 1$$

$$\lambda^* = -\frac{1}{\mathbf{d}^H \mathbf{R}_X^{-1} \mathbf{d}}$$

### 2.4 MVDR 解

$$\mathbf{w}_{\text{MVDR}} = \frac{\mathbf{R}_X^{-1} \mathbf{d}}{\mathbf{d}^H \mathbf{R}_X^{-1} \mathbf{d}}$$

这是 MVDR 波束形成器的**闭式解**。

---

## 3. 信号加噪声模型

### 3.1 协方差矩阵分解

假设信号和噪声不相关：

$$\mathbf{R}_X = \mathbf{R}_S + \mathbf{R}_N = \sigma_s^2 \mathbf{d}\mathbf{d}^H + \mathbf{R}_N$$

其中：
- $\sigma_s^2$：信号功率
- $\mathbf{R}_N$：噪声协方差矩阵

### 3.2 矩阵求逆引理

使用 Woodbury 矩阵恒等式：

$$(\mathbf{A} + \mathbf{uv}^H)^{-1} = \mathbf{A}^{-1} - \frac{\mathbf{A}^{-1}\mathbf{uv}^H\mathbf{A}^{-1}}{1 + \mathbf{v}^H\mathbf{A}^{-1}\mathbf{u}}$$

应用到 $\mathbf{R}_X^{-1}$：

$$\mathbf{R}_X^{-1} = \mathbf{R}_N^{-1} - \frac{\sigma_s^2 \mathbf{R}_N^{-1}\mathbf{d}\mathbf{d}^H\mathbf{R}_N^{-1}}{1 + \sigma_s^2 \mathbf{d}^H\mathbf{R}_N^{-1}\mathbf{d}}$$

### 3.3 简化形式

当只有噪声时（$\sigma_s^2 = 0$）：

$$\mathbf{w}_{\text{MVDR}} = \frac{\mathbf{R}_N^{-1} \mathbf{d}}{\mathbf{d}^H \mathbf{R}_N^{-1} \mathbf{d}}$$

这表明 MVDR 主要依赖于**噪声协方差矩阵** $\mathbf{R}_N$。

---

## 4. 性能分析

### 4.1 输出信噪比

MVDR 的输出 SNR 为：

$$\text{SNR}_{\text{out}} = \frac{\sigma_s^2 |\mathbf{w}^H\mathbf{d}|^2}{\mathbf{w}^H\mathbf{R}_N\mathbf{w}}$$

对于 MVDR：

$$\text{SNR}_{\text{out}} = \sigma_s^2 \mathbf{d}^H \mathbf{R}_N^{-1} \mathbf{d}$$

### 4.2 阵列增益

$$AG_{\text{MVDR}} = \frac{\text{SNR}_{\text{out}}}{\text{SNR}_{\text{in}}} = \mathbf{d}^H \mathbf{R}_N^{-1} \mathbf{d}$$

这是所有满足无失真约束的波束形成器中**最大的阵列增益**。

### 4.3 最优性

MVDR 在以下意义下是最优的：
1. 在无失真约束下，输出功率最小
2. 在无失真约束下，输出 SNR 最大
3. 等价于最大似然估计（在高斯假设下）

---

## 5. 实现细节

### 5.1 协方差矩阵估计

**时间平均**：

$$\hat{\mathbf{R}}_X = \frac{1}{T}\sum_{t=1}^{T} \mathbf{X}(t)\mathbf{X}^H(t)$$

**指数加权**：

$$\hat{\mathbf{R}}_X(t) = \alpha \hat{\mathbf{R}}_X(t-1) + (1-\alpha) \mathbf{X}(t)\mathbf{X}^H(t)$$

其中 $\alpha \in (0,1)$ 是遗忘因子。

### 5.2 噪声协方差估计

**方法1：语音活动检测（VAD）**

在语音不活动段估计：

$$\hat{\mathbf{R}}_N = \frac{1}{T_{\text{noise}}}\sum_{t \in \text{noise}} \mathbf{X}(t)\mathbf{X}^H(t)$$

**方法2：最小统计**

假设噪声功率在短时窗内最小：

$$\hat{\mathbf{R}}_N(f) = \min_{t \in [t-W, t]} \mathbf{X}(f,t)\mathbf{X}^H(f,t)$$

**方法3：递归更新**

$$\hat{\mathbf{R}}_N(t) = \beta \hat{\mathbf{R}}_N(t-1) + (1-\beta) \mathbf{X}(t)\mathbf{X}^H(t)$$

在检测到噪声时更新。

### 5.3 数值稳定性

**对角加载 (Diagonal Loading)**：

$$\mathbf{R}_X \leftarrow \mathbf{R}_X + \epsilon \mathbf{I}$$

其中 $\epsilon$ 是小正数（如 $10^{-6}$），防止矩阵奇异。

**归一化**：

$$\mathbf{R}_X \leftarrow \frac{\mathbf{R}_X}{\text{tr}(\mathbf{R}_X)}$$

---

## 6. MVDR 算法流程

### 6.1 离线处理

```
输入: X(f,t) ∈ C^(P×T), 目标方向 θ₀
输出: Y(f,t) - 增强信号

1. 计算导向矢量
   d = steering_vector(θ₀, f, array_geometry)

2. 估计协方差矩阵
   R_X = (1/T) * Σ_t X(f,t) * X^H(f,t)
   
3. 对角加载
   R_X = R_X + ε*I

4. 计算 MVDR 权重
   w = (R_X^(-1) * d) / (d^H * R_X^(-1) * d)

5. 应用波束形成
   for t = 1 to T:
     Y(f,t) = w^H * X(f,t)
   end

6. return Y(f,t)
```

### 6.2 在线处理

```
初始化:
  R_X = ε*I
  α = 0.95  (遗忘因子)

for each frame t:
  // 更新协方差
  R_X = α*R_X + (1-α)*X(t)*X^H(t)
  
  // 计算权重
  w = (R_X^(-1) * d) / (d^H * R_X^(-1) * d)
  
  // 波束形成
  Y(t) = w^H * X(t)
end
```

---

## 7. Python 实现

```python
import numpy as np

def mvdr_beamformer(X, d, epsilon=1e-6):
    """
    MVDR 波束形成器
    
    参数:
        X: 输入信号 [P, T] (麦克风数, 时间帧数)
        d: 导向矢量 [P, 1]
        epsilon: 对角加载系数
    
    返回:
        Y: 输出信号 [T]
        w: MVDR 权重 [P]
    """
    P, T = X.shape
    
    # 估计协方差矩阵
    R_X = (X @ X.conj().T) / T
    
    # 对角加载
    R_X += epsilon * np.eye(P)
    
    # 计算 MVDR 权重
    R_inv_d = np.linalg.solve(R_X, d)
    w = R_inv_d / (d.conj().T @ R_inv_d)
    
    # 应用波束形成
    Y = w.conj().T @ X
    
    return Y, w


def mvdr_online(X, d, alpha=0.95, epsilon=1e-6):
    """
    在线 MVDR 波束形成器
    
    参数:
        X: 输入信号 [P, T]
        d: 导向矢量 [P]
        alpha: 遗忘因子
        epsilon: 对角加载系数
    
    返回:
        Y: 输出信号 [T]
    """
    P, T = X.shape
    
    # 初始化
    R_X = epsilon * np.eye(P, dtype=complex)
    Y = np.zeros(T, dtype=complex)
    
    for t in range(T):
        # 更新协方差
        x_t = X[:, t:t+1]
        R_X = alpha * R_X + (1 - alpha) * (x_t @ x_t.conj().T)
        
        # 计算权重
        R_inv_d = np.linalg.solve(R_X, d)
        w = R_inv_d / (d.conj().T @ R_inv_d)
        
        # 波束形成
        Y[t] = w.conj().T @ x_t
    
    return Y
```

---

## 8. 与其他方法对比

| 特性 | 延迟求和 | MVDR | GSC |
|------|----------|------|-----|
| 自适应性 | 无 | 有 | 有 |
| 需要噪声统计 | 否 | 是 | 是 |
| 计算复杂度 | 低 | 中 | 高 |
| 鲁棒性 | 高 | 中 | 中 |
| 干扰抑制 | 弱 | 强 | 强 |
| 白噪声增益 | 高 | 可能低 | 可调 |

---

## 9. 优势与局限

### 9.1 优势

1. **最优性**：在无失真约束下最大化 SNR
2. **自适应**：自动适应噪声环境
3. **理论完备**：有严格的数学推导
4. **实现简单**：闭式解，易于实现

### 9.2 局限

1. **需要准确的导向矢量**：对 DOA 误差敏感
2. **需要噪声协方差**：估计不准确会降低性能
3. **白噪声增益可能低**：在低 SNR 时可能放大噪声
4. **计算量**：需要矩阵求逆

---

## 10. 改进方法

### 10.1 鲁棒 MVDR

添加不确定性约束：

$$\min_{\mathbf{w}} \mathbf{w}^H \mathbf{R}_X \mathbf{w} \quad \text{s.t.} \quad |\mathbf{w}^H \mathbf{d} - 1| \leq \delta$$

### 10.2 对角加载 MVDR

$$\mathbf{w} = \frac{(\mathbf{R}_X + \mu\mathbf{I})^{-1} \mathbf{d}}{\mathbf{d}^H (\mathbf{R}_X + \mu\mathbf{I})^{-1} \mathbf{d}}$$

其中 $\mu$ 根据白噪声增益约束选择。

### 10.3 子空间 MVDR

利用信号子空间和噪声子空间的正交性。
