# ESPRIT 算法

## 1. 概述

**ESPRIT (Estimation of Signal Parameters via Rotational Invariance Techniques)** 是一种基于旋转不变性的DOA估计算法，由Roy和Kailath于1989年提出。与MUSIC不同，ESPRIT无需角度搜索，直接通过特征值计算DOA。

### 1.1 核心思想

```
双重阵列结构
    ↓
[信号子空间估计]
    ↓
[利用旋转不变性]
    ↓
[求解广义特征值问题]
    ↓
DOA估计（无需搜索）
```

**关键性质**：相同信号在两个平移阵列上的信号子空间存在旋转关系。

---

## 2. 数学推导

### 2.1 阵列结构

**双重阵列**：

考虑两个相同的子阵列，一个相对于另一个有固定的位移：

```
子阵列1: 🎤 🎤 🎤 ... 🎤
子阵列2:   🎤 🎤 🎤 ... 🎤
          ←─ Δ ─→
```

**阵列配置**：
- 子阵列1：麦克风 $1, 2, \ldots, M$
- 子阵列2：麦克风 $2, 3, \ldots, M+1$
- 总麦克风数：$P = M + 1$

### 2.2 信号模型

**接收信号**：

$$\mathbf{X}(t) = \mathbf{D}\mathbf{s}(t) + \mathbf{N}(t)$$

其中导向矩阵：

$$\mathbf{D} = [\mathbf{d}(\theta_1), \mathbf{d}(\theta_2), \ldots, \mathbf{d}(\theta_K)]$$

**子阵列信号**：

$$\mathbf{X}_1(t) = \mathbf{D}_1\mathbf{s}(t) + \mathbf{N}_1(t)$$
$$\mathbf{X}_2(t) = \mathbf{D}_2\mathbf{s}(t) + \mathbf{N}_2(t)$$

其中：
- $\mathbf{X}_1(t) \in \mathbb{C}^M$：子阵列1的信号
- $\mathbf{X}_2(t) \in \mathbb{C}^M$：子阵列2的信号
- $\mathbf{D}_1, \mathbf{D}_2 \in \mathbb{C}^{M \times K}$：子阵列导向矩阵

### 2.3 旋转不变性

**关键关系**：

对于均匀线性阵列，两个子阵列的导向矩阵满足：

$$\mathbf{D}_2 = \mathbf{D}_1\mathbf{\Phi}$$

其中 $\mathbf{\Phi}$ 是**旋转算子**：

$$\mathbf{\Phi} = \text{diag}(e^{-jkd\sin\theta_1}, e^{-jkd\sin\theta_2}, \ldots, e^{-jkd\sin\theta_K})$$

**物理意义**：
- $d$：子阵列间距
- $k = 2\pi f/c$：波数
- $e^{-jkd\sin\theta_i}$：第$i$个源在两子阵列间的相位差

**推导**：

对于均匀线性阵列，第$i$个麦克风的导向向量元素为：

$$[\mathbf{d}(\theta)]_i = e^{-jk(i-1)d\sin\theta}$$

因此：
- 子阵列1的第$m$个元素：$e^{-jk(m-1)d\sin\theta}$
- 子阵列2的第$m$个元素：$e^{-jkmd\sin\theta} = e^{-jkd\sin\theta} \cdot e^{-jk(m-1)d\sin\theta}$

这说明子阵列2的导向向量是子阵列1的导向向量乘以相位因子 $e^{-jkd\sin\theta}$。

### 2.4 信号子空间关系

设 $\mathbf{U}_S$ 是总阵列信号子空间的基，可以分解为：

$$\mathbf{U}_S = \begin{bmatrix} \mathbf{U}_{S1} \\ \mathbf{U}_{S2} \end{bmatrix}$$

其中：
- $\mathbf{U}_{S1} \in \mathbb{C}^{M \times K}$：对应子阵列1
- $\mathbf{U}_{S2} \in \mathbb{C}^{M \times K}$：对应子阵列2

**旋转不变性**：

由于 $\mathbf{D}_2 = \mathbf{D}_1\mathbf{\Phi}$，存在非奇异矩阵 $\mathbf{T}$ 使得：

$$\mathbf{D}_1 = \mathbf{U}_{S1}\mathbf{T}$$
$$\mathbf{D}_2 = \mathbf{U}_{S2}\mathbf{T}$$

因此：

$$\mathbf{U}_{S2}\mathbf{T} = \mathbf{U}_{S1}\mathbf{T}\mathbf{\Phi}$$

$$\mathbf{U}_{S2} = \mathbf{U}_{S1}\mathbf{T}\mathbf{\Phi}\mathbf{T}^{-1}$$

定义 $\mathbf{\Psi} = \mathbf{T}\mathbf{\Phi}\mathbf{T}^{-1}$，则：

$$\boxed{\mathbf{U}_{S2} = \mathbf{U}_{S1}\mathbf{\Psi}}$$

**关键洞察**：$\mathbf{\Psi}$ 和 $\mathbf{\Phi}$ 具有相同的特征值！

### 2.5 ESPRIT求解

**广义特征值问题**：

从 $\mathbf{U}_{S2} = \mathbf{U}_{S1}\mathbf{\Psi}$ 可得：

$$\mathbf{U}_{S2}\mathbf{v}_i = \psi_i\mathbf{U}_{S1}\mathbf{v}_i$$

这是广义特征值问题，其中：
- $\psi_i$：广义特征值（等于 $e^{-jkd\sin\theta_i}$）
- $\mathbf{v}_i$：广义特征向量

**DOA估计**：

$$\boxed{\hat{\theta}_i = \arcsin\left(-\frac{\arg(\psi_i)}{kd}\right)}$$

其中 $\arg(\psi_i)$ 是 $\psi_i$ 的相位。

---

## 3. 算法实现

### 3.1 基本ESPRIT算法

```python
import numpy as np
from scipy.linalg import eig

class ESPRIT:
    def __init__(self, array_spacing, n_sources, fs=16000, c=343):
        """
        ESPRIT算法
        
        参数:
            array_spacing: 子阵列间距 (m)
            n_sources: 声源数量
            fs: 采样率 (Hz)
            c: 声速 (m/s)
        """
        self.d = array_spacing
        self.K = n_sources
        self.fs = fs
        self.c = c
        
    def estimate_doa(self, X, f):
        """
        估计DOA
        
        参数:
            X: [P, T] - 接收信号（P=M+1）
            f: 频率 (Hz)
        
        返回:
            doa_estimates: [K] - DOA估计 (弧度)
        """
        P, T = X.shape
        M = P - 1  # 子阵列大小
        
        # 1. 估计协方差矩阵
        R_X = (X @ X.conj().T) / T
        
        # 2. 特征分解
        eigenvalues, eigenvectors = np.linalg.eigh(R_X)
        
        # 3. 降序排列
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # 4. 提取信号子空间
        U_S = eigenvectors[:, :self.K]
        
        # 5. 分割子阵列
        U_S1 = U_S[:M, :]  # 前M行
        U_S2 = U_S[1:, :]  # 后M行
        
        # 6. 求解广义特征值问题
        # 方法1：最小二乘解
        Psi = np.linalg.lstsq(U_S1, U_S2, rcond=None)[0]
        eigenvalues_psi = np.linalg.eigvals(Psi)
        
        # 7. 转换为DOA
        k = 2 * np.pi * f / self.c
        doa_estimates = np.arcsin(-np.angle(eigenvalues_psi) / (k * self.d))
        
        # 8. 确保在有效范围内
        doa_estimates = np.clip(doa_estimates, -np.pi/2, np.pi/2)
        
        return doa_estimates
```

### 3.2 TLS-ESPRIT (总体最小二乘)

标准ESPRIT使用最小二乘求解，但TLS-ESPRIT考虑了 $\mathbf{U}_{S1}$ 和 $\mathbf{U}_{S2}$ 都存在误差的情况。

```python
def tls_esprit(self, U_S1, U_S2):
    """
    总体最小二乘ESPRIT
    
    参数:
        U_S1, U_S2: 子阵列信号子空间
    
    返回:
        eigenvalues: 广义特征值
    """
    # 构造增广矩阵
    C = np.hstack([-U_S2, U_S1])
    
    # SVD分解
    U, S, Vh = np.linalg.svd(C)
    
    # 最小奇异值对应的右奇异向量
    V = Vh.conj().T
    V12 = V[:self.K, self.K:]
    V22 = V[self.K:, self.K:]
    
    # 广义特征值
    eigenvalues = np.linalg.eigvals(-V12 @ np.linalg.inv(V22))
    
    return eigenvalues
```

**TLS vs LS**：
- **LS**：假设 $\mathbf{U}_{S1}$ 无误差，只有 $\mathbf{U}_{S2}$ 有误差
- **TLS**：同时考虑两者的误差，更加鲁棒

### 3.3 完整示例

```python
# 使用示例
import matplotlib.pyplot as plt

# 参数设置
fs = 16000
c = 343
d = 0.05  # 5cm间距
n_mics = 5
n_sources = 2
f = 1000  # Hz

# 创建ESPRIT对象
esprit = ESPRIT(array_spacing=d, n_sources=n_sources, fs=fs, c=c)

# 模拟信号
true_doas = np.array([np.deg2rad(30), np.deg2rad(-45)])
T = 1000  # 时间采样点

# 生成导向矩阵
k = 2 * np.pi * f / c
D = np.zeros((n_mics, n_sources), dtype=complex)
for i in range(n_mics):
    for j in range(n_sources):
        D[i, j] = np.exp(-1j * k * i * d * np.sin(true_doas[j]))

# 生成信号
s = np.random.randn(n_sources, T) + 1j * np.random.randn(n_sources, T)
X = D @ s

# 添加噪声
snr_db = 20
signal_power = np.mean(np.abs(X)**2)
noise_power = signal_power / (10**(snr_db/10))
noise = np.sqrt(noise_power/2) * (np.random.randn(*X.shape) + 
                                   1j * np.random.randn(*X.shape))
X = X + noise

# DOA估计
doa_est = esprit.estimate_doa(X, f)

# 结果
print("真实DOA:", np.degrees(true_doas))
print("估计DOA:", np.degrees(doa_est))
print("误差:", np.degrees(doa_est - true_doas))
```

---

## 4. 2D-ESPRIT

对于平面阵列，可以同时估计方位角和俯仰角。

### 4.1 平面阵列结构

```
🎤 🎤 🎤 🎤
🎤 🎤 🎤 🎤  ← y方向位移
🎤 🎤 🎤 🎤
🎤 🎤 🎤 🎤
↑
x方向位移
```

### 4.2 旋转不变性

对于2D阵列，存在两个方向的旋转不变性：

$$\mathbf{D}_{x+} = \mathbf{D}_x\mathbf{\Phi}_x$$
$$\mathbf{D}_{y+} = \mathbf{D}_y\mathbf{\Phi}_y$$

其中：

$$\mathbf{\Phi}_x = \text{diag}(e^{-jkd_x\sin\theta_i\cos\phi_i})$$
$$\mathbf{\Phi}_y = \text{diag}(e^{-jkd_y\sin\theta_i\sin\phi_i})$$

- $\theta_i$：第$i$个源的俯仰角
- $\phi_i$：第$i$个源的方位角

### 4.3 联合估计算法

```python
def esprit_2d(X, K, dx, dy, f, c=343):
    """
    2D-ESPRIT算法
    
    参数:
        X: [Px*Py, T] - 平面阵列信号
        K: 声源数量
        dx, dy: x和y方向间距
        f: 频率
        c: 声速
    
    返回:
        azimuth: [K] - 方位角
        elevation: [K] - 俯仰角
    """
    Px_Py, T = X.shape
    
    # 假设矩形阵列
    Px = int(np.sqrt(Px_Py))
    Py = Px
    
    # 1. 协方差矩阵和特征分解
    R_X = (X @ X.conj().T) / T
    eigenvalues, eigenvectors = np.linalg.eigh(R_X)
    idx = eigenvalues.argsort()[::-1]
    U_S = eigenvectors[:, idx[:K]]
    
    # 2. 重塑为2D结构
    U_S_2d = U_S.reshape(Px, Py, K)
    
    # 3. x方向子阵列
    U_Sx1 = U_S_2d[:-1, :, :].reshape(-1, K)
    U_Sx2 = U_S_2d[1:, :, :].reshape(-1, K)
    
    # 4. y方向子阵列
    U_Sy1 = U_S_2d[:, :-1, :].reshape(-1, K)
    U_Sy2 = U_S_2d[:, 1:, :].reshape(-1, K)
    
    # 5. 求解旋转算子
    Psi_x = np.linalg.lstsq(U_Sx1, U_Sx2, rcond=None)[0]
    Psi_y = np.linalg.lstsq(U_Sy1, U_Sy2, rcond=None)[0]
    
    # 6. 联合对角化（自动配对）
    eigenvals_x, eigenvecs = np.linalg.eig(Psi_x)
    Psi_y_transformed = eigenvecs.conj().T @ Psi_y @ eigenvecs
    eigenvals_y = np.diag(Psi_y_transformed)
    
    # 7. 转换为角度
    k = 2 * np.pi * f / c
    
    # 从特征值恢复角度
    phase_x = -np.angle(eigenvals_x) / (k * dx)
    phase_y = -np.angle(eigenvals_y) / (k * dy)
    
    # 计算方位角和俯仰角
    azimuth = np.arctan2(phase_y, phase_x)  # 方位角
    elevation = np.arcsin(np.sqrt(phase_x**2 + phase_y**2))  # 俯仰角
    
    return azimuth, elevation
```

**配对问题**：
- x和y方向的特征值需要正确配对
- 通过联合对角化自动实现配对

---

## 5. 性能分析

### 5.1 优势

1. **无需搜索**：直接通过特征值计算DOA，避免MUSIC的角度扫描
2. **计算高效**：复杂度 $O(P^3 + K^3)$，远小于MUSIC的 $O(P^3 + NP^2)$（N为搜索点数）
3. **高精度**：在高SNR下接近Cramér-Rao下界
4. **自动配对**：2D情况下可以自动配对方位角和俯仰角

### 5.2 局限

1. **阵列结构要求**：需要特定的阵列几何（平移不变性）
   - 均匀线性阵列
   - 均匀矩形阵列
   - 不适用于任意阵列

2. **相干源问题**：与MUSIC类似，对相干源敏感
   - 解决方法：空间平滑

3. **数值稳定性**：广义特征值求解可能不稳定
   - 解决方法：TLS-ESPRIT

4. **源数量要求**：需要预知源数量
   - 解决方法：信息论准则（AIC、MDL）

### 5.3 计算复杂度对比

| 算法 | 复杂度 | 搜索 | 精度 |
|------|--------|------|------|
| MUSIC | $O(P^3 + NP^2)$ | 需要 | 高 |
| ESPRIT | $O(P^3 + K^3)$ | 无需 | 高 |
| GCC-PHAT | $O(P^2 \log T)$ | 无需 | 中 |
| SRP-PHAT | $O(NP^2)$ | 需要 | 中 |

其中：
- $P$：麦克风数量
- $K$：声源数量（通常 $K \ll P$）
- $N$：搜索点数（通常 $N \gg K$）
- $T$：时间采样点数

**结论**：ESPRIT在保持高精度的同时，计算效率显著优于MUSIC。

### 5.4 Cramér-Rao下界

ESPRIT的估计方差在高SNR下接近CRLB：

$$\text{Var}(\hat{\theta}_i) \approx \frac{6}{(2\pi)^2 \text{SNR} \cdot T \cdot (d/\lambda)^2 \cdot M(M^2-1)}$$

其中：
- $M$：子阵列大小
- $d/\lambda$：归一化阵列间距
- $T$：快拍数

**影响因素**：
- SNR越高，精度越高
- 阵列孔径越大，精度越高
- 麦克风数量越多，精度越高

---

## 6. 改进方法

### 6.1 Unitary ESPRIT

**动机**：利用实值运算提高计算效率和数值稳定性。

对于中心对称的阵列，可以利用共轭对称性将复数运算转换为实数运算。

```python
def unitary_esprit(X, K, d, f, c=343):
    """
    Unitary ESPRIT算法
    
    参数:
        X: [P, T] - 接收信号
        K: 声源数量
        d: 阵列间距
        f: 频率
        c: 声速
    
    返回:
        doa_estimates: [K] - DOA估计
    """
    P, T = X.shape
    
    # 构造实值变换矩阵（中心对称阵列）
    Q = np.zeros((P, P), dtype=complex)
    
    if P % 2 == 0:  # 偶数个麦克风
        for i in range(P//2):
            Q[i, i] = 1/np.sqrt(2)
            Q[i, P-1-i] = 1j/np.sqrt(2)
            Q[P//2+i, i] = 1j/np.sqrt(2)
            Q[P//2+i, P-1-i] = 1/np.sqrt(2)
    else:  # 奇数个麦克风
        Q[P//2, P//2] = 1  # 中心元素
        for i in range(P//2):
            Q[i, i] = 1/np.sqrt(2)
            Q[i, P-1-i] = 1j/np.sqrt(2)
            Q[P//2+1+i, i] = 1j/np.sqrt(2)
            Q[P//2+1+i, P-1-i] = 1/np.sqrt(2)
    
    # 实值变换
    X_real = Q.conj().T @ X
    
    # 协方差矩阵（实值）
    R_real = np.real((X_real @ X_real.conj().T) / T)
    
    # 特征分解（实值）
    eigenvalues, eigenvectors = np.linalg.eigh(R_real)
    idx = eigenvalues.argsort()[::-1]
    U_S_real = eigenvectors[:, idx[:K]]
    
    # 转换回复数域
    U_S = Q @ U_S_real
    
    # 标准ESPRIT处理
    M = P - 1
    U_S1 = U_S[:M, :]
    U_S2 = U_S[1:, :]
    
    Psi = np.linalg.lstsq(U_S1, U_S2, rcond=None)[0]
    eigenvalues_psi = np.linalg.eigvals(Psi)
    
    # DOA估计
    k = 2 * np.pi * f / c
    doa_estimates = np.arcsin(-np.angle(eigenvalues_psi) / (k * d))
    
    return doa_estimates
```

**优势**：
- 计算效率提高约2倍
- 数值稳定性更好
- 适用于中心对称阵列

### 6.2 Forward-Backward ESPRIT

利用阵列的前向和后向信息，提高估计精度。

```python
def fb_esprit(X, K, d, f, c=343):
    """
    Forward-Backward ESPRIT
    
    参数:
        X: [P, T] - 接收信号
        K: 声源数量
        d: 阵列间距
        f: 频率
        c: 声速
    
    返回:
        doa_estimates: [K] - DOA估计
    """
    P, T = X.shape
    
    # 前向协方差
    R_f = (X @ X.conj().T) / T
    
    # 后向协方差（共轭翻转）
    J = np.eye(P)[::-1]  # 翻转矩阵
    R_b = J @ R_f.conj() @ J
    
    # 平均协方差
    R_avg = (R_f + R_b) / 2
    
    # 特征分解
    eigenvalues, eigenvectors = np.linalg.eigh(R_avg)
    idx = eigenvalues.argsort()[::-1]
    U_S = eigenvectors[:, idx[:K]]
    
    # 标准ESPRIT处理
    M = P - 1
    U_S1 = U_S[:M, :]
    U_S2 = U_S[1:, :]
    
    Psi = np.linalg.lstsq(U_S1, U_S2, rcond=None)[0]
    eigenvalues_psi = np.linalg.eigvals(Psi)
    
    # DOA估计
    k = 2 * np.pi * f / c
    doa_estimates = np.arcsin(-np.angle(eigenvalues_psi) / (k * d))
    
    return doa_estimates
```

**优势**：
- 利用更多信息
- 提高估计精度
- 对噪声更鲁棒

### 6.3 宽带ESPRIT

对于宽带信号，需要在多个频率上进行处理。

**相干信号子空间方法 (CSM)**：

```python
def wideband_esprit_csm(X_stft, K, d, freq_bins, c=343):
    """
    宽带ESPRIT（相干信号子空间方法）
    
    参数:
        X_stft: [P, F, T] - STFT信号
        K: 声源数量
        d: 阵列间距
        freq_bins: 使用的频率bin索引
        c: 声速
    
    返回:
        doa_estimates: [K] - DOA估计
    """
    P, F, T = X_stft.shape
    
    # 选择参考频率（通常选中间频率）
    f_ref_idx = freq_bins[len(freq_bins)//2]
    f_ref = f_ref_idx * (c / (2 * d * F))  # 简化的频率计算
    
    # 聚焦协方差矩阵
    R_focused = np.zeros((P, P), dtype=complex)
    
    for f_idx in freq_bins:
        f = f_idx * (c / (2 * d * F))
        
        # 当前频率的协方差
        X_f = X_stft[:, f_idx, :]
        R_f = (X_f @ X_f.conj().T) / T
        
        # 聚焦矩阵（相位补偿）
        T_f = compute_focusing_matrix(f, f_ref, d, P, c)
        
        # 聚焦到参考频率
        R_focused += T_f @ R_f @ T_f.conj().T
    
    R_focused /= len(freq_bins)
    
    # 在聚焦后的协方差矩阵上应用ESPRIT
    eigenvalues, eigenvectors = np.linalg.eigh(R_focused)
    idx = eigenvalues.argsort()[::-1]
    U_S = eigenvectors[:, idx[:K]]
    
    # 标准ESPRIT处理
    M = P - 1
    U_S1 = U_S[:M, :]
    U_S2 = U_S[1:, :]
    
    Psi = np.linalg.lstsq(U_S1, U_S2, rcond=None)[0]
    eigenvalues_psi = np.linalg.eigvals(Psi)
    
    # DOA估计
    k_ref = 2 * np.pi * f_ref / c
    doa_estimates = np.arcsin(-np.angle(eigenvalues_psi) / (k_ref * d))
    
    return doa_estimates

def compute_focusing_matrix(f, f_ref, d, P, c):
    """
    计算聚焦矩阵（简化版本）
    
    参数:
        f: 当前频率
        f_ref: 参考频率
        d: 阵列间距
        P: 麦克风数量
        c: 声速
    
    返回:
        T: 聚焦矩阵
    """
    # 简化实现：只考虑相位补偿
    phase_ratio = f / f_ref
    T = np.diag([phase_ratio**i for i in range(P)])
    return T
```

---

## 7. 实际应用考虑

### 7.1 阵列校准

**位置误差影响**：

阵列位置误差会导致旋转不变性破坏，影响估计精度。

**误差模型**：

$$\mathbf{r}_i = \mathbf{r}_{i,\text{nominal}} + \Delta\mathbf{r}_i$$

其中 $\Delta\mathbf{r}_i$ 是位置误差。

**自校准方法**：

```python
def self_calibrating_esprit(X, K, nominal_positions, max_iter=10):
    """
    自校准ESPRIT
    
    参数:
        X: [P, T] - 接收信号
        K: 声源数量
        nominal_positions: [P, 3] - 标称位置
        max_iter: 最大迭代次数
    
    返回:
        doa_estimates: [K] - DOA估计
        calibrated_positions: [P, 3] - 校准后位置
    """
    positions = nominal_positions.copy()
    
    for iteration in range(max_iter):
        # 1. 基于当前位置估计DOA
        doa_est = esprit_with_positions(X, K, positions)
        
        # 2. 基于DOA估计校准位置
        positions = calibrate_positions(X, doa_est, positions)
        
        # 3. 检查收敛
        if iteration > 0:
            position_change = np.linalg.norm(positions - prev_positions)
            if position_change < 1e-4:
                break
        
        prev_positions = positions.copy()
    
    return doa_est, positions
```

### 7.2 源数量估计

ESPRIT需要预知源数量，可以使用信息论准则估计。

**AIC准则 (Akaike Information Criterion)**：

$$\text{AIC}(k) = -2\log L(\hat{\theta}_k) + 2k$$

**MDL准则 (Minimum Description Length)**：

$$\text{MDL}(k) = -\log L(\hat{\theta}_k) + \frac{k}{2}\log N$$

```python
def estimate_source_number(R, P, T, method='mdl'):
    """
    估计声源数量
    
    参数:
        R: [P, P] - 协方差矩阵
        P: 麦克风数量
        T: 快拍数
        method: 'aic' 或 'mdl'
    
    返回:
        K_est: 估计的源数量
    """
    # 特征分解
    eigenvalues = np.linalg.eigvalsh(R)
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    # 计算准则
    criteria = []
    for k in range(P):
        if k == P:
            criteria.append(np.inf)
            continue
        
        # 似然函数
        noise_eigenvalues = eigenvalues[k:]
        geometric_mean = np.exp(np.mean(np.log(noise_eigenvalues + 1e-10)))
        arithmetic_mean = np.mean(noise_eigenvalues)
        
        log_likelihood = -T * (P - k) * np.log(arithmetic_mean / geometric_mean)
        
        # 惩罚项
        if method == 'aic':
            penalty = 2 * k * (2*P - k)
        elif method == 'mdl':
            penalty = 0.5 * k * (2*P - k) * np.log(T)
        else:
            raise ValueError("method must be 'aic' or 'mdl'")
        
        criteria.append(-log_likelihood + penalty)
    
    # 选择最小准则对应的k
    K_est = np.argmin(criteria)
    
    return K_est
```

### 7.3 相干源处理

**空间平滑 (Spatial Smoothing)**：

对于相干源，可以使用空间平滑技术。

```python
def spatial_smoothing_esprit(X, K, d, f, L, c=343):
    """
    空间平滑ESPRIT
    
    参数:
        X: [P, T] - 接收信号
        K: 声源数量
        d: 阵列间距
        f: 频率
        L: 子阵列长度
        c: 声速
    
    返回:
        doa_estimates: [K] - DOA估计
    """
    P, T = X.shape
    M = P - L + 1  # 子阵列数量
    
    # 空间平滑协方差矩阵
    R_smooth = np.zeros((L, L), dtype=complex)
    
    for m in range(M):
        X_sub = X[m:m+L, :]
        R_sub = (X_sub @ X_sub.conj().T) / T
        R_smooth += R_sub
    
    R_smooth /= M
    
    # 在平滑后的协方差矩阵上应用ESPRIT
    eigenvalues, eigenvectors = np.linalg.eigh(R_smooth)
    idx = eigenvalues.argsort()[::-1]
    U_S = eigenvectors[:, idx[:K]]
    
    # 标准ESPRIT处理
    U_S1 = U_S[:L-1, :]
    U_S2 = U_S[1:, :]
    
    Psi = np.linalg.lstsq(U_S1, U_S2, rcond=None)[0]
    eigenvalues_psi = np.linalg.eigvals(Psi)
    
    # DOA估计
    k = 2 * np.pi * f / c
    doa_estimates = np.arcsin(-np.angle(eigenvalues_psi) / (k * d))
    
    return doa_estimates
```

**前后向空间平滑**：

结合前向和后向平滑，进一步提高性能。

```python
def fb_spatial_smoothing_esprit(X, K, d, f, L, c=343):
    """
    前后向空间平滑ESPRIT
    """
    P, T = X.shape
    M = P - L + 1
    
    # 前向平滑
    R_forward = np.zeros((L, L), dtype=complex)
    for m in range(M):
        X_sub = X[m:m+L, :]
        R_sub = (X_sub @ X_sub.conj().T) / T
        R_forward += R_sub
    R_forward /= M
    
    # 后向平滑
    J = np.eye(L)[::-1]
    R_backward = J @ R_forward.conj() @ J
    
    # 平均
    R_smooth = (R_forward + R_backward) / 2
    
    # 应用ESPRIT
    eigenvalues, eigenvectors = np.linalg.eigh(R_smooth)
    idx = eigenvalues.argsort()[::-1]
    U_S = eigenvectors[:, idx[:K]]
    
    U_S1 = U_S[:L-1, :]
    U_S2 = U_S[1:, :]
    
    Psi = np.linalg.lstsq(U_S1, U_S2, rcond=None)[0]
    eigenvalues_psi = np.linalg.eigvals(Psi)
    
    k = 2 * np.pi * f / c
    doa_estimates = np.arcsin(-np.angle(eigenvalues_psi) / (k * d))
    
    return doa_estimates
```

---

## 8. 应用场景

### 8.1 雷达系统

**优势**：
- 高精度角度估计
- 实时处理能力
- 多目标检测

**配置**：
- 均匀线性阵列
- 窄带信号
- 高SNR环境

### 8.2 无线通信

**优势**：
- 快速DOA估计
- 适合移动环境
- 低计算复杂度

**应用**：
- 基站天线阵列
- 智能天线系统
- 多用户MIMO

### 8.3 声学定位

**挑战**：
- 宽带信号
- 混响环境
- 阵列误差

**解决方案**：
- 宽带ESPRIT (CSM)
- 自校准算法
- 空间平滑

**应用**：
- 智能音箱
- 视频会议系统
- 机器人听觉

---

## 9. 与其他方法对比

### 9.1 ESPRIT vs MUSIC

| 特性 | ESPRIT | MUSIC |
|------|--------|-------|
| **搜索需求** | 无需 | 需要 |
| **计算复杂度** | $O(P^3 + K^3)$ | $O(P^3 + NP^2)$ |
| **阵列要求** | 特定结构 | 任意 |
| **精度** | 高（接近CRLB） | 高（接近CRLB） |
| **分辨率** | 高 | 超高 |
| **实时性** | 好 | 差 |
| **2D扩展** | 自动配对 | 需要配对 |

**选择建议**：
- **实时性要求高** → ESPRIT
- **任意阵列** → MUSIC
- **均匀阵列 + 高效** → ESPRIT
- **最高分辨率** → MUSIC

### 9.2 ESPRIT vs GCC-PHAT

| 特性 | ESPRIT | GCC-PHAT |
|------|--------|----------|
| **分辨率** | 超高 | 中 |
| **麦克风数** | ≥3 | 2 |
| **多源能力** | 有 | 无 |
| **计算复杂度** | 中 | 低 |
| **混响鲁棒性** | 中 | 好 |

### 9.3 综合对比

| 方法 | 分辨率 | 计算量 | 搜索 | 阵列要求 | 多源 | 实时性 |
|------|--------|--------|------|----------|------|--------|
| GCC-PHAT | 中 | 低 | 无 | 任意 | 无 | 很好 |
| SRP-PHAT | 中 | 高 | 需要 | 任意 | 有 | 差 |
| MUSIC | 超高 | 高 | 需要 | 任意 | 有 | 差 |
| ESPRIT | 超高 | 中 | 无 | 特定 | 有 | 好 |

---

## 10. 总结

### 10.1 核心公式

**旋转不变性**：
$$\mathbf{U}_{S2} = \mathbf{U}_{S1}\mathbf{\Psi}$$

**DOA估计**：
$$\hat{\theta}_i = \arcsin\left(-\frac{\arg(\psi_i)}{kd}\right)$$

其中 $\psi_i$ 是 $\mathbf{\Psi}$ 的特征值。

### 10.2 关键优势

1. **无需搜索**：直接特征值计算
2. **计算高效**：避免角度扫描
3. **高精度**：接近CRLB
4. **可扩展**：易于扩展到2D

### 10.3 主要局限

1. **阵列结构**：需要平移不变性
2. **相干源**：需要空间平滑
3. **源数量**：需要预知或估计
4. **数值稳定性**：需要TLS改进

### 10.4 适用场景

**最适合**：
- 均匀线性/平面阵列
- 实时处理需求
- 高精度要求
- 多源环境

**不适合**：
- 任意阵列几何
- 强相干源
- 极低SNR

### 10.5 实践建议

1. **阵列设计**：
   - 使用均匀间距
   - 间距 $d \leq \lambda/2$ 避免模糊
   - 增加麦克风数量提高精度

2. **算法选择**：
   - 标准场景：基本ESPRIT
   - 相干源：空间平滑ESPRIT
   - 宽带信号：CSM-ESPRIT
   - 高精度：TLS-ESPRIT

3. **参数设置**：
   - 使用MDL估计源数量
   - 选择合适的频率范围
   - 足够的快拍数（T > 10P）

4. **性能优化**：
   - 预处理降噪
   - 频率选择
   - 后处理平滑

---

## 参考文献

1. Roy, R., & Kailath, T. (1989). "ESPRIT-estimation of signal parameters via rotational invariance techniques." IEEE Transactions on acoustics, speech, and signal processing.

2. Haardt, M., & Nossek, J. A. (1995). "Unitary ESPRIT: how to obtain increased estimation accuracy with a reduced computational burden." IEEE Transactions on Signal processing.

3. Zoltowski, M. D., Haardt, M., & Mathews, C. P. (1996). "Closed-form 2-D angle estimation with rectangular arrays in element space or beamspace via unitary ESPRIT." IEEE Transactions on Signal Processing.

4. Pillai, S. U., & Kwon, B. H. (1989). "Forward/backward spatial smoothing techniques for coherent signal identification." IEEE Transactions on Acoustics, Speech, and Signal Processing.
