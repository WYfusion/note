# MUSIC 算法

## 1. 概述

**MUSIC (Multiple Signal Classification)** 是一种基于子空间分解的高分辨率DOA估计算法，由Schmidt于1986年提出。它利用信号子空间和噪声子空间的正交性来估计声源方向。

### 1.1 核心思想

```
协方差矩阵 R_X
    ↓
[特征分解]
    ↓
信号子空间 U_S  |  噪声子空间 U_N
    ↓                    ↓
包含导向矢量      ⊥ 导向矢量
    ↓
搜索使 d^H U_N U_N^H d 最小的方向
    ↓
DOA估计
```

**关键性质**：真实声源的导向矢量与噪声子空间正交。

---

## 2. 数学推导

### 2.1 信号模型

**窄带信号模型**：

$$\mathbf{X}(t) = \sum_{k=1}^{K} \mathbf{d}(\theta_k) s_k(t) + \mathbf{N}(t) = \mathbf{D}\mathbf{s}(t) + \mathbf{N}(t)$$

其中：
- $\mathbf{X}(t) \in \mathbb{C}^P$：接收信号向量
- $\mathbf{D} = [\mathbf{d}(\theta_1), \ldots, \mathbf{d}(\theta_K)]$：导向矩阵
- $\mathbf{s}(t) \in \mathbb{C}^K$：源信号向量
- $\mathbf{N}(t) \in \mathbb{C}^P$：噪声向量

**假设**：
1. 源信号不相关
2. 噪声为白噪声
3. 信号与噪声不相关

### 2.2 协方差矩阵

$$\mathbf{R}_X = \mathbb{E}[\mathbf{X}(t)\mathbf{X}^H(t)] = \mathbf{D}\mathbf{R}_s\mathbf{D}^H + \sigma_n^2\mathbf{I}$$

### 2.3 特征分解

$$\mathbf{R}_X = \sum_{i=1}^{P} \lambda_i \mathbf{u}_i \mathbf{u}_i^H = \mathbf{U}_S\mathbf{\Lambda}_S\mathbf{U}_S^H + \mathbf{U}_N\mathbf{\Lambda}_N\mathbf{U}_N^H$$

其中：
- $\mathbf{U}_S = [\mathbf{u}_1, \ldots, \mathbf{u}_K]$：信号子空间
- $\mathbf{U}_N = [\mathbf{u}_{K+1}, \ldots, \mathbf{u}_P]$：噪声子空间

### 2.4 MUSIC谱

$$P_{\text{MUSIC}}(\theta) = \frac{1}{\mathbf{d}^H(\theta)\mathbf{U}_N\mathbf{U}_N^H\mathbf{d}(\theta)}$$

**DOA估计**：
$$\hat{\theta}_k = \arg\max_{\theta} P_{\text{MUSIC}}(\theta)$$

---

## 3. 算法实现

```python
import numpy as np

class MUSIC:
    def __init__(self, array_geometry, n_sources, fs=16000, c=343):
        self.mic_positions = array_geometry
        self.P = len(array_geometry)
        self.K = n_sources
        self.fs = fs
        self.c = c
    
    def steering_vector(self, theta, phi, f):
        k = 2 * np.pi * f / self.c
        direction = np.array([
            np.cos(phi) * np.cos(theta),
            np.cos(phi) * np.sin(theta),
            np.sin(phi)
        ])
        delays = self.mic_positions @ direction
        d = np.exp(-1j * k * delays)
        return d
    
    def estimate_doa(self, X, f):
        P, T = X.shape
        R_X = (X @ X.conj().T) / T
        eigenvalues, eigenvectors = np.linalg.eigh(R_X)
        idx = eigenvalues.argsort()[::-1]
        U_N = eigenvectors[:, idx[self.K:]]
        
        theta_grid = np.linspace(-np.pi, np.pi, 360)
        spectrum = np.zeros(len(theta_grid))
        
        for i, theta in enumerate(theta_grid):
            d = self.steering_vector(theta, 0, f)
            denominator = np.abs(d.conj() @ U_N @ U_N.conj().T @ d)
            spectrum[i] = 1.0 / (denominator + 1e-10)
        
        from scipy.signal import find_peaks
        peaks_idx, _ = find_peaks(spectrum, distance=10)
        if len(peaks_idx) > self.K:
            peak_heights = spectrum[peaks_idx]
            top_k_idx = np.argsort(peak_heights)[-self.K:]
            peaks_idx = peaks_idx[top_k_idx]
        
        doa_estimates = theta_grid[peaks_idx]
        return doa_estimates, spectrum
```

---

## 4. 改进方法

### 4.1 Root-MUSIC

避免角度搜索，直接通过求根得到DOA（仅适用于ULA）。

### 4.2 空间平滑MUSIC

处理相干源问题，将阵列分为重叠子阵列。

### 4.3 宽带MUSIC

使用相干信号子空间方法处理宽带信号。

---

## 5. 性能分析

### 5.1 分辨率

MUSIC可以突破Rayleigh限，分辨更接近的声源。

### 5.2 CRLB

MUSIC的估计方差在高SNR下接近Cramér-Rao下界。

### 5.3 计算复杂度

- 协方差估计：$O(P^2T)$
- 特征分解：$O(P^3)$
- 谱搜索：$O(NP^2)$

---

## 6. 优势与局限

### 6.1 优势

1. 超分辨能力
2. 多源定位
3. 理论基础扎实
4. 灵活性强

### 6.2 局限

1. 计算复杂度高
2. 相干源问题
3. 需要已知源数量
4. 对模型误差敏感

---

## 7. 应用场景

- 智能音箱
- 视频会议
- 声学相机
- 雷达系统

---

## 参考文献

1. Schmidt, R. (1986). "Multiple emitter location and signal parameter estimation." IEEE TAP.

2. Krim, H., & Viberg, M. (1996). "Two decades of array signal processing research." IEEE SP Magazine.

3. Van Trees, H. L. (2002). "Optimum Array Processing." Wiley.
