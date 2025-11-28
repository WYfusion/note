# GCC-PHAT 算法

## 1. 概述

**GCC-PHAT (Generalized Cross-Correlation with Phase Transform)** 是最经典的TDOA估计算法，通过相位变换增强互相关峰值，对混响具有良好的鲁棒性。

### 1.1 核心思想

```
信号1: x₁(t) ────┐
                  ├─→ [互相关] ─→ [相位变换] ─→ [峰值检测] ─→ TDOA
信号2: x₂(t) ────┘
```

**关键步骤**：
1. 计算两路信号的互相关
2. 应用相位变换（PHAT）白化频谱
3. 检测相关峰值位置
4. 峰值对应的时延即为TDOA

---

## 2. 数学推导

### 2.1 互相关函数

**时域定义**：

$$R_{12}(\tau) = \int_{-\infty}^{\infty} x_1(t) x_2(t - \tau) dt$$

**物理意义**：衡量信号1与信号2平移$\tau$后的相似度。

**频域实现**：

根据Wiener-Khinchin定理：

$$R_{12}(\tau) = \mathcal{F}^{-1}\{X_1(f) X_2^*(f)\}$$

其中：
- $X_1(f), X_2(f)$：信号的傅里叶变换
- $X_2^*(f)$：$X_2(f)$的共轭
- $\mathcal{F}^{-1}$：逆傅里叶变换

**离散形式**：

$$R_{12}[k] = \text{IFFT}\{X_1[m] \cdot X_2^*[m]\}$$

### 2.2 信号模型

假设两个麦克风接收到的信号：

$$x_1(t) = s(t) + n_1(t)$$
$$x_2(t) = \alpha s(t - \tau_0) + n_2(t)$$

其中：
- $s(t)$：源信号
- $\tau_0$：真实TDOA
- $\alpha$：衰减系数
- $n_1(t), n_2(t)$：噪声

**互相关**：

$$R_{12}(\tau) = \alpha R_{ss}(\tau - \tau_0) + R_{n_1n_2}(\tau)$$

其中 $R_{ss}$ 是源信号的自相关。

**峰值位置**：

$$\hat{\tau}_0 = \arg\max_{\tau} R_{12}(\tau)$$

### 2.3 广义互相关 (GCC)

引入加权函数 $\Psi(f)$：

$$R_{12}^{\Psi}(\tau) = \int_{-\infty}^{\infty} \Psi(f) X_1(f) X_2^*(f) e^{j2\pi f\tau} df$$

**离散形式**：

$$R_{12}^{\Psi}[k] = \text{IFFT}\{\Psi[m] \cdot X_1[m] \cdot X_2^*[m]\}$$

**不同加权函数**：

| 名称 | 加权函数 $\Psi(f)$ | 特点 |
|------|-------------------|------|
| **基本CC** | $1$ | 无加权 |
| **PHAT** | $\frac{1}{\|X_1(f)X_2^*(f)\|}$ | 白化，抑制混响 |
| **SCOT** | $\frac{1}{\sqrt{\|X_1(f)\|^2\|X_2(f)\|^2}}$ | 平滑 |
| **Roth** | $\frac{1}{\|X_1(f)\|^2}$ | 逆滤波 |
| **ML** | $\frac{\|S(f)\|^2}{\|N_1(f)\|^2\|N_2(f)\|^2}$ | 最大似然 |

### 2.4 PHAT推导

**目标**：白化频谱，使所有频率分量贡献相等。

**PHAT加权**：

$$\Psi_{\text{PHAT}}(f) = \frac{1}{|X_1(f)X_2^*(f)|}$$

**GCC-PHAT**：

$$R_{12}^{\text{PHAT}}(\tau) = \int_{-\infty}^{\infty} \frac{X_1(f)X_2^*(f)}{|X_1(f)X_2^*(f)|} e^{j2\pi f\tau} df$$

简化为：

$$R_{12}^{\text{PHAT}}(\tau) = \int_{-\infty}^{\infty} e^{j\angle(X_1(f)X_2^*(f))} e^{j2\pi f\tau} df$$

$$= \int_{-\infty}^{\infty} e^{j(\phi_1(f) - \phi_2(f))} e^{j2\pi f\tau} df$$

**物理意义**：
- 只保留相位信息，丢弃幅度
- 所有频率等权重
- 抑制混响（混响改变幅度，但相位相对稳定）

### 2.5 理想情况分析

假设 $x_2(t) = x_1(t - \tau_0)$（无噪声，无衰减）：

$$X_2(f) = X_1(f)e^{-j2\pi f\tau_0}$$

$$\frac{X_1(f)X_2^*(f)}{|X_1(f)X_2^*(f)|} = \frac{X_1(f)X_1^*(f)e^{j2\pi f\tau_0}}{|X_1(f)|^2} = e^{j2\pi f\tau_0}$$

$$R_{12}^{\text{PHAT}}(\tau) = \int_{-\infty}^{\infty} e^{j2\pi f(\tau_0 + \tau)} df = \delta(\tau + \tau_0)$$

**结论**：理想情况下，GCC-PHAT给出完美的冲激函数！

---

## 3. 算法实现

### 3.1 基本流程

```
输入: x₁[n], x₂[n] (时域信号)
输出: τ̂ (TDOA估计)

1. 分帧
   x₁_frame = x₁[n:n+N]
   x₂_frame = x₂[n:n+N]

2. 加窗
   x₁_win = x₁_frame * window
   x₂_win = x₂_frame * window

3. FFT
   X₁ = FFT(x₁_win)
   X₂ = FFT(x₂_win)

4. 计算互功率谱
   G₁₂ = X₁ · X₂*

5. PHAT加权
   G₁₂_phat = G₁₂ / |G₁₂|

6. IFFT
   R₁₂ = IFFT(G₁₂_phat)

7. 峰值检测
   τ̂ = argmax(R₁₂)

8. 转换为时间
   τ̂_time = τ̂ / fs
```

### 3.2 Python实现

```python
import numpy as np
from scipy.signal import stft, istft
from scipy.fft import fft, ifft, fftshift

def gcc_phat(x1, x2, fs=16000, frame_length=1024, hop_length=512):
    """
    GCC-PHAT算法
    
    参数:
        x1, x2: 输入信号
        fs: 采样率
        frame_length: 帧长
        hop_length: 帧移
    
    返回:
        tau: TDOA估计 (秒)
        confidence: 置信度
    """
    # 确保信号长度相同
    min_len = min(len(x1), len(x2))
    x1 = x1[:min_len]
    x2 = x2[:min_len]
    
    # 分帧FFT
    X1 = fft(x1, n=frame_length)
    X2 = fft(x2, n=frame_length)
    
    # 互功率谱
    G12 = X1 * np.conj(X2)
    
    # PHAT加权
    G12_phat = G12 / (np.abs(G12) + 1e-10)
    
    # IFFT
    R12 = np.real(ifft(G12_phat))
    
    # 重排（将负延迟移到前半部分）
    R12 = fftshift(R12)
    
    # 峰值检测
    max_idx = np.argmax(R12)
    confidence = R12[max_idx]
    
    # 转换为TDOA（秒）
    tau_samples = max_idx - frame_length // 2
    tau = tau_samples / fs
    
    return tau, confidence


def gcc_phat_realtime(x1, x2, fs=16000):
    """
    实时GCC-PHAT（单帧）
    
    参数:
        x1, x2: [N] - 单帧信号
        fs: 采样率
    
    返回:
        tau: TDOA (秒)
        R12: 相关函数
    """
    N = len(x1)
    
    # FFT
    X1 = fft(x1)
    X2 = fft(x2)
    
    # GCC-PHAT
    G12 = X1 * np.conj(X2)
    G12_phat = G12 / (np.abs(G12) + 1e-10)
    
    # IFFT
    R12 = np.real(ifft(G12_phat))
    R12 = fftshift(R12)
    
    # 峰值
    max_idx = np.argmax(R12)
    tau = (max_idx - N // 2) / fs
    
    return tau, R12


class GCCPHATTracker:
    """GCC-PHAT追踪器（带平滑）"""
    
    def __init__(self, fs=16000, frame_length=1024, alpha=0.7):
        self.fs = fs
        self.frame_length = frame_length
        self.alpha = alpha  # 平滑系数
        self.tau_prev = 0
        
    def process(self, x1, x2):
        """处理一帧"""
        tau, conf = gcc_phat(x1, x2, self.fs, self.frame_length)
        
        # 指数平滑
        tau_smooth = self.alpha * self.tau_prev + (1 - self.alpha) * tau
        self.tau_prev = tau_smooth
        
        return tau_smooth, conf
```

### 3.3 峰值检测改进

**多峰检测**：

```python
def find_peaks(R12, threshold=0.5, min_distance=10):
    """
    检测多个峰值
    
    参数:
        R12: 相关函数
        threshold: 相对阈值（相对于最大值）
        min_distance: 峰值间最小距离
    
    返回:
        peaks: 峰值位置列表
        values: 峰值大小列表
    """
    from scipy.signal import find_peaks as scipy_find_peaks
    
    # 归一化
    R12_norm = R12 / np.max(R12)
    
    # 检测峰值
    peaks, properties = scipy_find_peaks(
        R12_norm, 
        height=threshold,
        distance=min_distance
    )
    
    values = properties['peak_heights']
    
    # 按大小排序
    sorted_idx = np.argsort(values)[::-1]
    peaks = peaks[sorted_idx]
    values = values[sorted_idx]
    
    return peaks, values
```

**抛物线插值**（亚采样精度）：

```python
def parabolic_interpolation(R12, peak_idx):
    """
    抛物线插值提高精度
    
    参数:
        R12: 相关函数
        peak_idx: 峰值索引
    
    返回:
        refined_idx: 精细化的峰值位置
    """
    if peak_idx == 0 or peak_idx == len(R12) - 1:
        return peak_idx
    
    # 三点拟合抛物线
    y1 = R12[peak_idx - 1]
    y2 = R12[peak_idx]
    y3 = R12[peak_idx + 1]
    
    # 抛物线顶点
    delta = 0.5 * (y1 - y3) / (y1 - 2*y2 + y3)
    
    return peak_idx + delta
```

---

## 4. 从TDOA到位置

### 4.1 双麦克风定位

**1D定位**（线性阵列）：

$$\theta = \arccos\left(\frac{c\tau}{d}\right)$$

其中 $d$ 是麦克风间距。

**2D定位**（需要第三个麦克风）：

使用三角定位或最小二乘。

### 4.2 多麦克风定位

**配对TDOA**：

对于 $P$ 个麦克风，有 $P(P-1)/2$ 个麦克风对。

选择参考麦克风（如第1个），计算：

$$\tau_{1i}, \quad i = 2, 3, \ldots, P$$

**最小二乘定位**：

$$\hat{\mathbf{r}}_s = \arg\min_{\mathbf{r}_s} \sum_{i=2}^{P} \left(\tau_{1i}^{\text{meas}} - \tau_{1i}(\mathbf{r}_s)\right)^2$$

```python
def tdoa_to_position_2d(tdoa_list, mic_positions, c=343):
    """
    从TDOA估计2D位置
    
    参数:
        tdoa_list: [P-1] - TDOA列表（相对于第一个麦克风）
        mic_positions: [P, 2] - 麦克风位置
        c: 声速
    
    返回:
        position: [2] - 估计的声源位置
    """
    from scipy.optimize import least_squares
    
    def residual(pos):
        # 计算理论TDOA
        distances = np.linalg.norm(mic_positions - pos, axis=1)
        tdoa_pred = (distances - distances[0]) / c
        return tdoa_pred[1:] - tdoa_list
    
    # 初始猜测（麦克风中心）
    x0 = np.mean(mic_positions, axis=0)
    
    # 优化
    result = least_squares(residual, x0)
    
    return result.x
```

---

## 5. 性能分析

### 5.1 优势

1. **混响鲁棒性**：PHAT白化抑制混响
2. **计算高效**：FFT实现，$O(N\log N)$
3. **简单直观**：物理意义明确
4. **无需训练**：不依赖数据

### 5.2 局限

1. **单源假设**：多源时性能下降
2. **低SNR敏感**：噪声影响峰值检测
3. **相位模糊**：高频或大间距时
4. **需要后处理**：TDOA→位置需要几何求解

### 5.3 性能指标

**TDOA估计误差**：

$$\sigma_{\tau} \approx \frac{1}{2\pi\beta\sqrt{\text{SNR}}}$$

其中 $\beta$ 是信号有效带宽。

**角度估计误差**（远场）：

$$\sigma_{\theta} \approx \frac{c}{2\pi f d\sqrt{\text{SNR}}}$$

---

## 6. 改进方法

### 6.1 自适应PHAT

根据信噪比自适应调整加权：

$$\Psi_{\text{adaptive}}(f) = \frac{1}{|X_1(f)X_2^*(f)|^{\gamma}}$$

其中 $\gamma \in [0, 1]$：
- $\gamma = 0$：无加权
- $\gamma = 1$：标准PHAT
- $\gamma = 0.5$：折中

### 6.2 多频带GCC-PHAT

分频带处理，提高鲁棒性：

```python
def multiband_gcc_phat(x1, x2, fs=16000, n_bands=4):
    """
    多频带GCC-PHAT
    
    参数:
        x1, x2: 输入信号
        fs: 采样率
        n_bands: 频带数量
    
    返回:
        tau: TDOA估计
    """
    from scipy.signal import butter, filtfilt
    
    # 频带划分
    freq_edges = np.logspace(
        np.log10(100), 
        np.log10(fs/2), 
        n_bands + 1
    )
    
    tau_list = []
    conf_list = []
    
    for i in range(n_bands):
        # 带通滤波
        sos = butter(
            4, 
            [freq_edges[i], freq_edges[i+1]], 
            btype='band', 
            fs=fs, 
            output='sos'
        )
        x1_band = filtfilt(sos, x1)
        x2_band = filtfilt(sos, x2)
        
        # GCC-PHAT
        tau, conf = gcc_phat(x1_band, x2_band, fs)
        tau_list.append(tau)
        conf_list.append(conf)
    
    # 加权平均
    conf_array = np.array(conf_list)
    tau_array = np.array(tau_list)
    tau_final = np.average(tau_array, weights=conf_array)
    
    return tau_final
```

### 6.3 时频掩码增强

结合深度学习掩码：

```python
def masked_gcc_phat(x1, x2, mask, fs=16000):
    """
    掩码增强的GCC-PHAT
    
    参数:
        x1, x2: 输入信号
        mask: [F, T] - 时频掩码（0-1）
        fs: 采样率
    
    返回:
        tau: TDOA估计
    """
    # STFT
    f, t, X1 = stft(x1, fs=fs)
    _, _, X2 = stft(x2, fs=fs)
    
    # 应用掩码
    X1_masked = X1 * mask
    X2_masked = X2 * mask
    
    # 对每一帧计算GCC-PHAT
    tau_list = []
    for i in range(X1.shape[1]):
        G12 = X1_masked[:, i] * np.conj(X2_masked[:, i])
        G12_phat = G12 / (np.abs(G12) + 1e-10)
        R12 = np.real(ifft(G12_phat))
        R12 = fftshift(R12)
        tau_list.append(np.argmax(R12))
    
    # 中值滤波
    tau_median = np.median(tau_list)
    tau = (tau_median - len(R12) // 2) / fs
    
    return tau
```

---

## 7. 应用示例

### 7.1 双麦克风方向估计

```python
# 配置
fs = 16000
d = 0.1  # 10cm间距
c = 343

# 录音（假设已有）
x1, x2 = record_audio(duration=1.0, fs=fs)

# GCC-PHAT
tau, conf = gcc_phat(x1, x2, fs)

# 转换为角度
theta = np.arccos(c * tau / d)
theta_deg = np.degrees(theta)

print(f"TDOA: {tau*1000:.2f} ms")
print(f"方向: {theta_deg:.1f}°")
print(f"置信度: {conf:.3f}")
```

### 7.2 三麦克风2D定位

```python
# 三角形阵列
mic_pos = np.array([
    [0, 0],
    [0.1, 0],
    [0.05, 0.087]
])

# 录音
x1, x2, x3 = record_audio_3ch(duration=1.0, fs=16000)

# 计算TDOA
tau12, _ = gcc_phat(x1, x2, fs=16000)
tau13, _ = gcc_phat(x1, x3, fs=16000)

# 定位
tdoa_list = [tau12, tau13]
position = tdoa_to_position_2d(tdoa_list, mic_pos)

print(f"估计位置: ({position[0]:.2f}, {position[1]:.2f}) m")
```

---

## 8. 总结

GCC-PHAT是声源定位的基础算法：

**核心公式**：

$$R_{12}^{\text{PHAT}}(\tau) = \text{IFFT}\left\{\frac{X_1(f)X_2^*(f)}{|X_1(f)X_2^*(f)|}\right\}$$

**关键优势**：
- 混响鲁棒性好
- 计算高效
- 实现简单

**适用场景**：
- 双麦克风方向估计
- 实时定位系统
- 作为其他算法的预处理

**改进方向**：
- 多频带处理
- 自适应加权
- 深度学习增强
