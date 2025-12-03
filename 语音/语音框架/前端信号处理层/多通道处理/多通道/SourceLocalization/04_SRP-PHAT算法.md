# SRP-PHAT 算法

## 1. 概述

**SRP-PHAT (Steered Response Power with Phase Transform)** 是一种基于波束形成的声源定位算法，通过在空间网格上搜索使能量最大的位置来估计声源位置。

### 1.1 核心思想

```
多通道信号
    ↓
[空间网格扫描]
    ↓
对每个候选位置:
  - 计算所有麦克风对的GCC-PHAT
  - 在对应时延处求和
    ↓
[找到最大能量位置]
    ↓
声源位置
```

**物理意义**：当候选位置与真实声源位置一致时，所有麦克风对的相位对齐，能量最大。

---

## 2. 数学推导

### 2.1 可控响应功率 (SRP)

**基本SRP**：

对于候选位置 $\mathbf{r}$，计算波束形成器输出功率：

$$P(\mathbf{r}) = \int_{-\infty}^{\infty} \left|\sum_{p=1}^{P} w_p X_p(f) e^{j2\pi f\tau_p(\mathbf{r})}\right|^2 df$$

其中：
- $\mathbf{r}$：候选声源位置
- $X_p(f)$：第$p$个麦克风的频域信号
- $\tau_p(\mathbf{r})$：从位置$\mathbf{r}$到麦克风$p$的传播时延
- $w_p$：权重

**时延计算**：

$$\tau_p(\mathbf{r}) = \frac{\|\mathbf{r} - \mathbf{r}_p\|}{c}$$

### 2.2 展开为麦克风对

将求和展开：

$$P(\mathbf{r}) = \int_{-\infty}^{\infty} \sum_{i=1}^{P}\sum_{j=1}^{P} w_i w_j X_i(f) X_j^*(f) e^{j2\pi f(\tau_i(\mathbf{r}) - \tau_j(\mathbf{r}))} df$$

定义麦克风对的时延差：

$$\tau_{ij}(\mathbf{r}) = \tau_i(\mathbf{r}) - \tau_j(\mathbf{r}) = \frac{\|\mathbf{r} - \mathbf{r}_i\| - \|\mathbf{r} - \mathbf{r}_j\|}{c}$$

则：

$$P(\mathbf{r}) = \sum_{i=1}^{P}\sum_{j=1}^{P} w_i w_j \int_{-\infty}^{\infty} X_i(f) X_j^*(f) e^{j2\pi f\tau_{ij}(\mathbf{r})} df$$

识别出这是互相关函数：

$$P(\mathbf{r}) = \sum_{i=1}^{P}\sum_{j=1}^{P} w_i w_j R_{ij}(\tau_{ij}(\mathbf{r}))$$

### 2.3 SRP-PHAT

应用PHAT加权（$w_i = w_j = 1$）：

$$P_{\text{PHAT}}(\mathbf{r}) = \sum_{i=1}^{P}\sum_{j=1}^{P} R_{ij}^{\text{PHAT}}(\tau_{ij}(\mathbf{r}))$$

其中：

$$R_{ij}^{\text{PHAT}}(\tau) = \int_{-\infty}^{\infty} \frac{X_i(f)X_j^*(f)}{|X_i(f)X_j^*(f)|} e^{j2\pi f\tau} df$$

**简化**（只考虑不同麦克风对）：

$$P_{\text{PHAT}}(\mathbf{r}) = \sum_{i=1}^{P-1}\sum_{j=i+1}^{P} R_{ij}^{\text{PHAT}}(\tau_{ij}(\mathbf{r}))$$

**声源位置估计**：

$$\hat{\mathbf{r}}_s = \arg\max_{\mathbf{r}} P_{\text{PHAT}}(\mathbf{r})$$

### 2.4 物理解释

当候选位置 $\mathbf{r}$ 等于真实声源位置 $\mathbf{r}_s$ 时：

$$\tau_{ij}(\mathbf{r}_s) = \tau_{ij}^{\text{true}}$$

此时所有麦克风对的GCC-PHAT在正确时延处取值，相位对齐，能量最大。

---

## 3. 算法实现

### 3.1 基本流程

```
输入: X[p, f, t] - 多通道频域信号
      mic_pos[p, 3] - 麦克风位置
      search_grid[N, 3] - 搜索网格
输出: r_est - 估计的声源位置

1. 预计算所有麦克风对的GCC-PHAT
   for i, j in pairs:
       R_ij = GCC-PHAT(X[i], X[j])

2. 对每个候选位置
   for r in search_grid:
       P[r] = 0
       for i, j in pairs:
           τ_ij = compute_tdoa(r, mic_pos[i], mic_pos[j])
           P[r] += R_ij(τ_ij)

3. 找到最大值
   r_est = search_grid[argmax(P)]
```

### 3.2 Python实现

```python
import numpy as np
from scipy.fft import fft, ifft, fftshift
from itertools import combinations

class SRPPHAT:
    def __init__(self, mic_positions, fs=16000, c=343):
        """
        SRP-PHAT定位器
        
        参数:
            mic_positions: [P, 3] - 麦克风位置 (m)
            fs: 采样率 (Hz)
            c: 声速 (m/s)
        """
        self.mic_pos = np.array(mic_positions)
        self.P = len(mic_positions)
        self.fs = fs
        self.c = c
        
        # 麦克风对
        self.pairs = list(combinations(range(self.P), 2))
        
    def compute_gcc_phat(self, X):
        """
        计算所有麦克风对的GCC-PHAT
        
        参数:
            X: [P, N] - 频域信号
        
        返回:
            gcc_dict: {(i,j): R_ij} - GCC-PHAT字典
        """
        gcc_dict = {}
        
        for i, j in self.pairs:
            # 互功率谱
            G_ij = X[i] * np.conj(X[j])
            
            # PHAT加权
            G_ij_phat = G_ij / (np.abs(G_ij) + 1e-10)
            
            # IFFT
            R_ij = np.real(ifft(G_ij_phat))
            R_ij = fftshift(R_ij)
            
            gcc_dict[(i, j)] = R_ij
        
        return gcc_dict
    
    def compute_tdoa(self, source_pos, mic_i, mic_j):
        """
        计算理论TDOA
        
        参数:
            source_pos: [3] - 声源位置
            mic_i, mic_j: [3] - 麦克风位置
        
        返回:
            tau_ij: TDOA (秒)
        """
        d_i = np.linalg.norm(source_pos - mic_i)
        d_j = np.linalg.norm(source_pos - mic_j)
        tau_ij = (d_i - d_j) / self.c
        return tau_ij
    
    def localize(self, X, search_grid):
        """
        SRP-PHAT定位
        
        参数:
            X: [P, N] - 频域信号
            search_grid: [M, 3] - 搜索网格
        
        返回:
            position: [3] - 估计位置
            power_map: [M] - 功率图
        """
        # 计算GCC-PHAT
        gcc_dict = self.compute_gcc_phat(X)
        
        N = X.shape[1]
        M = len(search_grid)
        power_map = np.zeros(M)
        
        # 对每个候选位置
        for m, candidate_pos in enumerate(search_grid):
            power = 0
            
            # 累加所有麦克风对的贡献
            for i, j in self.pairs:
                # 计算理论TDOA
                tau_ij = self.compute_tdoa(
                    candidate_pos, 
                    self.mic_pos[i], 
                    self.mic_pos[j]
                )
                
                # 转换为采样点
                tau_samples = int(tau_ij * self.fs)
                
                # 从GCC-PHAT中取值
                R_ij = gcc_dict[(i, j)]
                idx = N // 2 + tau_samples
                
                if 0 <= idx < N:
                    power += R_ij[idx]
            
            power_map[m] = power
        
        # 找到最大值
        max_idx = np.argmax(power_map)
        position = search_grid[max_idx]
        
        return position, power_map
    
    def create_search_grid_2d(self, x_range, y_range, z=0, resolution=0.1):
        """
        创建2D搜索网格
        
        参数:
            x_range: [x_min, x_max]
            y_range: [y_min, y_max]
            z: 固定z坐标
            resolution: 网格分辨率 (m)
        
        返回:
            grid: [M, 3] - 搜索网格
        """
        x = np.arange(x_range[0], x_range[1], resolution)
        y = np.arange(y_range[0], y_range[1], resolution)
        
        xx, yy = np.meshgrid(x, y)
        zz = np.full_like(xx, z)
        
        grid = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
        
        return grid, (xx, yy)
```

### 3.3 使用示例

```python
# 配置
mic_pos = np.array([
    [0, 0, 0],
    [0.2, 0, 0],
    [0.1, 0.17, 0],
    [0.1, 0.057, 0]
])

srp = SRPPHAT(mic_pos, fs=16000)

# 录音（假设已有）
signals = record_audio(n_channels=4, duration=1.0, fs=16000)

# FFT
X = np.array([fft(sig) for sig in signals])

# 创建搜索网格
search_grid, (xx, yy) = srp.create_search_grid_2d(
    x_range=[-1, 3],
    y_range=[-1, 3],
    z=0,
    resolution=0.05
)

# 定位
position, power_map = srp.localize(X, search_grid)

print(f"估计位置: {position}")

# 可视化
import matplotlib.pyplot as plt
power_2d = power_map.reshape(xx.shape)
plt.contourf(xx, yy, power_2d, levels=20)
plt.plot(mic_pos[:, 0], mic_pos[:, 1], 'r^', markersize=10)
plt.plot(position[0], position[1], 'g*', markersize=15)
plt.colorbar(label='SRP Power')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('SRP-PHAT Power Map')
plt.show()
```

---

## 4. 优化方法

### 4.1 粗到精搜索

**两阶段搜索**：

1. **粗搜索**：大范围，低分辨率
2. **精搜索**：小范围（粗搜索结果附近），高分辨率

```python
def coarse_to_fine_search(srp, X, initial_range, initial_res=0.2, fine_res=0.02):
    """
    粗到精搜索
    
    参数:
        srp: SRPPHAT对象
        X: 频域信号
        initial_range: [[x_min, x_max], [y_min, y_max]]
        initial_res: 粗搜索分辨率
        fine_res: 精搜索分辨率
    
    返回:
        position: 估计位置
    """
    # 粗搜索
    grid_coarse, _ = srp.create_search_grid_2d(
        initial_range[0], initial_range[1], 
        resolution=initial_res
    )
    pos_coarse, _ = srp.localize(X, grid_coarse)
    
    # 精搜索范围
    fine_range_x = [pos_coarse[0] - initial_res, pos_coarse[0] + initial_res]
    fine_range_y = [pos_coarse[1] - initial_res, pos_coarse[1] + initial_res]
    
    grid_fine, _ = srp.create_search_grid_2d(
        fine_range_x, fine_range_y,
        resolution=fine_res
    )
    pos_fine, _ = srp.localize(X, grid_fine)
    
    return pos_fine
```

### 4.2 随机采样

使用随机采样减少计算量：

```python
def random_search(srp, X, search_range, n_samples=1000):
    """
    随机采样搜索
    
    参数:
        srp: SRPPHAT对象
        X: 频域信号
        search_range: [[x_min, x_max], [y_min, y_max]]
        n_samples: 采样点数
    
    返回:
        position: 估计位置
    """
    # 随机采样
    x = np.random.uniform(search_range[0][0], search_range[0][1], n_samples)
    y = np.random.uniform(search_range[1][0], search_range[1][1], n_samples)
    z = np.zeros(n_samples)
    
    search_grid = np.stack([x, y, z], axis=1)
    
    # 定位
    position, _ = srp.localize(X, search_grid)
    
    return position
```

### 4.3 梯度优化

使用梯度下降优化：

```python
def gradient_based_srp(srp, X, initial_guess, learning_rate=0.1, n_iter=50):
    """
    基于梯度的SRP优化
    
    参数:
        srp: SRPPHAT对象
        X: 频域信号
        initial_guess: [3] - 初始位置
        learning_rate: 学习率
        n_iter: 迭代次数
    
    返回:
        position: 优化后的位置
    """
    gcc_dict = srp.compute_gcc_phat(X)
    position = initial_guess.copy()
    
    for _ in range(n_iter):
        # 计算梯度（数值微分）
        delta = 0.01
        grad = np.zeros(3)
        
        for dim in range(3):
            pos_plus = position.copy()
            pos_plus[dim] += delta
            
            pos_minus = position.copy()
            pos_minus[dim] -= delta
            
            power_plus = srp._compute_power(pos_plus, gcc_dict)
            power_minus = srp._compute_power(pos_minus, gcc_dict)
            
            grad[dim] = (power_plus - power_minus) / (2 * delta)
        
        # 梯度上升（最大化功率）
        position += learning_rate * grad
    
    return position
```

---

## 5. 性能分析

### 5.1 优势

1. **鲁棒性强**：对混响和噪声鲁棒
2. **适用任意阵列**：不限制阵列几何
3. **多源能力**：可以检测多个峰值
4. **无需DOA**：直接估计位置

### 5.2 局限

1. **计算复杂度高**：$O(MN)$，$M$是网格点数
2. **分辨率受限**：网格分辨率限制精度
3. **近场假设**：需要准确的传播模型
4. **实时性差**：不适合高速移动源

### 5.3 计算复杂度

**网格搜索**：

- 网格点数：$M = \frac{(x_{\max} - x_{\min})(y_{\max} - y_{\min})}{r^2}$
- 麦克风对数：$N_{\text{pairs}} = P(P-1)/2$
- 总复杂度：$O(M \cdot N_{\text{pairs}})$

**示例**：
- 搜索范围：$4 \times 4$ m²
- 分辨率：$r = 0.1$ m
- 麦克风：$P = 4$
- 网格点：$M = 1600$
- 麦克风对：$N_{\text{pairs}} = 6$
- 总计算：$1600 \times 6 = 9600$ 次查表

---

## 6. 改进方法

### 6.1 SRP-PHAT-HSDA

**分层搜索与定向算法 (Hierarchical Search with Directional Adaptation)**

结合粗到精搜索和方向信息：

1. 粗搜索找到大致区域
2. 使用DOA信息缩小搜索范围
3. 精搜索得到精确位置

### 6.2 SRP-PHAT with Interpolation

使用插值提高分辨率：

```python
def srp_with_interpolation(srp, X, search_grid, power_map):
    """
    使用插值提高精度
    
    参数:
        srp: SRPPHAT对象
        X: 频域信号
        search_grid: 搜索网格
        power_map: 功率图
    
    返回:
        refined_position: 精细化位置
    """
    from scipy.interpolate import RegularGridInterpolator
    
    # 找到最大值附近区域
    max_idx = np.argmax(power_map)
    
    # 创建插值器
    # ... (需要重塑为规则网格)
    
    # 在最大值附近进行精细插值
    # ...
    
    return refined_position
```

### 6.3 时频掩码增强

结合深度学习掩码：

```python
def masked_srp_phat(srp, X, mask):
    """
    掩码增强的SRP-PHAT
    
    参数:
        srp: SRPPHAT对象
        X: [P, F, T] - 时频域信号
        mask: [F, T] - 时频掩码
    
    返回:
        position: 估计位置
    """
    # 应用掩码
    X_masked = X * mask[np.newaxis, :, :]
    
    # 对每一帧进行SRP-PHAT
    positions = []
    for t in range(X.shape[2]):
        X_frame = X_masked[:, :, t]
        pos, _ = srp.localize(X_frame, search_grid)
        positions.append(pos)
    
    # 中值滤波
    position = np.median(positions, axis=0)
    
    return position
```

---

## 7. 应用场景

### 7.1 智能音箱

**配置**：
- 6-8麦克风圆阵
- 2D定位（桌面平面）
- 实时追踪

**优化**：
- 粗到精搜索
- 卡尔曼滤波平滑
- 结合VAD

### 7.2 会议室

**配置**：
- 分布式阵列
- 3D定位
- 多说话人

**优化**：
- 多峰检测
- 说话人追踪
- 摄像头联动

### 7.3 机器人听觉

**配置**：
- 移动平台
- 实时定位
- 环境适应

**优化**：
- 快速搜索算法
- 运动补偿
- SLAM融合

---

## 8. 与其他方法对比

| 特性 | GCC-PHAT | SRP-PHAT | MUSIC |
|------|----------|----------|-------|
| 计算复杂度 | 低 | 高 | 中 |
| 混响鲁棒性 | 中 | 高 | 低 |
| 多源能力 | 无 | 有 | 有 |
| 实时性 | 好 | 差 | 中 |
| 精度 | 中 | 高 | 高 |
| 阵列要求 | 2麦克风 | 任意 | 多麦克风 |

**选择建议**：
- 实时性要求高 → GCC-PHAT
- 混响环境 → SRP-PHAT
- 高分辨率 → MUSIC
- 多声源 → SRP-PHAT或MUSIC

---

## 9. 总结

SRP-PHAT是一种强大的声源定位算法：

**核心公式**：

$$\hat{\mathbf{r}}_s = \arg\max_{\mathbf{r}} \sum_{i<j} R_{ij}^{\text{PHAT}}(\tau_{ij}(\mathbf{r}))$$

**关键优势**：
- 混响鲁棒性强
- 适用任意阵列
- 可检测多源

**主要挑战**：
- 计算复杂度高
- 需要优化搜索策略

**实用技巧**：
- 粗到精搜索
- 结合DOA预估
- 时频掩码增强
