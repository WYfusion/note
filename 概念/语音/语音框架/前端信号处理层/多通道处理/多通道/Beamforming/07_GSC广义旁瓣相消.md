# GSC 广义旁瓣相消器

## 1. 概述

**GSC (Generalized Sidelobe Canceller)** 是一种自适应波束形成结构，由 Griffiths 和 Jim 于 1982 年提出。它将约束优化问题转化为无约束优化问题，使得自适应滤波器的设计更加简单。

### 1.1 核心思想

将波束形成器分解为两条路径：
1. **固定路径**：保证目标信号无失真通过
2. **自适应路径**：估计并消除残余噪声和干扰

### 1.2 与MVDR的关系

GSC是MVDR的一种等价实现，但结构更清晰，更易于实现和分析。

---

## 2. GSC结构

### 2.1 系统框图

```
输入 X(f,t)
    ↓
    ├─────────────────────┐
    ↓                     ↓
[固定波束形成器]      [阻塞矩阵]
    w_q                   B
    ↓                     ↓
  Z(f,t)               U(f,t)
    ↓                     ↓
    |                [自适应滤波器]
    |                     w_a
    |                     ↓
    └──────(-)───────── Y_a(f,t)
           ↓
        Y(f,t)
```

**数学表达**：
$$Y(f,t) = Z(f,t) - Y_a(f,t)$$

其中：
- $Z(f,t) = \mathbf{w}_q^H \mathbf{X}(f,t)$：固定波束形成器输出
- $\mathbf{U}(f,t) = \mathbf{B}^H \mathbf{X}(f,t)$：阻塞矩阵输出
- $Y_a(f,t) = \mathbf{w}_a^H \mathbf{U}(f,t)$：自适应滤波器输出

### 2.2 完整表达式

$$Y(f,t) = (\mathbf{w}_q - \mathbf{B}\mathbf{w}_a)^H \mathbf{X}(f,t)$$

等价的波束形成权重：
$$\mathbf{w}_{\text{GSC}} = \mathbf{w}_q - \mathbf{B}\mathbf{w}_a$$


---

## 3. 各组件详解

### 3.1 固定波束形成器 $\mathbf{w}_q$

**作用**：保证目标方向信号无失真通过（满足无失真约束）。

**约束条件**：
$$\mathbf{w}_q^H \mathbf{d}(\theta_0) = 1$$

其中 $\mathbf{d}(\theta_0)$ 是目标方向的导向矢量。

**常见选择**：

1. **Delay-and-Sum (DS)**：
$$\mathbf{w}_q = \frac{\mathbf{d}(\theta_0)}{M}$$
其中 $M$ 为麦克风数量。

2. **MVDR 固定部分**：
$$\mathbf{w}_q = \frac{\mathbf{R}_{nn}^{-1}\mathbf{d}(\theta_0)}{\mathbf{d}^H(\theta_0)\mathbf{R}_{nn}^{-1}\mathbf{d}(\theta_0)}$$

3. **超指向性波束形成器**：
$$\mathbf{w}_q = \frac{\boldsymbol{\Gamma}^{-1}\mathbf{d}(\theta_0)}{\mathbf{d}^H(\theta_0)\boldsymbol{\Gamma}^{-1}\mathbf{d}(\theta_0)}$$
其中 $\boldsymbol{\Gamma}$ 为扩散场噪声相干矩阵。

### 3.2 阻塞矩阵 $\mathbf{B}$

**作用**：阻止目标信号进入自适应路径，只让噪声/干扰通过。

**约束条件**：
$$\mathbf{B}^H \mathbf{d}(\theta_0) = \mathbf{0}$$

即阻塞矩阵的列空间与目标导向矢量正交。

**维度**：
- 输入：$M$ 维（麦克风数）
- 输出：$M-1$ 维（或更少）
- $\mathbf{B} \in \mathbb{C}^{M \times (M-1)}$

**常见设计方法**：

#### 方法1：相邻麦克风差分

$$\mathbf{B} = \begin{pmatrix}
1 & 0 & \cdots & 0 \\
-1 & 1 & \cdots & 0 \\
0 & -1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & -1
\end{pmatrix}$$

**优点**：简单直观
**缺点**：仅在宽带信号时近似阻塞目标

#### 方法2：基于导向矢量的正交补

利用 QR 分解或 SVD 构造：
$$\mathbf{B} = \mathbf{I} - \mathbf{d}(\theta_0)(\mathbf{d}^H(\theta_0)\mathbf{d}(\theta_0))^{-1}\mathbf{d}^H(\theta_0)$$

取该投影矩阵的 $M-1$ 个线性无关列向量。

**更精确的方法**：对 $\mathbf{d}(\theta_0)$ 做 QR 分解
$$[\mathbf{d}(\theta_0), \mathbf{B}] = \mathbf{Q}\mathbf{R}$$
取 $\mathbf{Q}$ 的后 $M-1$ 列作为 $\mathbf{B}$。

#### 方法3：频率相关阻塞矩阵

对于宽带信号，阻塞矩阵应随频率变化：
$$\mathbf{B}(f)^H \mathbf{d}(f, \theta_0) = \mathbf{0}, \quad \forall f$$

### 3.3 自适应滤波器 $\mathbf{w}_a$

**作用**：从阻塞矩阵输出中估计残余噪声，并从固定波束输出中减去。

**维度**：$\mathbf{w}_a \in \mathbb{C}^{(M-1) \times 1}$

**优化目标**：最小化输出功率
$$\mathbf{w}_a^{\text{opt}} = \arg\min_{\mathbf{w}_a} \mathbb{E}[|Y(f,t)|^2]$$


---

## 4. 最优解推导

### 4.1 输出功率表达式

输出信号：
$$Y = \mathbf{w}_q^H \mathbf{X} - \mathbf{w}_a^H \mathbf{B}^H \mathbf{X}$$

输出功率：
$$\begin{aligned}
P_Y &= \mathbb{E}[|Y|^2] \\
&= \mathbb{E}[|(\mathbf{w}_q - \mathbf{B}\mathbf{w}_a)^H \mathbf{X}|^2] \\
&= (\mathbf{w}_q - \mathbf{B}\mathbf{w}_a)^H \mathbf{R}_{xx} (\mathbf{w}_q - \mathbf{B}\mathbf{w}_a)
\end{aligned}$$

其中 $\mathbf{R}_{xx} = \mathbb{E}[\mathbf{X}\mathbf{X}^H]$ 是输入协方差矩阵。

### 4.2 展开与求导

展开输出功率：
$$\begin{aligned}
P_Y &= \mathbf{w}_q^H \mathbf{R}_{xx} \mathbf{w}_q 
- \mathbf{w}_q^H \mathbf{R}_{xx} \mathbf{B}\mathbf{w}_a 
- \mathbf{w}_a^H \mathbf{B}^H \mathbf{R}_{xx} \mathbf{w}_q 
+ \mathbf{w}_a^H \mathbf{B}^H \mathbf{R}_{xx} \mathbf{B}\mathbf{w}_a
\end{aligned}$$

定义：
- $\mathbf{R}_{uu} = \mathbf{B}^H \mathbf{R}_{xx} \mathbf{B}$：阻塞矩阵输出的协方差
- $\mathbf{r}_{uz} = \mathbf{B}^H \mathbf{R}_{xx} \mathbf{w}_q$：阻塞输出与固定波束输出的互相关

对 $\mathbf{w}_a$ 求导并令其为零：
$$\frac{\partial P_Y}{\partial \mathbf{w}_a^*} = -\mathbf{r}_{uz} + \mathbf{R}_{uu}\mathbf{w}_a = \mathbf{0}$$

### 4.3 最优自适应滤波器

$$\boxed{\mathbf{w}_a^{\text{opt}} = \mathbf{R}_{uu}^{-1} \mathbf{r}_{uz} = (\mathbf{B}^H \mathbf{R}_{xx} \mathbf{B})^{-1} \mathbf{B}^H \mathbf{R}_{xx} \mathbf{w}_q}$$

### 4.4 等价于 MVDR 的证明

GSC 的等价权重：
$$\mathbf{w}_{\text{GSC}} = \mathbf{w}_q - \mathbf{B}\mathbf{w}_a^{\text{opt}}$$

代入最优解：
$$\mathbf{w}_{\text{GSC}} = \mathbf{w}_q - \mathbf{B}(\mathbf{B}^H \mathbf{R}_{xx} \mathbf{B})^{-1} \mathbf{B}^H \mathbf{R}_{xx} \mathbf{w}_q$$

$$= \left[\mathbf{I} - \mathbf{B}(\mathbf{B}^H \mathbf{R}_{xx} \mathbf{B})^{-1} \mathbf{B}^H \mathbf{R}_{xx}\right] \mathbf{w}_q$$

当 $\mathbf{w}_q = \mathbf{d}/M$ 且满足适当条件时，可证明：
$$\mathbf{w}_{\text{GSC}} = \frac{\mathbf{R}_{xx}^{-1}\mathbf{d}}{\mathbf{d}^H\mathbf{R}_{xx}^{-1}\mathbf{d}} = \mathbf{w}_{\text{MVDR}}$$

---

## 5. 自适应算法

### 5.1 LMS 算法

**更新公式**：
$$\mathbf{w}_a(t+1) = \mathbf{w}_a(t) + \mu \cdot \mathbf{U}(t) \cdot Y^*(t)$$

其中：
- $\mu$：步长参数
- $\mathbf{U}(t) = \mathbf{B}^H \mathbf{X}(t)$：阻塞矩阵输出
- $Y(t) = Z(t) - \mathbf{w}_a^H(t)\mathbf{U}(t)$：GSC 输出

**步长选择**：
$$0 < \mu < \frac{2}{\lambda_{\max}(\mathbf{R}_{uu})}$$

实际中常用：
$$\mu = \frac{\tilde{\mu}}{\|\mathbf{U}(t)\|^2 + \epsilon}$$

其中 $\tilde{\mu} \in (0, 1)$，$\epsilon$ 为正则化常数。

### 5.2 NLMS 算法

**归一化 LMS**：
$$\mathbf{w}_a(t+1) = \mathbf{w}_a(t) + \frac{\mu}{\|\mathbf{U}(t)\|^2 + \epsilon} \mathbf{U}(t) Y^*(t)$$

**优点**：收敛速度更稳定，对输入功率变化不敏感。

### 5.3 RLS 算法

**递归最小二乘**：

初始化：
- $\mathbf{P}(0) = \delta^{-1}\mathbf{I}$（$\delta$ 为小正数）
- $\mathbf{w}_a(0) = \mathbf{0}$

更新：
$$\begin{aligned}
\mathbf{k}(t) &= \frac{\lambda^{-1}\mathbf{P}(t-1)\mathbf{U}(t)}{1 + \lambda^{-1}\mathbf{U}^H(t)\mathbf{P}(t-1)\mathbf{U}(t)} \\
e(t) &= Z(t) - \mathbf{w}_a^H(t-1)\mathbf{U}(t) \\
\mathbf{w}_a(t) &= \mathbf{w}_a(t-1) + \mathbf{k}(t)e^*(t) \\
\mathbf{P}(t) &= \lambda^{-1}\mathbf{P}(t-1) - \lambda^{-1}\mathbf{k}(t)\mathbf{U}^H(t)\mathbf{P}(t-1)
\end{aligned}$$

其中 $\lambda \in (0.95, 1)$ 为遗忘因子。

**优点**：收敛速度快
**缺点**：计算复杂度高 $O((M-1)^2)$

### 5.4 频域自适应（用于宽带信号）

对每个频率 bin 独立应用自适应算法：
$$\mathbf{w}_a(f, t+1) = \mathbf{w}_a(f, t) + \mu(f) \mathbf{U}(f, t) Y^*(f, t)$$


---

## 6. 信号泄漏问题与解决方案

### 6.1 问题描述

**理想情况**：阻塞矩阵完全阻止目标信号
$$\mathbf{B}^H \mathbf{d}(\theta_0) = \mathbf{0}$$

**实际问题**：
1. **导向矢量失配**：实际导向矢量与假设不符
2. **混响**：目标信号经反射后从其他方向到达
3. **近场效应**：近场源的导向矢量与远场假设不同

**后果**：目标信号泄漏到自适应路径，被当作噪声消除，导致**目标信号失真**。

### 6.2 解决方案

#### 方案1：语音活动检测（VAD）

在目标信号活跃时冻结自适应：
$$\mathbf{w}_a(t+1) = \begin{cases}
\mathbf{w}_a(t) + \mu \mathbf{U}(t) Y^*(t), & \text{VAD} = 0 \\
\mathbf{w}_a(t), & \text{VAD} = 1
\end{cases}$$

#### 方案2：传递函数比（TF-GSC）

使用相对传递函数（RTF）代替导向矢量：
$$\mathbf{h} = \frac{\mathbf{a}}{\mathbf{a}_{\text{ref}}}$$

其中 $\mathbf{a}$ 是实际的声学传递函数。

#### 方案3：自适应阻塞矩阵（ABM）

让阻塞矩阵也自适应更新：
$$\mathbf{B}(t+1) = \mathbf{B}(t) - \mu_B \mathbf{X}(t) \mathbf{U}^H(t)$$

约束：保持 $\mathbf{B}^H \mathbf{d} = \mathbf{0}$

#### 方案4：正则化

在自适应滤波器中加入正则化项：
$$\mathbf{w}_a^{\text{opt}} = (\mathbf{R}_{uu} + \delta\mathbf{I})^{-1} \mathbf{r}_{uz}$$

限制自适应滤波器的范数，减少过度消除。

#### 方案5：多通道后置滤波

在 GSC 输出后加入后置滤波器进一步增强：
$$\hat{S}(f,t) = G(f,t) \cdot Y_{\text{GSC}}(f,t)$$

---

## 7. 扩展结构

### 7.1 多约束 GSC (LCMV-GSC)

**多个线性约束**：
$$\mathbf{C}^H \mathbf{w} = \mathbf{f}$$

其中 $\mathbf{C} \in \mathbb{C}^{M \times L}$ 包含 $L$ 个约束。

**固定波束形成器**：
$$\mathbf{w}_q = \mathbf{C}(\mathbf{C}^H\mathbf{C})^{-1}\mathbf{f}$$

**阻塞矩阵**：
$$\mathbf{B}^H \mathbf{C} = \mathbf{0}$$

$\mathbf{B}$ 的列空间是 $\mathbf{C}$ 列空间的正交补，维度为 $M - L$。

### 7.2 宽带 GSC

对于宽带信号，使用 FIR 滤波器：

**时域结构**：
- 固定波束形成器：$\mathbf{w}_q[n]$，长度 $L_q$
- 阻塞矩阵：$\mathbf{B}[n]$，长度 $L_B$
- 自适应滤波器：$\mathbf{w}_a[n]$，长度 $L_a$

**频域实现**：
使用 STFT，对每个频率 bin 独立处理。

### 7.3 双通道 GSC

对于 $M = 2$ 的特殊情况：

**阻塞矩阵**（标量）：
$$B = [1, -e^{j\omega\tau_0}]^T$$

其中 $\tau_0$ 是目标方向的双耳时延。

**自适应滤波器**：标量 $w_a$


---

## 8. 性能分析

### 8.1 输出信噪比

$$\text{SNR}_{\text{out}} = \frac{|\mathbf{w}_{\text{GSC}}^H \mathbf{d}|^2 \sigma_s^2}{\mathbf{w}_{\text{GSC}}^H \mathbf{R}_{nn} \mathbf{w}_{\text{GSC}}}$$

由于无失真约束 $\mathbf{w}_{\text{GSC}}^H \mathbf{d} = 1$：
$$\text{SNR}_{\text{out}} = \frac{\sigma_s^2}{\mathbf{w}_{\text{GSC}}^H \mathbf{R}_{nn} \mathbf{w}_{\text{GSC}}}$$

### 8.2 阵列增益

$$\text{AG} = \frac{\text{SNR}_{\text{out}}}{\text{SNR}_{\text{in}}} = \frac{M \cdot \mathbf{d}^H \mathbf{R}_{nn} \mathbf{d}}{\mathbf{w}_{\text{GSC}}^H \mathbf{R}_{nn} \mathbf{w}_{\text{GSC}}}$$

### 8.3 白噪声增益

$$\text{WNG} = \frac{|\mathbf{w}_{\text{GSC}}^H \mathbf{d}|^2}{\|\mathbf{w}_{\text{GSC}}\|^2} = \frac{1}{\|\mathbf{w}_{\text{GSC}}\|^2}$$

### 8.4 收敛性能

**LMS 收敛时间常数**：
$$\tau_{\text{LMS}} \approx \frac{1}{2\mu\lambda_{\min}(\mathbf{R}_{uu})}$$

**失调误差**：
$$\text{Misadjustment} \approx \frac{\mu \cdot \text{tr}(\mathbf{R}_{uu})}{2}$$

---

## 9. 实现要点

### 9.1 频域实现流程

```python
# 伪代码
def gsc_process(x_multichannel, w_q, B, mu=0.1, eps=1e-6):
    """
    x_multichannel: [M, num_samples] 多通道输入
    w_q: [M, F] 固定波束形成器 (每个频率)
    B: [M, M-1, F] 阻塞矩阵 (每个频率)
    """
    # STFT
    X = stft(x_multichannel)  # [M, F, T]
    M, F, T = X.shape
    
    # 初始化自适应滤波器
    w_a = np.zeros((M-1, F), dtype=complex)
    Y = np.zeros((F, T), dtype=complex)
    
    for t in range(T):
        for f in range(F):
            # 固定波束形成
            Z = w_q[:, f].conj() @ X[:, f, t]
            
            # 阻塞矩阵
            U = B[:, :, f].conj().T @ X[:, f, t]  # [M-1]
            
            # 自适应滤波
            Y_a = w_a[:, f].conj() @ U
            
            # 相减得到输出
            Y[f, t] = Z - Y_a
            
            # NLMS 更新
            norm_U = np.sum(np.abs(U)**2)
            w_a[:, f] += mu / (norm_U + eps) * U * np.conj(Y[f, t])
    
    # ISTFT
    y = istft(Y)
    return y
```

### 9.2 参数选择建议

| 参数 | 典型值 | 说明 |
|------|--------|------|
| 帧长 | 512-2048 | 频率分辨率与时间分辨率折中 |
| 帧移 | 帧长/2 或 帧长/4 | 50% 或 75% 重叠 |
| LMS 步长 $\mu$ | 0.01-0.5 | 收敛速度与稳定性折中 |
| RLS 遗忘因子 $\lambda$ | 0.95-0.999 | 跟踪速度与稳定性折中 |
| 正则化 $\epsilon$ | $10^{-6}$ - $10^{-3}$ | 防止除零 |

### 9.3 初始化

- $\mathbf{w}_a(0) = \mathbf{0}$：从零开始
- 或使用离线估计的初值

---

## 10. 优缺点总结

### 10.1 优点

1. **结构清晰**：固定路径保证无失真，自适应路径消除噪声
2. **无约束优化**：自适应滤波器的更新是无约束的，算法简单
3. **模块化设计**：各组件可独立设计和优化
4. **灵活性**：可方便地扩展到多约束情况

### 10.2 缺点

1. **信号泄漏**：导向矢量失配时目标信号会泄漏并被消除
2. **收敛速度**：LMS 算法收敛较慢
3. **对阻塞矩阵敏感**：阻塞矩阵设计不当会严重影响性能
4. **计算复杂度**：宽带实现需要对每个频率独立处理

---

## 11. 与其他方法的比较

| 特性 | GSC | 直接 MVDR | GEV |
|------|-----|-----------|-----|
| 结构 | 两路径 | 单路径 | 单路径 |
| 约束处理 | 隐式（通过阻塞矩阵） | 显式 | 无 |
| 自适应算法 | 无约束 LMS/RLS | 约束优化 | 特征值分解 |
| 信号失真 | 低（理想情况） | 低 | 可能较高 |
| 实现复杂度 | 中等 | 中等 | 较高 |

---

## 12. 参考文献

1. Griffiths, L. J., & Jim, C. W. (1982). An alternative approach to linearly constrained adaptive beamforming. *IEEE Trans. Antennas Propag.*
2. Hoshuyama, O., et al. (1999). A robust adaptive beamformer for microphone arrays with a blocking matrix using constrained adaptive filters. *IEEE Trans. Signal Process.*
3. Gannot, S., et al. (2001). Signal enhancement using beamforming and nonstationarity with applications to speech. *IEEE Trans. Signal Process.*
