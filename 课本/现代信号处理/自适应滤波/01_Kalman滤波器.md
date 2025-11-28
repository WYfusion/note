# Kalman滤波器

## 1. 引言

Kalman滤波器是由Rudolf E. Kálmán于1960年提出的一种最优递归状态估计算法。它在含噪声的线性动态系统中，通过融合系统模型预测和观测数据，得到状态的最小均方误差估计。

**核心思想**：在贝叶斯框架下，利用系统的状态空间模型，递归地进行"预测-更新"两步操作。

**应用领域**：
- 导航与定位（GPS/INS组合导航）
- 目标跟踪（雷达、声纳）
- 信号处理（自适应滤波、系统辨识）
- 经济预测、机器人控制

---

## 2. 状态空间模型

### 2.1 离散时间线性系统

考虑如下离散时间线性动态系统：

**状态方程**（描述系统状态如何演化）：
$$\mathbf{x}_k = \mathbf{F}_{k-1}\mathbf{x}_{k-1} + \mathbf{B}_{k-1}\mathbf{u}_{k-1} + \mathbf{w}_{k-1}$$

**观测方程**（描述如何从状态得到观测）：
$$\mathbf{z}_k = \mathbf{H}_k\mathbf{x}_k + \mathbf{v}_k$$

其中：
| 符号 | 维度 | 含义 |
| ------ | ------ | ------ |
| $\mathbf{x}_k$ | $n \times 1$ | $k$时刻的状态向量 |
| $\mathbf{z}_k$ | $m \times 1$ | $k$时刻的观测向量 |
| $\mathbf{u}_k$ | $p \times 1$ | $k$时刻的控制输入 |
| $\mathbf{F}_k$ | $n \times n$ | 状态转移矩阵 |
| $\mathbf{B}_k$ | $n \times p$ | 控制输入矩阵 |
| $\mathbf{H}_k$ | $m \times n$ | 观测矩阵 |
| $\mathbf{w}_k$ | $n \times 1$ | 过程噪声 |
| $\mathbf{v}_k$ | $m \times 1$ | 观测噪声 |

### 2.2 噪声假设

过程噪声和观测噪声满足：
- **零均值高斯白噪声**：
  $$\mathbf{w}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_k), \quad \mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_k)$$

- **相互独立**：
  $$E[\mathbf{w}_i\mathbf{v}_j^T] = \mathbf{0}, \quad \forall i, j$$

- **与初始状态独立**：
  $$E[\mathbf{w}_k\mathbf{x}_0^T] = \mathbf{0}, \quad E[\mathbf{v}_k\mathbf{x}_0^T] = \mathbf{0}$$

其中$\mathbf{Q}_k$为过程噪声协方差矩阵，$\mathbf{R}_k$为观测噪声协方差矩阵。

---

## 3. Kalman滤波推导

### 3.1 问题定义

**目标**：基于到$k$时刻的所有观测$\mathbf{Z}_k = \{\mathbf{z}_1, \mathbf{z}_2, ..., \mathbf{z}_k\}$，求状态$\mathbf{x}_k$的最小均方误差（MMSE）估计。

定义：
- **先验估计**：$\hat{\mathbf{x}}_k^- = E[\mathbf{x}_k | \mathbf{Z}_{k-1}]$（基于$k-1$时刻观测的预测）
- **后验估计**：$\hat{\mathbf{x}}_k = E[\mathbf{x}_k | \mathbf{Z}_k]$（融合$k$时刻观测后的估计）
- **先验误差协方差**：$\mathbf{P}_k^- = E[(\mathbf{x}_k - \hat{\mathbf{x}}_k^-)(\mathbf{x}_k - \hat{\mathbf{x}}_k^-)^T]$
- **后验误差协方差**：$\mathbf{P}_k = E[(\mathbf{x}_k - \hat{\mathbf{x}}_k)(\mathbf{x}_k - \hat{\mathbf{x}}_k)^T]$

### 3.2 预测步骤（Time Update）

**状态预测**：
$$\hat{\mathbf{x}}_k^- = \mathbf{F}_{k-1}\hat{\mathbf{x}}_{k-1} + \mathbf{B}_{k-1}\mathbf{u}_{k-1}$$

**推导**：
$$\begin{aligned}
\hat{\mathbf{x}}_k^- &= E[\mathbf{x}_k | \mathbf{Z}_{k-1}] \\
&= E[\mathbf{F}_{k-1}\mathbf{x}_{k-1} + \mathbf{B}_{k-1}\mathbf{u}_{k-1} + \mathbf{w}_{k-1} | \mathbf{Z}_{k-1}] \\
&= \mathbf{F}_{k-1}E[\mathbf{x}_{k-1} | \mathbf{Z}_{k-1}] + \mathbf{B}_{k-1}\mathbf{u}_{k-1} + E[\mathbf{w}_{k-1}] \\
&= \mathbf{F}_{k-1}\hat{\mathbf{x}}_{k-1} + \mathbf{B}_{k-1}\mathbf{u}_{k-1}
\end{aligned}$$

**协方差预测**：
$$\mathbf{P}_k^- = \mathbf{F}_{k-1}\mathbf{P}_{k-1}\mathbf{F}_{k-1}^T + \mathbf{Q}_{k-1}$$

**推导**：
定义先验误差$\mathbf{e}_k^- = \mathbf{x}_k - \hat{\mathbf{x}}_k^-$，则：
$$\begin{aligned}
\mathbf{e}_k^- &= \mathbf{F}_{k-1}\mathbf{x}_{k-1} + \mathbf{w}_{k-1} - \mathbf{F}_{k-1}\hat{\mathbf{x}}_{k-1} \\
&= \mathbf{F}_{k-1}(\mathbf{x}_{k-1} - \hat{\mathbf{x}}_{k-1}) + \mathbf{w}_{k-1} \\
&= \mathbf{F}_{k-1}\mathbf{e}_{k-1} + \mathbf{w}_{k-1}
\end{aligned}$$

因此：
$$\begin{aligned}
\mathbf{P}_k^- &= E[\mathbf{e}_k^-(\mathbf{e}_k^-)^T] \\
&= E[(\mathbf{F}_{k-1}\mathbf{e}_{k-1} + \mathbf{w}_{k-1})(\mathbf{F}_{k-1}\mathbf{e}_{k-1} + \mathbf{w}_{k-1})^T] \\
&= \mathbf{F}_{k-1}E[\mathbf{e}_{k-1}\mathbf{e}_{k-1}^T]\mathbf{F}_{k-1}^T + E[\mathbf{w}_{k-1}\mathbf{w}_{k-1}^T] \\
&= \mathbf{F}_{k-1}\mathbf{P}_{k-1}\mathbf{F}_{k-1}^T + \mathbf{Q}_{k-1}
\end{aligned}$$

### 3.3 更新步骤（Measurement Update）

**核心思想**：将后验估计表示为先验估计加上一个修正项：
$$\hat{\mathbf{x}}_k = \hat{\mathbf{x}}_k^- + \mathbf{K}_k(\mathbf{z}_k - \mathbf{H}_k\hat{\mathbf{x}}_k^-)$$

其中$\mathbf{K}_k$为**Kalman增益**，$(\mathbf{z}_k - \mathbf{H}_k\hat{\mathbf{x}}_k^-)$称为**新息（Innovation）**或**残差**。

**Kalman增益推导**：

目标是选择$\mathbf{K}_k$使后验误差协方差$\mathbf{P}_k$的迹最小。

后验误差：
$$\begin{aligned}
\mathbf{e}_k &= \mathbf{x}_k - \hat{\mathbf{x}}_k \\
&= \mathbf{x}_k - \hat{\mathbf{x}}_k^- - \mathbf{K}_k(\mathbf{z}_k - \mathbf{H}_k\hat{\mathbf{x}}_k^-) \\
&= \mathbf{e}_k^- - \mathbf{K}_k(\mathbf{H}_k\mathbf{x}_k + \mathbf{v}_k - \mathbf{H}_k\hat{\mathbf{x}}_k^-) \\
&= \mathbf{e}_k^- - \mathbf{K}_k\mathbf{H}_k\mathbf{e}_k^- - \mathbf{K}_k\mathbf{v}_k \\
&= (\mathbf{I} - \mathbf{K}_k\mathbf{H}_k)\mathbf{e}_k^- - \mathbf{K}_k\mathbf{v}_k
\end{aligned}$$

后验误差协方差：
$$\begin{aligned}
\mathbf{P}_k &= E[\mathbf{e}_k\mathbf{e}_k^T] \\
&= (\mathbf{I} - \mathbf{K}_k\mathbf{H}_k)\mathbf{P}_k^-(\mathbf{I} - \mathbf{K}_k\mathbf{H}_k)^T + \mathbf{K}_k\mathbf{R}_k\mathbf{K}_k^T
\end{aligned}$$

对$\mathbf{K}_k$求导并令其为零：
$$\frac{\partial \text{tr}(\mathbf{P}_k)}{\partial \mathbf{K}_k} = -2(\mathbf{H}_k\mathbf{P}_k^-)^T + 2\mathbf{K}_k(\mathbf{H}_k\mathbf{P}_k^-\mathbf{H}_k^T + \mathbf{R}_k) = \mathbf{0}$$

解得**最优Kalman增益**：
$$\boxed{\mathbf{K}_k = \mathbf{P}_k^-\mathbf{H}_k^T(\mathbf{H}_k\mathbf{P}_k^-\mathbf{H}_k^T + \mathbf{R}_k)^{-1}}$$

**后验协方差更新**：
$$\boxed{\mathbf{P}_k = (\mathbf{I} - \mathbf{K}_k\mathbf{H}_k)\mathbf{P}_k^-}$$

或使用Joseph形式（数值更稳定）：
$$\mathbf{P}_k = (\mathbf{I} - \mathbf{K}_k\mathbf{H}_k)\mathbf{P}_k^-(\mathbf{I} - \mathbf{K}_k\mathbf{H}_k)^T + \mathbf{K}_k\mathbf{R}_k\mathbf{K}_k^T$$

---

## 4. Kalman滤波算法总结

### 4.1 算法流程

```
初始化: x̂₀, P₀

对于 k = 1, 2, 3, ...：
    ┌─────────────────────────────────────────┐
    │           预测步骤 (Predict)              │
    ├─────────────────────────────────────────┤
    │  x̂ₖ⁻ = Fₖ₋₁x̂ₖ₋₁ + Bₖ₋₁uₖ₋₁             │
    │  Pₖ⁻ = Fₖ₋₁Pₖ₋₁Fₖ₋₁ᵀ + Qₖ₋₁             │
    └─────────────────────────────────────────┘
                        ↓
    ┌─────────────────────────────────────────┐
    │           更新步骤 (Update)              │
    ├─────────────────────────────────────────┤
    │  Kₖ = Pₖ⁻Hₖᵀ(HₖPₖ⁻Hₖᵀ + Rₖ)⁻¹          │
    │  x̂ₖ = x̂ₖ⁻ + Kₖ(zₖ - Hₖx̂ₖ⁻)             │
    │  Pₖ = (I - KₖHₖ)Pₖ⁻                     │
    └─────────────────────────────────────────┘
```

### 4.2 公式汇总

| 步骤       | 公式                                                                                                          | 说明      |
| -------- | ----------------------------------------------------------------------------------------------------------- | ------- |
| 状态预测     | $\hat{\mathbf{x}}_k^- = \mathbf{F}_{k-1}\hat{\mathbf{x}}_{k-1} + \mathbf{B}_{k-1}\mathbf{u}_{k-1}$          | 先验状态估计  |
| 协方差预测    | $\mathbf{P}_k^- = \mathbf{F}_{k-1}\mathbf{P}_{k-1}\mathbf{F}_{k-1}^T + \mathbf{Q}_{k-1}$                    | 先验误差协方差 |
| Kalman增益 | $\mathbf{K}_k = \mathbf{P}_k^-\mathbf{H}_k^T(\mathbf{H}_k\mathbf{P}_k^-\mathbf{H}_k^T + \mathbf{R}_k)^{-1}$ | 最优增益    |
| 状态更新     | $\hat{\mathbf{x}}_k = \hat{\mathbf{x}}_k^- + \mathbf{K}_k(\mathbf{z}_k - \mathbf{H}_k\hat{\mathbf{x}}_k^-)$ | 后验状态估计  |
| 协方差更新    | $\mathbf{P}_k = (\mathbf{I} - \mathbf{K}_k\mathbf{H}_k)\mathbf{P}_k^-$                                      | 后验误差协方差 |

---

## 5. 物理意义与直观理解

### 5.1 Kalman增益的意义

$$\mathbf{K}_k = \frac{\text{预测不确定性}}{\text{预测不确定性} + \text{观测不确定性}}$$

- 当$\mathbf{R}_k \to \mathbf{0}$（观测非常精确）：$\mathbf{K}_k \to \mathbf{H}_k^{-1}$，完全信任观测
- 当$\mathbf{P}_k^- \to \mathbf{0}$（预测非常精确）：$\mathbf{K}_k \to \mathbf{0}$，完全信任预测
- 一般情况：在预测和观测之间进行最优加权

### 5.2 新息的意义

新息$\boldsymbol{\nu}_k = \mathbf{z}_k - \mathbf{H}_k\hat{\mathbf{x}}_k^-$表示实际观测与预测观测之间的差异：
- 新息为零：预测完美，无需修正
- 新息较大：预测偏差大，需要较大修正

新息协方差：
$$\mathbf{S}_k = \mathbf{H}_k\mathbf{P}_k^-\mathbf{H}_k^T + \mathbf{R}_k$$

---

## 6. 最优性证明

### 6.1 MMSE最优性

**定理**：在线性高斯假设下，Kalman滤波器给出的估计是最小均方误差（MMSE）估计。

**证明要点**：
1. 对于高斯分布，条件期望$E[\mathbf{x}_k|\mathbf{Z}_k]$是MMSE估计
2. 线性变换保持高斯性
3. Kalman滤波递归地计算条件期望

### 6.2 BLUE性质

即使噪声不是高斯分布，只要满足零均值和给定协方差，Kalman滤波器仍是**最佳线性无偏估计（BLUE）**。

---

## 7. 扩展与变体

### 7.1 扩展Kalman滤波（EKF）

用于非线性系统：
$$\mathbf{x}_k = f(\mathbf{x}_{k-1}, \mathbf{u}_{k-1}) + \mathbf{w}_{k-1}$$
$$\mathbf{z}_k = h(\mathbf{x}_k) + \mathbf{v}_k$$

通过一阶Taylor展开线性化：
$$\mathbf{F}_k = \frac{\partial f}{\partial \mathbf{x}}\bigg|_{\hat{\mathbf{x}}_{k-1}}, \quad \mathbf{H}_k = \frac{\partial h}{\partial \mathbf{x}}\bigg|_{\hat{\mathbf{x}}_k^-}$$

### 7.2 无迹Kalman滤波（UKF）

使用sigma点采样来近似非线性变换后的均值和协方差，避免计算Jacobian矩阵。

### 7.3 信息滤波器

使用信息矩阵$\mathbf{Y}_k = \mathbf{P}_k^{-1}$和信息向量$\hat{\mathbf{y}}_k = \mathbf{P}_k^{-1}\hat{\mathbf{x}}_k$，适合多传感器融合。

---

## 8. 参考文献

1. Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1), 35-45.
2. Welch, G., & Bishop, G. (1995). An introduction to the Kalman filter. *University of North Carolina at Chapel Hill*.
3. Simon, D. (2006). *Optimal State Estimation: Kalman, H∞, and Nonlinear Approaches*. Wiley.
