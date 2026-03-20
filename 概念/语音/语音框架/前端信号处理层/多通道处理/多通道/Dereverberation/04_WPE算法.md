# WPE算法 - 加权预测误差

## 1. 概述

**WPE (Weighted Prediction Error)** 是一种基于线性预测的盲去混响算法，由Nakatani等人提出，是目前最有效的去混响方法之一。

### 1.1 核心思想

利用语音信号的时变特性和混响的长时相关性，通过线性预测去除后期混响。

**关键假设**：
- 语音信号短时平稳
- 混响具有长时相关性
- 后期混响可以从早期信号预测

---

## 2. 数学模型

### 2.1 信号模型

**混响信号**：
$$y_d(n) = x_d(n) + r_d(n)$$

其中：
- $x_d(n)$：期望信号（直达声+早期反射）
- $r_d(n)$：后期混响
- $d$：麦克风索引

### 2.2 线性预测模型

**预测混响**：
$$\hat{r}_d(n) = \sum_{\tau=\Delta}^{\Delta+K-1} \sum_{d'=1}^{D} g_{d,d'}(\tau) y_{d'}(n-\tau)$$

其中：
- $\Delta$：预测延迟
- $K$：滤波器长度
- $g_{d,d'}(\tau)$：预测滤波器系数

**去混响**：
$$\hat{x}_d(n) = y_d(n) - \hat{r}_d(n)$$

---

## 3. WPE优化问题

### 3.1 最大似然估计

**目标函数**：
$$\min_{\mathbf{G}, \boldsymbol{\lambda}} \sum_{n,f} \frac{|\hat{X}_d(n,f)|^2}{\lambda_d(n,f)} + \log\lambda_d(n,f)$$

其中：
- $\hat{X}_d(n,f)$：去混响后的STFT系数
- $\lambda_d(n,f)$：时变方差

### 3.2 迭代求解

**E步**：更新方差
$$\lambda_d(n,f) = |\hat{X}_d(n,f)|^2$$

**M步**：更新滤波器
$$\mathbf{g}_f = \left(\sum_n \frac{\mathbf{y}_f(n)\mathbf{y}_f^H(n)}{\lambda_f(n)}\right)^{-1} \sum_n \frac{\mathbf{y}_f(n)y_f^*(n)}{\lambda_f(n)}$$

---

## 4. 算法实现

### 4.1 批处理WPE

```python
import numpy as np

def wpe_batch(Y, delay=3, iterations=3, K=10):
    """
    批处理WPE算法
    
    参数:
        Y: [F, N, D] - STFT系数
        delay: 预测延迟
        iterations: 迭代次数
        K: 滤波器长度
    
    返回:
        X: 去混响后的信号
    """
    F, N, D = Y.shape
    X = Y.copy()
    
    for iteration in range(iterations):
        # 对每个频率独立处理
        for f in range(F):
            # 构造预测矩阵
            Y_pred = []
            for n in range(delay + K - 1, N):
                y_vec = []
                for tau in range(delay, delay + K):
                    y_vec.append(Y[f, n-tau, :])
                Y_pred.append(np.concatenate(y_vec))
            Y_pred = np.array(Y_pred)  # [N', D*K]
            
            # 计算方差
            lambda_f = np.mean(np.abs(X[f, delay+K-1:, :])**2, axis=1, keepdims=True)
            
            # 更新滤波器
            R = np.zeros((D*K, D*K), dtype=complex)
            r = np.zeros((D*K, D), dtype=complex)
            
            for n in range(len(Y_pred)):
                weight = 1.0 / (lambda_f[n] + 1e-10)
                R += weight * np.outer(Y_pred[n], Y_pred[n].conj())
                r += weight * np.outer(Y_pred[n], Y[f, n+delay+K-1, :].conj())
            
            G = np.linalg.solve(R + 1e-6*np.eye(D*K), r)
            
            # 应用滤波器
            for n in range(len(Y_pred)):
                X[f, n+delay+K-1, :] = Y[f, n+delay+K-1, :] - Y_pred[n] @ G
    
    return X
```

### 4.2 在线WPE

**递归更新**：
$$\mathbf{R}_f(n) = \alpha \mathbf{R}_f(n-1) + (1-\alpha) \frac{\mathbf{y}_f(n)\mathbf{y}_f^H(n)}{\lambda_f(n)}$$

---

## 5. 性能分析

### 5.1 参数影响

**预测延迟 $\Delta$**：
- 太小：去除直达声
- 太大：预测不准确
- 典型值：3-5帧

**滤波器长度 $K$**：
- 太短：去混响不充分
- 太长：过拟合
- 典型值：10-20

### 5.2 计算复杂度

**批处理**：$O(FN(DK)^3)$
**在线**：$O(F(DK)^2)$ 每帧

---

## 6. 改进方法

### 6.1 多通道WPE

利用空间信息提升性能。

### 6.2 神经网络增强

使用DNN估计更准确的方差。

---

## 参考文献

1. Nakatani, T., et al. (2010). "Speech dereverberation based on variance-normalized delayed linear prediction." IEEE TASLP.

2. Yoshioka, T., & Nakatani, T. (2012). "Generalization of multi-channel linear prediction methods." IEEE TASLP.
