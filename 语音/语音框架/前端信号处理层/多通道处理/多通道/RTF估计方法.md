# 相对传递函数(RTF)的估计方法

相对传递函数 (RTF, Relative Transfer Function) 描述了声源到阵列中某个麦克风与到参考麦克风之间的传递函数之比。在 MVDR 等波束形成算法中，它作为导向矢量的一部分，是实现无失真约束的关键。 [[相对传输函数RTF]] 对此有很好的介绍。

这里，我们将专注于**如何估计RTF**。RTF的准确估计是高性能波束形成的前提。

假设参考麦克风是第1个，那么第 $p$ 个通道的RTF定义为 $d_p(f) = H_p(f) / H_1(f)$。目标就是估计这个复数向量 $\mathbf{d}(f) = [1, d_2(f), \dots, d_P(f)]^T$。

## 1. 基于协方差矩阵的方法

当有语音存在时，信号的协方差矩阵 $\mathbf{\Phi}_{ss}(f)$ 是一个秩为1的矩阵，可以表示为 $\phi_{ss}(f) \mathbf{d}(f) \mathbf{d}^H(f)$。这个特性是许多估计方法的基础。

### 基于广义特征值分解 (GEVD) 的方法

如果我们能够得到语音段的协方差矩阵 $\mathbf{\Phi}_{xx}(f)$（语音+噪声）和纯噪声段的协方差矩阵 $\mathbf{\Phi}_{nn}(f)$，可以通过求解广义特征值问题来估计RTF：

$$
\mathbf{\Phi}_{xx}(f) \mathbf{v} = \lambda \mathbf{\Phi}_{nn}(f) \mathbf{v}
$$ 

**理论依据**:
由于 $\mathbf{\Phi}_{xx} = \mathbf{\Phi}_{ss} + \mathbf{\Phi}_{nn} = \phi_{ss}\mathbf{d}\mathbf{d}^H + \mathbf{\Phi}_{nn}$，代入上式可得：
$(\phi_{ss}\mathbf{d}\mathbf{d}^H + \mathbf{\Phi}_{nn}) \mathbf{v} = \lambda \mathbf{\Phi}_{nn} \mathbf{v}$
$\phi_{ss}\mathbf{d}(\mathbf{d}^H \mathbf{v}) = (\lambda - 1) \mathbf{\Phi}_{nn} \mathbf{v}$

当 $\mathbf{v}$ 与 $\mathbf{d}$ 方向一致时，可以得到最大特征值 $\lambda_{max}$。因此，对应于**最大广义特征值**的特征向量 $\mathbf{v}_{max}$ 就是RTF向量 $\mathbf{d}(f)$ 的一个估计（相差一个复数标量）。

**估计步骤**:
1.  在语音间隙估计噪声协方差矩阵 $\hat{\mathbf{\Phi}}_{nn}(f)$。
2.  在语音活动区域估计混合信号协方差矩阵 $\hat{\mathbf{\Phi}}_{xx}(f)$。
3.  求解广义特征值问题 $\hat{\mathbf{\Phi}}_{xx}(f) \mathbf{v} = \lambda \hat{\mathbf{\Phi}}_{nn}(f) \mathbf{v}$。
4.  取最大特征值对应的特征向量 $\mathbf{v}_{max}$。
5.  对 $\mathbf{v}_{max}$进行归一化，使其第一个元素为1，得到RTF估计值 $\hat{\mathbf{d}}(f)$。

**优点**: 理论完备，是许多传统高性能系统的核心。
**缺点**: 需要准确的语音活动检测 (VAD) 来区分语音段和噪声段，且对 $\mathbf{\Phi}_{nn}$ 的求逆操作敏感，在低信噪比和强混响下性能下降。

### 基于主成分分析 (PCA) / 特征值分解 (EVD) 的方法

在信噪比较高的情况下，可以近似认为 $\mathbf{\Phi}_{xx}(f) \approx \mathbf{\Phi}_{ss}(f) = \phi_{ss}(f) \mathbf{d}(f) \mathbf{d}^H(f)$。此时，$\mathbf{\Phi}_{xx}(f)$ 的主特征向量（对应最大特征值的特征向量）就是RTF的估计。

这种方法更简单，但只适用于高SNR场景。

## 2. 基于相关性的方法

### 互相关法 (Cross-correlation)

RTF的相位信息主要由信号到达不同麦克风的时间差 (TDoA (Time Difference of Arrival)) 决定。我们可以通过计算参考通道与其他通道信号的**广义互相关 (Generalized Cross-Correlation, GCC)** 来估计TDOA。

**GCC-PHAT (相位变换)** 是最常用的方法：
1.  计算两通道信号 $x_1(t)$ 和 $x_p(t)$ 的互功率谱：$C_{1p}(f) = X_1(f, t) X_p^*(f, t)$。
2.  进行相位变换，只保留相位信息：$\psi_{1p}(f) = \frac{C_{1p}(f)}{|C_{1p}(f)|}$。
3.  通过逆傅里叶变换得到互相关函数，其峰值位置就对应于TDOA $\tau_{1p}$。
4.  利用估计出的TDOA，可以构建一个只包含相位信息的简化版RTF（适用于自由场假设）：
    $$
    \hat{d}_p(f) = e^{-j 2\pi f \tau_{1p}}
    $$ 

**优点**: 计算简单，对幅度变化不敏感。
**缺点**: 只估计了相位部分，忽略了幅度差异；**在强混响环境下，互相关函数的峰值可能不准**。

## 3. 基于深度学习的方法

近年来，深度神经网络被用于隐式或显式地估计RTF（或直接估计波束形成器权重）。

- **隐式估计**: 如 `EaBNet`，网络直接从多通道输入中学习一个嵌入(embedding)，然后映射到波束形成权重。这个过程**隐式地**包含了对RTF信息的编码和利用，但并不显式地输出RTF。这种端到端的方式通常更鲁棒。
- **显式估计**: 也可以设计一个神经网络，其训练目标就是输出精确的RTF。例如，输入多通道频谱，输出RTF的实部和虚部。这个估计出的RTF随后可以被送入一个传统的MVDR波束形成器中。

**总结**: RTF估计是多通道语音增强中的一个经典而持续研究的课题。传统方法（如GEVD）理论坚实，但依赖于对信号统计特性的准确估计。现代深度学习方法则倾向于通过端到端学习来绕过显式估计的困难，从而在复杂场景下获得更好的鲁棒性。
