# 空间相关矩阵 (Spatial Covariance Matrix, SCM)

空间相关矩阵（在文献中也常称为**空间协方差矩阵**或**跨谱密度矩阵 Cross-spectral Density, CSD**）是多通道信号处理中的核心数学工具。它描述了麦克风阵列中不同麦克风接收到的信号之间的相关性，包含了关于声源数量、位置以及噪声场特性的丰富信息。

## 1. 定义

对于一个包含 $P$ 个麦克风的阵列，其接收信号在频域的向量为 $\mathbf{X}(f, t) = [X_1(f, t), \dots, X_P(f, t)]^T$。

**空间相关矩阵 $\mathbf{\Phi}_{xx}(f)$** 被定义为该信号向量的外积的期望（或时间平均）：

$$
\mathbf{\Phi}_{xx}(f) = E\{ \mathbf{X}(f, t) \mathbf{X}^H(f, t) \}
$$

其中 $E\{\cdot\}$ 表示期望运算，$^H$ 表示共轭转置。 $\mathbf{\Phi}_{xx}(f)$ 是一个 $P \times P$ 的**半正定 Hermitian 矩阵**，自共轭矩阵，共轭转置等于自身。

- **对角线元素**: $\mathbf{\Phi}_{xx}(f)_{pp} = E\{ |X_p(f, t)|^2 \}$ 是第 $p$ 个麦克风接收信号的**功率谱密度 (Power Spectral Density, PSD)**。
- **非对角线元素**: $\mathbf{\Phi}_{xx}(f)_{pq} = E\{ X_p(f, t) X_q^*(f, t) \}$ 是第 $p$ 个和第 $q$ 个麦克风接收信号的**互功率谱密度 (Cross-Power Spectral Density)**，它是一个复数，包含了两个通道间的幅度和相位关系。

## 2. 信号模型与SCM

假设环境中只有一个目标声源 $S(f,t)$ 和与它不相关的加性噪声 $\mathbf{V}(f,t)$，信号模型为 $\mathbf{X}(f, t) = \mathbf{d}(f) S(f, t) + \mathbf{V}(f, t)$。

此时，总的SCM可以分解为信号SCM和噪声SCM之和：

$$
\mathbf{\Phi}_{xx}(f) = \mathbf{\Phi}_{ss}(f) + \mathbf{\Phi}_{nn}(f)
$$

其中：
- **信号SCM**:
  $$ 
  \mathbf{\Phi}_{ss}(f) = E\{ (\mathbf{d}S)(\mathbf{d}S)^H \} = \mathbf{d}(f) E\{|S(f,t)|^2\} \mathbf{d}^H(f) = \phi_{ss}(f) \mathbf{d}(f) \mathbf{d}^H(f) 
  $$ 
  这里 $\phi_{ss}(f)$ 是纯净语音信号的功率谱。可见，对于单个声源，其SCM是一个**秩为1**的矩阵。这个特性在声源分离和DOA估计中非常重要。

- **噪声SCM**:
  $$ 
  \mathbf{\Phi}_{nn}(f) = E\{ \mathbf{V}(f, t) \mathbf{V}^H(f, t) \} 
  $$ 
  噪声SCM的结构取决于噪声场的空间特性。
    - **空间白噪声 (Spatially White Noise)**: 噪声来自四面八方且互不相关。此时 $\mathbf{\Phi}_{nn}(f) = \phi_{nn}(f) \mathbf{I}$，其中 $\mathbf{I}$ 是单位矩阵。
    - **扩散噪声场 (Diffuse Noise Field)**: 在一个理想的混响环境中，噪声SCM的 $(p, q)$ 元素可以由 sinc 函数 $\text{sinc}(2\pi f d_{pq}/c)$ 描述，其中 $d_{pq}$ 是麦克风 $p$ 和 $q$ 之间的距离，$c$ 是声速。

## 3. SCM 的估计

在实际应用中，我们无法得到真实的期望值，因此需要从有限的观测数据中估计SCM。常用的方法是使用**时间平滑**：

$$ 
\hat{\mathbf{\Phi}}_{xx}(f) = \frac{1}{N} \sum_{t=1}^{N} \mathbf{X}(f, t) \mathbf{X}^H(f, t) 
$$ 

为了获得更平滑的估计，通常采用**递归平滑（一阶IIR滤波器）**，前后时间点的叠加，将随机噪声平均掉了，同时语音分量由于较为恒定逐渐叠加变大，因此相当于突出了语音抑制了随机噪声：

$$ 
\hat{\mathbf{\Phi}}_{xx}(f, t) = \alpha \hat{\mathbf{\Phi}}_{xx}(f, t-1) + (1-\alpha) \mathbf{X}(f, t) \mathbf{X}^H(f, t) 
$$ 

其中 $\alpha$ 是平滑因子，取值范围在 (0, 1) 之间。

## 4. SCM 的重要作用

SCM是自适应波束形成算法的基石，例如：

- **MVDR 波束形成器**: 其权重计算公式为 $\mathbf{w}_{mvdr} \propto \mathbf{\Phi}_{nn}^{-1} \mathbf{d}$，直接依赖于对噪声SCM的准确估计和求逆。
- **声源定位 (DOA Estimation)**: 像 MUSIC (Multiple Signal Classification) 这样的高分辨率DOA估计算法，通过对SCM进行特征值分解，将其划分为信号子空间和噪声子空间，从而估计声源方向。
- **盲源分离 (BSS)**: 像独立向量分析 (IVA) 等算法，通过迭代更新分离矩阵来使得分离后信号的SCM对角化或满足某种独立性假设。
- **噪声估计**: 在语音活动检测 (VAD) 的辅助下，我们可以在没有语音的时刻更新噪声SCM $\mathbf{\Phi}_{nn}$，然后在有语音的时刻使用它进行降噪。

**总结**: SCM将多通道信号从简单的时域波形转化为一个包含丰富空间结构信息的矩阵。如何从含噪数据中鲁棒地估计信号和噪声的SCM，并有效利用其结构特性，是整个多通道语音处理领域研究的核心问题之一。
