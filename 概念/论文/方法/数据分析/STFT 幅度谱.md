# STFT 幅度谱（Short-Time Fourier Transform Magnitude Spectrum）

- **定义**：对音频信号分帧（加窗）后进行傅里叶变换，取幅度值形成的频谱，忽略相位信息。
- **数学公式**： 设信号为 $x(t)$，窗函数为 $w(t)$，第 n 帧的 STFT 为：$X(n, k) = \sum_{t} x(t) \cdot w(t - nT) \cdot e^{-j2\pi kt/F_s}$ 幅度谱为 $|X(n, k)|$，其中 $F_s$ ，$k$ 为频率索引。
- **特点**：
    - 横轴：时间（帧），纵轴：线性频率（Hz）。
    - 仅保留幅度信息，丢失相位（但相位对波形重构至关重要）。
    - 是语谱图和梅尔频谱图的基础变换。