- **定义**：在 STFT 幅度谱基础上，通过**梅尔滤波器组**（$\text{Mel Filter Bank}$）对频率轴进行非线性压缩，使频率轴符合人类听觉感知特性。
- **数学公式**：
    1. 计算 STFT 幅度谱 $|X(n, k)|$；
    2. 用 M 个三角滤波器（梅尔刻度）对幅度谱加权求和，得到梅尔能量：$S(n, m) = \sum_{k=1}^{K} |X(n, k)|^2 \cdot H_m(k), \quad m=1,2,...,M$ 其中 $H_m(k)$ 是第 $m$ 个梅尔滤波器在频率 $k$ 处的响应。
    3. 通常对 $S(n, m)$ 取对数，得到梅尔频谱图。
- **梅尔刻度（Mel Scale）**： 频率 f 到梅尔值 $\text{mel}(f)$ 的转换公式：$\text{mel}(f) = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$ 特点：**低频分辨率高，高频分辨率低**，模拟人耳对频率的非线性感知（如人耳难以区分 $8000Hz$ 和 $8100Hz$，但易区分 $100Hz$ 和 $200Hz$）。