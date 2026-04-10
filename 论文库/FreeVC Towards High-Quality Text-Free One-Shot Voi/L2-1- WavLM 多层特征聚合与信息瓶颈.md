## 前置知识

> [!important]
> 
> 阅读本页前建议先读：[[FreeVC- Towards High-Quality Text-Free One-Shot Voice Conversion]]

---

## 0. 定位

> [!important]
> 
> 本页聚焦 FreeVC 如何利用 WavLM 多层特征的加权聚合 + 线性瓶颈层实现内容-音色分离，以及这种简单方法的优劣势分析。

---

## 1. WavLM 多层特征聚合

WavLM 是一个多层 Transformer，不同层编码不同层次的信息：

- **浅层**（Layer 1-6）：声学/音素级特征，含较多说话人信息

- **中层**（Layer 7-12）：平衡语义与声学

- **深层**（Layer 13-24）：语义/语言级特征，说话人信息较少

FreeVC 的**加权层聚合**：

$$h = \sum_{l=1}^{L} w_l \cdot h_l, \quad \text{where} \quad \sum w_l = 1, \; w_l \geq 0$$

权重 $w_l$ 是可学习参数，模型自动找到内容保持与音色去除的最优层组合。

---

## 2. 线性瓶颈层

聚合后的特征通过一个线性投影压缩维度：

$$z = W_{\text{bottleneck}} \cdot h + b, \quad W \in \mathbb{R}^{d' \times d}, \; d' < d$$

维度压缩迫使模型丢弃「不重要」的信息维度。理想情况下，音色信息的方差大但对内容无用，会被优先压缩掉。

> [!important]
> 
> **误区纠正：「线性瓶颈能有效去除音色」**
> 
> 不完全准确。线性投影在高维连续空间中保留了大量子空间方向信息，音色特征可能分布在与内容高度相关的方向上，线性压缩无法有效分离。这就是为什么后续工作纷纷转向更强的瓶颈：
> 
> - **Vevo**：VQ-VAE 量化（硬截断，信息丢失更彻底）
> 
> - **R-VC**：数据扰动 + K-means（信号级破坏 + 离散化）
> 
> - **Seed-VC**：Timbre Shifter（从源头改变音色）

---

## 3. 瓶颈方法对比

|方法|论文|去音色强度|内容损失|复杂度|
|---|---|---|---|---|
|**VQ 量化**|Vevo|中-强（$K$ 可控）|中（WER↑）|需训练 VQ-VAE|
|**外部 Timbre Shifter**|Seed-VC|强|低|需外部模型|

---

## 延伸阅读

> [!important]
> 
> - 下一页推荐：L2-2 VITS 框架改造详解
> 
> - 对比阅读：[[L2-1- 渐进自监督解耦方法（VQ-VAE 信息瓶颈详解）]]

## 参考文献

- [Li et al., 2022] FreeVC 原论文 §3 Method

- [Chen et al., 2022] "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing"