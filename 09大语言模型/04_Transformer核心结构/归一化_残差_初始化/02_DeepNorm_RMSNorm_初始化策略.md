# 进阶归一化：RMSNorm 与 DeepNorm

随着模型规模的扩大，标准的 Layer Normalization (LN) 逐渐暴露出计算开销大、深层模型不稳定等问题。

## 1. RMSNorm (Root Mean Square Layer Normalization)

RMSNorm 是目前最流行的大模型归一化方式（Llama, Gopher, Chinchilla 均采用）。

### 1.1 原理
作者认为 LN 的成功主要归功于**缩放**（Rescaling）不变性，而不是平移（Re-centering）不变性。因此，RMSNorm 去掉了减去均值（Mean）的操作，只保留除以均方根（RMS）。

### 1.2 公式
$$ \text{RMS}(\boldsymbol{x}) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2} $$
$$ \text{RMSNorm}(\boldsymbol{x}) = \gamma \cdot \frac{\boldsymbol{x}}{\text{RMS}(\boldsymbol{x}) + \epsilon} $$
注意：没有了 $\beta$（偏置项），也没有减去 $\mu$。

### 1.3 优势
*   **计算速度快**: 少了计算均值和平移的操作，在 GPU 上通常有 10%~40% 的加速。
*   **效果相当**: 在大模型实验中，性能与标准 LN 持平甚至略好。

## 2. DeepNorm

DeepNorm 是 Microsoft 提出的用于训练**极深 Transformer**（如 1000 层）的技术。

### 2.1 背景
Pre-Norm 虽然稳定，但会导致深层参数更新幅度过小。Post-Norm 虽然性能好，但梯度易爆炸。DeepNorm 试图结合二者优点。

### 2.2 核心做法
DeepNorm 本质上是一种**Post-Norm**，但配合了特殊的初始化策略和残差缩放。
$$ \boldsymbol{x}_{t+1} = \text{LN}(\alpha \cdot \boldsymbol{x}_t + \text{SubLayer}(\boldsymbol{x}_t)) $$
其中 $\alpha$ 是一个随层数增加的常数。

## 3. 初始化策略 (Initialization)

Transformer 的初始化对收敛至关重要。

### 3.1 Xavier / Glorot Initialization
$$ W \sim U\left[-\frac{\sqrt{6}}{\sqrt{d_{in}+d_{out}}}, \frac{\sqrt{6}}{\sqrt{d_{in}+d_{out}}}\right] $$
适用于 Tanh/Sigmoid 激活函数。

### 3.2 He Initialization (Kaiming Init)
$$ W \sim N\left(0, \frac{2}{d_{in}}\right) $$
适用于 ReLU/GELU 激活函数。

### 3.3 Transformer 特有的缩放
在 Pre-Norm 结构中，为了防止残差分支的方差随层数线性累积，通常会对残差分支的权重进行缩放：
$$ W_{output} \leftarrow W_{output} \cdot \frac{1}{\sqrt{2L}} $$
其中 $L$ 是总层数。GPT-2 就使用了这种策略。

## 4. 语音模型中的应用

### 4.1 梯度稳定性
语音信号的冗余度高，噪声大。在训练大型语音编码器（如 24 层 Conformer）时，梯度爆炸是常见问题。
*   **RMSNorm** 在语音大模型（如 Speech-LLaMA）中表现出良好的数值稳定性。
*   **Scale-Norm**: 一种更激进的归一化，在某些 ASR 任务中有应用。

### 4.2 零初始化 (Zero Init)
在一些语音生成模型（如 VALL-E）的残差连接中，有时会将输出层的权重初始化为 0，使得初始状态下模型近似恒等映射（Identity Mapping），有助于训练初期的稳定性。
