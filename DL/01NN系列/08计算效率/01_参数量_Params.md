# 模型参数量 (Parameters)

参数量决定了模型的大小（硬盘占用）以及在推理和训练时对内存（显存）的需求。它是衡量模型空间复杂度的核心指标。

## 1. 常规卷积层 (Standard Convolution)

对于一个标准的卷积层，参数量的计算主要取决于卷积核的大小、输入通道数和输出通道数。

### 计算公式

$$ 
\text{Params} = K \times C_{in} \times C_{out} + C_{out} \quad (\text{if bias=True}) 
$$
或者提取公因式：
$$ 
\text{Params} = (K \times C_{in} + 1) \times C_{out} 
$$

### 符号说明
- $K$: 卷积核尺寸，通常为 $k_h \times k_w$（例如 $3 \times 3 = 9$）。
- $C_{in}$: 输入特征矩阵的通道数 (Input Channels)。
- $C_{out}$: 输出特征矩阵的通道数 (Output Channels)，也即卷积核的个数。
- $+1$: 代表偏置项 (Bias)。每个输出通道通常对应一个偏置参数。如果设置 `bias=False`，则去除此项。

### 示例
假设输入特征图为 $(28 \times 28 \times 192)$，使用尺寸为 $(5 \times 5)$ 的卷积核，输出通道数为 $32$。
- **权重参数**: $5 \times 5 \times 192 \times 32 = 153,600$
- **偏置参数**: $32$
- **总参数量**: $153,600 + 32 = 153,632$

---

## 2. 全连接层 (Fully Connected Layer / Dense)

全连接层的参数量取决于输入向量的长度和输出向量的长度。

### 计算公式
$$ 
\text{Params} = I \times O + O \quad (\text{if bias=True}) 
$$

### 符号说明
- $I$: 输入特征的维度 (Input Features)。
- $O$: 输出特征的维度 (Output Features)。
- $+O$: 偏置项，每个输出神经元有一个偏置。

---

## 3. 分组卷积 (Group Convolution)

分组卷积将输入通道分成 $g$ 组，每组独立进行卷积。这大大减少了参数量。

### 计算公式
$$ 
\text{Params} = K \times \frac{C_{in}}{g} \times C_{out} 
$$
或者写作：
$$ 
\text{Params} = \frac{1}{g} \times (K \times C_{in} \times C_{out}) 
$$

### 特性
- 当 $g=1$ 时，为常规卷积。
- 当 $g=C_{in}$ 且 $C_{out}=C_{in}$ 时，即为 **深度卷积 (Depthwise Convolution)**。此时参数量极小：$K \times 1 \times C_{out}$。
