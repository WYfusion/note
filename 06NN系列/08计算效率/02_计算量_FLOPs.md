# 计算量 (FLOPs)

**FLOPs** (Floating Point Operations) 指的是浮点运算次数，通常用来衡量算法或模型的**时间复杂度**。在深度学习中，它直接关系到模型的推理速度和训练时间。

> **注意**：有时也会看到 **MACs** (Multiply-Accumulate Operations，乘加运算)。通常 $1 \text{ MAC} \approx 2 \text{ FLOPs}$（一次乘法+一次加法）。本文主要讨论 FLOPs。

## 1. 常规卷积层 (Standard Convolution)

卷积操作的计算量主要由输出特征图的大小和卷积核的运算密度决定。每一个输出像素点，都需要进行一次完整的卷积核点积运算。

### 计算公式

$$ 
\text{FLOPs} = H_{out} \times W_{out} \times C_{out} \times (k_h \times k_w \times C_{in}) 
$$

### 符号推导与说明
- **输出特征图大小** ($H_{out} \times W_{out}$): 卷积操作需要在输出特征图的每一个像素位置上进行。
- **单次卷积核运算量** ($k_h \times k_w \times C_{in}$): 在输出的每一个位置，卷积核需要在所有输入通道 ($C_{in}$) 的 $k_h \times k_w$ 窗口内进行乘加运算。
- **输出通道数** ($C_{out}$): 一共有 $C_{out}$ 个卷积核在同时工作。

### 示例
- **输入**: $(28 \times 28 \times 192)$
- **卷积核**: $(5 \times 5)$, 输出通道 $32$
- **输出尺寸**: 假设 padding 使得输出尺寸保持 $28 \times 28$
- **计算量**:
  $$ 
  28 \times 28 \times 32 \times (5 \times 5 \times 192) \approx 1.2 \times 10^8 \text{ (1.2亿次)} 
  $$ 

---

## 2. 分组卷积 (Group Convolution)

分组卷积由于不同分组之间不存在数据交换，其计算量也随分组数 $g$ 线性减少。

### 计算公式

$$ 
\text{FLOPs} = H_{out} \times W_{out} \times C_{out} \times (k_h \times k_w \times \frac{C_{in}}{g}) 
$$
即：
$$ 
\text{FLOPs}_{group} = \frac{\text{FLOPs}_{std}}{g} 
$$

### 特性
- **稀疏连接**: 卷积核不再与所有输入通道连接，而是只与 $\frac{1}{g}$ 的通道连接，因此计算量大幅降低。
- **深度可分离卷积 (Depthwise Separable Conv)**:
    - **Depthwise (DW)**: $g = C_{in} = C_{out}$，计算量极低。
    - **Pointwise (PW)**: $1 \times 1$ 卷积，用于融合通道信息，占据了主要计算量。

---

## 3. 全连接层 (Linear Layer)

全连接层的计算量相当于一次大型矩阵乘法。

### 计算公式
$$ 
\text{FLOPs} = I \times O 
$$ 
其中 $I$ 为输入节点数，$O$ 为输出节点数。每个输出节点都需要与所有输入节点进行加权求和。
