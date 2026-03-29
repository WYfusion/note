---
tags:
  - 机器学习
  - 深度学习
  - Transformer
  - Encoder
  - 编码器块
created: 2025-01-18
modified: 2025-01-18
difficulty: 中等
related:
  - [[Transformer/00_阅读指引]]
  - [[Transformer/03_注意力机制/01_自注意力基础]]
  - [[Transformer/04_模块与整体结构/02_Decoder_Block]]
---

# Encoder Block（编码器块）

## 结构（Pre-LN 常用）

### 1. 自注意力与残差连接

$$
H_1 = X + \text{MHA}(\text{LN}(X))
$$

其中：
- $\text{LN}(X)$ 是层归一化（Layer Normalization）
- $\text{MHA}(\cdot)$ 是多头自注意力
- 残差连接保持梯度流

### 2. 前馈网络与残差连接

$$
H_2 = H_1 + \text{FFN}(\text{LN}(H_1))
$$

其中：
- $\text{FFN}(\cdot)$ 是前馈神经网络
- 残差连接进一步稳定训练

### 前馈网络（FFN）

$$
\text{FFN}(x) = \sigma(xW_1 + b_1) W_2 + b_2
$$

#### 特点

- 常用 **GELU** 激活函数
- 中间维度 $d_{ff} \approx 4d_{model}$
- 提供非线性变换能力

### 架构特点

- **残差保持梯度流**：通过跳跃连接避免梯度消失
- **LayerNorm 稳定训练**：归一化层间分布
- **Post-LN**（Norm 放输出端）在浅层可行，但深层易不稳

---

## 功能

### 1. 全局上下文编码

将全局上下文编码为一组上下文相关表示，供下游任务或 Decoder 使用。

### 2. 维度保持

对称堆叠 $N$ 层，输出形状保持 $(L, d_{model})$。

### 3. 层次化表示

- 浅层：学习局部模式和低级特征
- 深层：学习全局模式和高级语义

---

## 实践提示

### Dropout

- 注意力权重：在 softmax 后应用
- FFN 输出：在激活函数后应用
- 残差旁路：可选择性插入 dropout

### 正则化

- **Attention Dropout**：防止过拟合
- **LayerDrop**：随机丢弃整个层，提高鲁棒性
- **RMSNorm**：深层时可用 RMSNorm 代替 LayerNorm

### 其他技巧

- **Warmup**：学习率预热，稳定训练初期
- **Label Smoothing**：标签平滑，防止过自信
- **Weight Decay**：权重衰减，正则化

---

## 相关链接

- [[00_transformer阅读指引]] - Transformer 学习指引
- [[Transformer/03_注意力机制/01_自注意力基础]] - 自注意力基础
- [[Transformer/04_模块与整体结构/02_Decoder_Block]] - Decoder Block

## 参考资料

- Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*
- Ba, J. L., et al. (2016). Layer Normalization. *arXiv*

