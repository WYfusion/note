# 残差缩放与 DeepNorm 初始化技巧

为了训练极深（Deep）的 Transformer 模型（如 1000 层），仅仅使用 Pre-LN 是不够的。我们需要更精细的初始化和缩放策略。

## 1. 残差缩放 (Residual Scaling)

### 1.1 问题背景
在 Pre-LN 结构中：
$$ x_{L} = x_0 + \sum_{l=1}^L F_l(x_{l-1}) $$
输出是所有层输出的累加。随着层数 $L$ 增加，输出的方差会线性增长，导致数值不稳定。

### 1.2 解决方案：$1/\sqrt{2L}$ 缩放
GPT-2 引入了这种策略：在初始化时，将残差分支（FFN 和 Attention 的输出投影层）的权重乘以 $1/\sqrt{2L}$。
*   **原理**: 假设每层贡献的方差是 $\sigma^2$，缩放后变为 $\sigma^2 / 2L$。累加 $L$ 层后，总方差得到控制。
*   **效果**: 显著提升了深层 GPT 模型的训练稳定性。

## 2. DeepNorm

DeepNorm 是 Microsoft 提出的用于训练超深 Transformer 的技术，它试图结合 Post-LN 的高性能和 Pre-LN 的稳定性。

### 2.1 公式
$$ x_{l+1} = \text{LN}(\alpha x_l + G_l(x_l)) $$
其中 $\alpha$ 是一个随层数变化的常数（通常 $>1$），用于放大主干信号。

### 2.2 初始化要求
配合 DeepNorm，权重初始化需要满足特定的界限：
$$ \text{Xavier Normal} \times \beta $$
其中 $\beta$ 是一个小于 1 的缩放因子。

## 3. 语音模型中的初始化技巧

### 3.1 Zero Initialization (零初始化)
在 VALL-E 和一些语音生成模型中，常采用零初始化策略。
*   **做法**: 将 Transformer 最后一层（或残差分支的最后一层）的权重初始化为 0。
*   **意义**: 初始状态下，模型等价于恒等映射（Identity Mapping）。这对于流式语音生成特别有用，因为它允许模型从“复制输入”开始，逐渐学习复杂的变换。

### 3.2 T-Fixup (Transformer-Fixup)
一种无需 Layer Norm 也能训练深层 Transformer 的初始化方法。
*   **核心**: 极大地缩小初始化权重，根据层数 $L$ 进行缩放。
*   **应用**: 在某些对 Latency 极度敏感的端侧语音模型中，为了省去 LN 的计算开销，会尝试使用 T-Fixup。
