# 位置编码：绝对、相对、RoPE 与 ALiBi

Transformer 的 Self-Attention 是**置换不变**的。为了让模型理解“顺序”，必须注入位置信息。

## 1. 绝对位置编码 (Absolute PE)

### 1.1 Sinusoidal (正弦/余弦)
原始 Transformer 使用。
$$ PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d}) $$
*   **优点**: 无需训练，理论上支持无限长度。
*   **缺点**: 外推性（Extrapolation）差，训练时没见过的长度效果不好。

### 1.2 Learnable (可学习)
BERT, GPT-2/3 使用。
直接学习一个 $L_{max} \times d$ 的矩阵。
*   **缺点**: 长度被限制在 $L_{max}$ (如 512, 2048)。超过这个长度无法处理。

## 2. 相对位置编码 (Relative PE)

核心思想：Token A 关注 Token B，只取决于它们之间的**距离** $i-j$，而不是绝对位置。

### 2.1 T5 Bias
在 Attention Score 计算时加入一个偏置项 $b_{i-j}$。
$$ \text{Attention}(Q, K) = \text{softmax}(QK^T + B) $$

## 3. RoPE (Rotary Positional Embeddings) —— 旋转位置编码

目前 LLM (Llama, Qwen, PaLM) 的**事实标准**。它巧妙地结合了绝对位置编码的实现便利性和相对位置编码的数学性质。

### 3.1 核心思想
通过将词向量在复平面上**旋转**一个角度来注入位置信息。旋转角度只与位置 $m$ 有关。

$$ f(x, m) = x e^{im\theta} $$

当计算 Attention 时：
$$ \langle f(q, m), f(k, n) \rangle = (q e^{im\theta}) (k e^{in\theta})^* = qk e^{i(m-n)\theta} $$
结果只包含 $(m-n)$，即**相对距离**！

### 3.2 优点
1.  **相对位置感知**: 天然捕捉相对距离。
2.  **外推性强**: 通过线性内插 (Linear Interpolation) 或 NTK-Aware Scaled，可以轻松扩展上下文长度（如 Llama 2 从 4k 扩展到 32k）。
3.  **实现简单**: 只是对 Q, K 向量做逐元素的旋转操作，无需额外的参数矩阵。

## 4. ALiBi (Attention with Linear Biases)

Bloom, MPT 模型使用。
不修改 Embedding，直接在 Attention Score 矩阵上减去一个与距离成正比的惩罚项。
$$ \text{Score} = QK^T - m \cdot |i-j| $$
*   **优点**: 外推性极强，训练短序列，推理长序列效果最好。
*   **缺点**: 表达能力可能不如 RoPE 丰富。

## 5. 总结

| 方法 | 类型 | 代表模型 | 外推能力 |
| :--- | :--- | :--- | :--- |
| **Sinusoidal** | 绝对 | Transformer (原版) | 弱 |
| **Learnable** | 绝对 | BERT, GPT-3 | 无 |
| **RoPE** | 混合/相对 | **Llama, Qwen, PaLM** | **强** (需技巧) |
| **ALiBi** | 相对 | Bloom | 极强 |
