# 相对位置编码：RoPE 与 ALiBi

随着序列长度的增加（尤其是长文档和长音频），绝对位置编码的外推性（Extrapolation）成为瓶颈。相对位置编码的核心思想是：**Token 之间的注意力强度只应取决于它们之间的距离 $i-j$，而不是绝对位置 $i$ 和 $j$。**

## 1. RoPE (Rotary Positional Embedding)

RoPE 是目前主流大模型（Llama, PaLM, Qwen, ChatGLM）的标配。它巧妙地通过**绝对位置编码的方式实现了相对位置编码的效果**。

### 1.1 核心思想
将向量视为复数，通过旋转角度来注入位置信息。
对于位置 $m$ 的向量 $\boldsymbol{x}_m$，将其乘以一个旋转矩阵 $R_{\Theta, m}$：
$$ f(\boldsymbol{x}, m) = \boldsymbol{x} e^{im\theta} $$

在实数域下的二维情况：
$$ \begin{pmatrix} x'_1 \\ x'_2 \end{pmatrix} = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} $$

### 1.2 相对位置性质
当我们计算 Query 和 Key 的内积（Attention Score）时：
$$ \langle f(\boldsymbol{q}, m), f(\boldsymbol{k}, n) \rangle = (\boldsymbol{q} e^{im\theta}) \cdot (\boldsymbol{k} e^{in\theta})^* = \boldsymbol{q}\boldsymbol{k}^T e^{i(m-n)\theta} $$
结果只包含 $(m-n)$，即相对距离。绝对位置 $m$ 和 $n$ 被消掉了。

### 1.3 优势
*   **外推性强**: 理论上可以处理比训练长度更长的序列（配合 NTK-Aware Scaling 等技术）。
*   **无需额外参数**: 旋转矩阵是固定的，不增加模型参数。
*   **乘法实现**: 相比加法式编码，RoPE 与 Attention 机制结合得更紧密。

## 2. ALiBi (Attention with Linear Biases)

ALiBi (Attention with Linear Biases) 是一种更简单粗暴但有效的方案，常见于 Bloom, MPT 等模型。

### 2.1 原理
不修改 Embedding，而是直接修改 Attention Score 的计算公式。
在 Softmax 之前，给 Attention Score 加上一个与距离成比例的惩罚项：

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + m \cdot [-(i-j)]\right)V $$

其中 $m$ 是一个特定于 Head 的斜率参数（Slope）。距离越远，惩罚越大，Attention 权重越小。

### 2.2 优势
*   **极强的外推性**: 甚至可以在训练时只用短序列，推理时直接处理超长序列。
*   **计算高效**: 只是加一个 Bias 矩阵，开销极小。

## 3. 语音大模型中的长序列挑战

语音序列通常比文本长得多（1秒音频 $\approx$ 50个 Token/帧）。

### 3.1 RoPE 在语音中的应用
*   **长音频建模**: 语音 LLM（如 Qwen-Audio, Speech-LLaMA）通常直接继承基座模型的 RoPE。
*   **频率调整**: 由于音频 Token 密度大，有时需要调整 RoPE 的 Base Frequency（例如从 10000 调大到 1000000），以避免旋转周期过短导致的位置混淆。

### 3.2 相对位置的必要性
*   **局部依赖**: 语音识别中，当前音素主要取决于前后几百毫秒的上下文。相对位置编码天然适合这种局部性。
*   **流式处理**: 在流式 ASR 中，绝对位置一直在增加，使用绝对位置编码会导致数值溢出或分布偏移，而相对位置编码则不受影响。
