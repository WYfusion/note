# SwiGLU 激活函数

![[Pasted image 20251228111649.png|800]]

## 1. 简介
**SwiGLU (Swish-Gated Linear Unit)** 是一种结合了 **Swish 激活函数**和 **GLU (门控线性单元)** 特性的激活函数。它在 PaLM、LLaMA、Qwen 等现代大型语言模型 (LLM) 中被广泛采用，用以替代传统的 ReLU 或 GELU。

## 2. 公式推导

### 2.1 前置概念
SwiGLU 是基于以下两个概念的结合与改进：

1.  **[[Swish]] 激活函数**：一种平滑、非单调的自门控激活函数。
2.  **[[GLU]] (门控线性单元)**：引入了门控机制来控制信息流的线性单元。

### 2.2 SwiGLU 定义
SwiGLU 是 GLU 的一种变体，它将 GLU 中的 Sigmoid 激活函数替换为 Swish 激活函数。
$$ \text{SwiGLU}(x, W, V, b, c, \beta) = \text{Swish}_{\beta}(xW + b) \otimes (xV + c) $$

在实际的大模型应用（如 Transformer 的 FFN 层）中，通常省略偏置项 $b$ 和 $c$，公式简化为：
$$ \text{SwiGLU}(x) = \text{Swish}_{\beta}(xW) \otimes (xV) $$

**在 Transformer FFN 层中的应用形式**：
标准的 FFN (Feed-Forward Network)：
$$ \text{FFN}_{\text{ReLU}}(x) = \text{ReLU}(xW_1)W_2 $$
使用 SwiGLU 的 FFN：
$$ \text{FFN}_{\text{SwiGLU}}(x) = (\text{Swish}(xW) \otimes (xV)) W_2 $$
这里输入 $x$ 被投影为两部分（$xW$ 和 $xV$），一部分经过 Swish 激活，另一部分保持线性，两者相乘后再经过输出投影 $W_2$。

## 3. 导函数公式
为了理解反向传播，我们需要计算 SwiGLU 的梯度。
令 $u = xW$ (门控路径输入), $v = xV$ (线性路径输入)。
输出 $y = \text{Swish}(u) \odot v$。

我们需要用到 Swish 的导数：
$$ \text{Swish}'(x) = \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x)) = \text{Swish}(x) + \sigma(x)(1 - \text{Swish}(x)) $$

SwiGLU 关于 $u$ 和 $v$ 的偏导数分别为：
1.  **关于 $v$ 的偏导**（线性部分）：
    $$ \frac{\partial y}{\partial v} = \text{Swish}(u) $$
    这表明线性部分的梯度直接由门控值决定。如果门控值为 0，则该部分的梯度被阻断。

2.  **关于 $u$ 的偏导**（门控部分）：
    $$ \frac{\partial y}{\partial u} = \text{Swish}'(u) \odot v $$
    $$ \frac{\partial y}{\partial u} = (\text{Swish}(u) + \sigma(u)(1 - \text{Swish}(u))) \odot v $$

## 4. 作用与意义
1.  **门控机制 (Gating Mechanism)**：
    SwiGLU 的核心在于“门控”。$xW$ 经过 Swish 激活后，其值作为一个“软门”，控制 $xV$ 中信息的保留程度。这允许网络更精细地选择通过哪些特征，类似于 LSTM 中的门控结构。

2.  **非线性增强**：
    相比于 ReLU 的简单截断，SwiGLU 引入了更复杂的非线性交互（Swish 本身非线性 + 两个投影的乘积），增加了模型的表达能力。

## 5. 优势
1.  **性能提升**：
    在 Google 的论文 *GLU Variants Improve Transformer* (Shazeer, 2020) 中，作者通过大量实验证明，在计算量相同的情况下，SwiGLU 在困惑度 (Perplexity) 和下游任务上的表现优于 ReLU、GELU 和标准 GLU。

2.  **平滑性 (Smoothness)**：
    Swish 函数是光滑的（处处可导），不像 ReLU 在 0 处不可导。这种平滑性有助于优化过程，使梯度下降更稳定，能训练出更深的网络。

3.  **缓解梯度消失**：
    线性路径 $xV$ 的存在使得梯度可以更容易地流回前面的层（类似于 ResNet 的残差连接思想，但这里是乘性的），有助于训练深层网络。

4.  **动态特征选择**：
    通过学习 $W$ 和 $V$，模型可以根据输入动态调整特征的通过率，这比静态的激活函数更具适应性。

## 6. 总结
SwiGLU 是目前大语言模型中的“标配”激活函数之一。它通过引入门控机制和 Swish 激活，在保持计算效率的同时，显著提升了模型的特征提取能力和训练稳定性。虽然参数量相比标准 FFN 有所增加（需要两个权重矩阵 $W$ 和 $V$），但通常通过减少隐藏层维度（例如从 $4d$ 减小到 $\frac{2}{3} 4d$）来保持总参数量或计算量不变，且依然能获得更好的效果。