# 线性注意力：Linear Attention 与 Performer

虽然 FlashAttention 优化了 IO，但其理论复杂度仍然是 $O(L^2)$。为了彻底解决长序列问题，研究者提出了复杂度为 $O(L)$ 的线性注意力机制。

## 1. 核心思想：结合律

标准 Attention：
$$ \text{Attention}(Q, K, V) = \text{softmax}(QK^T)V $$
这里必须先算 $QK^T$（大小 $L \times L$），再乘以 $V$。

如果我们去掉 Softmax（或者用某种核函数 $\phi$ 近似），就可以利用矩阵乘法的结合律：
$$ (QK^T)V \approx Q(K^TV) $$

*   $K^T$ 是 $d \times L$，$V$ 是 $L \times d$。
*   $K^TV$ 的结果是 $d \times d$。
*   $Q(K^TV)$ 的计算复杂度是 $O(L \cdot d^2)$。
*   当 $L \gg d$ 时，复杂度从 $O(L^2)$ 降到了 $O(L)$。

## 2. 常见变体

### 2.1 Kernel-based (Performer)
使用随机特征映射（Random Feature Map）来近似 Softmax 核。
$$ \text{softmax}(q^T k) \approx \phi(q)^T \phi(k) $$
Performer 证明了通过正交随机特征，可以无偏地逼近标准 Attention。

### 2.2 Linear Transformer (Katharopoulos et al.)
使用简单的特征映射，如 $\phi(x) = \text{elu}(x) + 1$。[[ELU（Exponential Linear Unit）|elu激活函数]]
这种方法可以写成 RNN 的形式，非常适合流式推理。

## 3. 优缺点分析

### 3.1 优点
*   **线性复杂度**: 显存和计算量都与序列长度成线性关系。
*   **无限长度**: 理论上可以处理任意长度的序列。

### 3.2 缺点
*   **性能损失**: 近似 Softmax 必然带来精度下降。在很多任务上，Linear Attention 的效果不如标准 Attention。
*   **训练困难**: 对梯度的数值稳定性要求更高。

## 4. 语音领域的应用

虽然在大语言模型（LLM）中 FlashAttention 占据了统治地位，但在语音领域，Linear Attention 仍有一席之地。
*   **流式 ASR**: Linear Transformer 的 RNN 形式天然适合流式处理，不需要像标准 Transformer 那样缓存巨大的 KV Cache。
*   **极长音频**: 对于几小时长的会议录音，即使是 FlashAttention 也可能吃不消，Linear Attention 提供了一种可行的低资源替代方案。
