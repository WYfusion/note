# Scaling Laws, Chinchilla 与计算预算

Scaling Laws（缩放定律）指导我们如何分配计算资源（Compute）、数据量（Data）和模型参数量（Parameters）以获得最佳性能。

## 1. KM Scaling Law (Kaplan et al., 2020)

OpenAI 提出的幂律关系：
$$ L(N, D) \propto N^{-\alpha_N} + D^{-\alpha_D} $$
*   $L$: Loss (Test Loss)。
*   $N$: 模型参数量。
*   $D$: 训练数据量 (Tokens)。
*   **结论**: 模型越大、数据越多，Loss 越低。且 Loss 与 $N$ 和 $D$ 呈幂律下降关系。

## 2. Chinchilla Scaling Law (Hoffmann et al., 2022)

DeepMind 修正了 KM 定律，提出了**计算最优 (Compute-Optimal)** 的分配方案。

### 2.1 核心结论
在给定计算预算 $C$ 的情况下，最优的模型大小 $N_{opt}$ 和数据量 $D_{opt}$ 应该满足：
$$ D_{opt} \approx 20 \times N_{opt} $$
即：**每 1 个参数大约需要 20 个 Token 的训练数据。**

### 2.2 启示
*   Llama (65B, 1.4T Tokens) 比 GPT-3 (175B, 300B Tokens) 效果好，因为 GPT-3 严重欠训练 (Undertrained)。
*   现在的趋势是：模型不用太大，但数据要给足。

## 3. 计算预算估算

训练一个 Transformer 模型的计算量（FLOPs）估算公式：
$$ C \approx 6 \times N \times D $$
*   $6$: 前向传播 2 FLOPs/param，反向传播 4 FLOPs/param。
*   **例子**: 训练一个 7B 模型，用 1T Tokens。
    $$ C \approx 6 \times 7 \times 10^9 \times 1 \times 10^{12} = 4.2 \times 10^{22} \text{ FLOPs} $$
    如果使用 A100 GPU (312 TFLOPS BF16)，假设利用率 50%：
    $$ \text{Time} = \frac{4.2 \times 10^{22}}{150 \times 10^{12} \times 3600 \times 24} \approx 3.2 \text{ GPU-Days} $$

## 4. 语音大模型的 Scaling Laws

语音领域的 Scaling 行为比纯文本复杂，因为涉及声学特征和语义特征的混合。

### 4.1 Whisper 的发现
OpenAI 在 Whisper 论文中指出：
*   **数据量至关重要**: 从 680 小时增加到 680,000 小时，WER 呈对数线性下降。
*   **模型规模**: 在数据充足时，增加模型参数 (Tiny -> Large) 能显著降低 WER。
*   **瓶颈**: 当数据量较少时，增大模型收益递减（过拟合）。

### 4.2 语音数据的 "Token" 换算
如何将 Chinchilla 定律应用到语音？
*   **理解任务**: 1 秒音频 $\approx$ 50 帧。如果下采样 4 倍，则 1 秒 $\approx$ 12.5 帧。
    *   1 小时音频 $\approx$ 45,000 帧 (相当于 45k Tokens)。
    *   Whisper 680k 小时 $\approx$ 300 亿 (30B) "Frames"。
*   **生成任务**: 1 秒音频 $\approx$ 75 个离散 Token (EnCodec 24kbps)。
    *   1 小时 $\approx$ 270,000 Tokens。
    *   语音 Token 的信息密度远低于文本 Token，因此通常需要**更多**的 Token 数量才能达到同等的“智能”水平。

### 4.3 混合模态 Scaling
对于 Speech-LLaMA 这类模型：
*   **LLM 基座**: 遵循文本 Scaling Law。
*   **Speech Encoder**: 遵循语音 Scaling Law。
*   **对齐阶段**: 数据质量比数量更重要。少量的精标指令数据（Instruction Data）即可激发跨模态能力。
