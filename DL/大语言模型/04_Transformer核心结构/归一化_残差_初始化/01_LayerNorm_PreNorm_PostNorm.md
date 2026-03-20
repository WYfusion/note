# Layer Normalization: Pre-Norm 与 Post-Norm

归一化（Normalization）是深度神经网络能够收敛的关键技术之一。在 Transformer 中，Layer Normalization (LN) 是标准配置。

## 1. Layer Normalization (LN) 基础

### 1.1 为什么不用 Batch Normalization (BN)?
*   **BN**: 对 Batch 维度做归一化。依赖 Batch Size，且对于变长序列（RNN/Transformer）处理麻烦（Padding 部分会影响统计量）。
*   **LN**: 对 Feature 维度做归一化。独立于 Batch Size，对每个样本单独计算。非常适合序列模型。

### 1.2 公式
对于一个输入向量 $\boldsymbol{x} \in \mathbb{R}^d$：
$$ \mu = \frac{1}{d} \sum_{i=1}^d x_i $$
$$ \sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2 $$
$$ \text{LN}(\boldsymbol{x}) = \gamma \cdot \frac{\boldsymbol{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta $$
其中 $\gamma$ 和 $\beta$ 是可学习的缩放和平移参数。

## 2. Post-Norm vs. Pre-Norm

Transformer 结构中有两种放置 LN 的位置，这对训练稳定性影响巨大。

### 2.1 Post-Norm (原始 Transformer)
先做 Attention/FFN，再加残差，最后做 LN。
$$ \boldsymbol{x}_{t+1} = \text{LN}(\boldsymbol{x}_t + \text{SubLayer}(\boldsymbol{x}_t)) $$

*   **优点**: 如果能训练好，通常性能略好（输出层方差受控）。
*   **缺点**: **极难训练**。梯度在反向传播时容易爆炸或消失，必须配合 Warmup 策略小心翼翼地调整学习率。
*   **代表模型**: BERT, 原始 Transformer。

### 2.2 Pre-Norm (现代主流)
先做 LN，再进 Attention/FFN，最后加残差。
$$ \boldsymbol{x}_{t+1} = \boldsymbol{x}_t + \text{SubLayer}(\text{LN}(\boldsymbol{x}_t)) $$

*   **优点**: **训练极其稳定**。梯度直接通过残差连接（Identity Path）传回底层，不需要 Warmup 也能收敛。
*   **缺点**: 理论上深层的表示能力可能略受限制（但大模型时代通过增加层数弥补了）。
*   **代表模型**: GPT-2, GPT-3, Llama, PaLM。

## 3. 语音数据中的归一化

### 3.1 输入特征归一化
在进入 Transformer 之前，音频特征（如 Mel-spectrogram）通常需要做 Instance Normalization 或 Cepstral Mean and Variance Normalization (CMVN)。
*   **目的**: 消除不同录音设备、不同说话人音量带来的分布差异。

### 3.2 语音模型中的 LN
*   **Wav2Vec 2.0**: 使用了 Group Normalization 或 Layer Normalization，取决于具体配置。
*   **Whisper**: 使用 Pre-Norm 结构的 Layer Normalization。由于音频信号动态范围大，Pre-Norm 的稳定性尤为重要。
