# Causal LM, Masked LM 与 Prefix LM

预训练的核心在于设计合理的**自监督学习目标 (Self-Supervised Learning Objectives)**。

## 1. Causal Language Modeling (CLM)

也就是标准的**自回归 (Auto-Regressive)** 建模。

### 1.1 定义
给定序列 $x = (x_1, ..., x_T)$，最大化似然概率：
$$ \mathcal{L}_{CLM} = - \sum_{t=1}^T \log P(x_t | x_{<t}; \theta) $$
*   **特点**: 单向注意力，只能看过去。
*   **代表**: GPT 系列, Llama。

### 1.2 语音中的 CLM
*   **VALL-E**: 将音频量化为离散 Token 序列，然后像 GPT 一样预测下一个 Audio Token。
*   **AudioLM**: 先预测语义 Token (Semantic Tokens)，再预测声学 Token (Acoustic Tokens)，均为 CLM 任务。

## 2. Masked Language Modeling (MLM)

也就是**自编码 (Auto-Encoding)** 建模。

### 2.1 定义
随机 Mask 掉序列中的一部分 $M$，利用上下文 $\tilde{x}$ 预测被 Mask 的部分 $x_M$。
$$ \mathcal{L}_{MLM} = - \sum_{i \in M} \log P(x_i | \tilde{x}; \theta) $$
*   **特点**: 双向注意力，能看到全局。
*   **代表**: BERT, RoBERTa。

### 2.2 语音中的 MLM
语音领域最成功的预训练范式。
*   **Wav2Vec 2.0**:
    *   **Masking**: 在 CNN 特征图上随机 Mask 时间步。
    *   **Contrastive Loss**: 要求模型预测被 Mask 位置的量化向量（正样本），并与同一句话中其他位置的向量（负样本）区分开。
*   **HuBERT**:
    *   **Masking**: 同上。
    *   **Classification Loss**: 预测被 Mask 帧所属的 K-Means 聚类中心 ID（类似于预测词 ID）。
*   **WavLM**:
    *   **Masking + Denoising**: 不仅 Mask，还在输入叠加噪声，要求模型还原干净的 Mask 内容。

## 3. Prefix LM (Prefix Language Modeling)

结合了 Encoder 和 Decoder 的特性。

### 3.1 定义
输入序列分为两部分：Prefix (双向可见) 和 Target (单向可见)。
*   **代表**: GLM (General Language Model), T5 (Encoder-Decoder 也可以看作广义的 Prefix LM)。

### 3.2 语音中的应用
*   **Whisper**:
    *   Encoder (Prefix): 看到完整的 Log-Mel 频谱图（双向）。
    *   Decoder (Target): 自回归生成文本 Token（单向）。
    *   这本质上是一个条件生成任务 $P(\text{Text} | \text{Audio})$。
