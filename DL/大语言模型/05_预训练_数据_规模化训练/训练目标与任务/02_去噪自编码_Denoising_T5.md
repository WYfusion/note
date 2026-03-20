# 去噪自编码 (Denoising Autoencoder) 与 T5

去噪自编码器 (DAE) 的核心思想是：**破坏输入，然后重建它**。这迫使模型学习数据的鲁棒特征。

## 1. 文本去噪任务 (T5 / BART)

### 1.1 Span Corruption (T5)
*   **操作**: 随机选中一段文本（Span），将其替换为一个特殊的 Sentinel Token（如 `<extra_id_0>`）。
*   **输入**: `The cat is <extra_id_0> the mat.`
*   **目标**: `<extra_id_0> sitting on <extra_id_1>`
*   **优势**: 相比 BERT 的单字 Mask，Span Mask 迫使模型预测更长的片段，难度更大。

### 1.2 Token Deletion / Permutation (BART)
*   **Deletion**: 随机删除词。
*   **Permutation**: 打乱句子顺序。
*   **Infilling**: 填补缺失片段。

## 2. 语音去噪任务

语音信号天然包含噪声，因此去噪任务在语音预训练中尤为重要。

### 2.1 WavLM 的去噪策略
WavLM 在 HuBERT 的基础上引入了 **Denoising Modeling**。
*   **输入变换**:
    $$ x' = \text{Mask}(x + \text{Noise}) $$
    不仅对特征进行 Mask，还叠加了背景噪声（Background Noise）和重叠语音（Overlapping Speech）。
*   **目标**: 预测原始、干净、未被 Mask 的语音对应的聚类 ID。
*   **效果**: WavLM 在“鸡尾酒会问题”（多人说话、嘈杂环境）上的表现显著优于 Wav2Vec 2.0。

### 2.2 语音增强与分离 (SE & SS)
虽然通常作为下游任务，但也可以作为预训练目标。
*   **输入**: 混合音频 $y = x_1 + x_2$。
*   **目标**: 分离出 $x_1$ 和 $x_2$。
*   **PIT (Permutation Invariant Training)**: 解决输出顺序不确定的问题。

## 3. 跨模态去噪 (Speech-Text Denoising)

在 SpeechT5 等模型中，尝试统一模态。
*   **Speech -> Text**: 相当于 ASR。
*   **Text -> Speech**: 相当于 TTS。
*   **Corrupted Speech -> Speech**: 语音增强。
*   **Corrupted Text -> Text**: 文本纠错。
通过在一个模型中混合这些任务，实现全能的语音-文本处理能力。
