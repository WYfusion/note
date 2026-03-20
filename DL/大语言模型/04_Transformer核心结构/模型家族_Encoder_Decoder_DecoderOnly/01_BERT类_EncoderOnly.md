# BERT 类：Encoder-Only 架构详解

Encoder-Only 架构是理解型任务（Understanding Tasks）的王者。

## 1. 架构核心
*   **双向注意力 (Bidirectional Attention)**: 序列中的每个 Token 都能“看见”其他所有 Token。
*   **无 Mask**: Self-Attention 矩阵中没有因果掩码（Causal Mask）。

## 2. 代表模型：BERT (Bidirectional Encoder Representations from Transformers)
*   **预训练任务**:
    1.  **MLM (Masked Language Modeling)**: 随机遮盖 15% 的词，预测它们。
    2.  **NSP (Next Sentence Prediction)**: 判断两个句子是否连续。
*   **应用**: 文本分类、情感分析、命名实体识别 (NER)。

## 3. 语音领域的 BERT：Wav2Vec 2.0 & HuBERT

语音信号处理中，Encoder-Only 架构占据了半壁江山，主要用于**语音表征学习 (Speech Representation Learning)**。

### 3.1 Wav2Vec 2.0
*   **输入**: 原始波形 (Raw Waveform)。
*   **结构**: CNN 特征提取器 + Transformer Encoder。
*   **任务**: 对比学习 (Contrastive Learning)。Mask 掉一段音频特征，要求模型从负样本中识别出正确的量化特征。
*   **地位**: 相当于语音界的 BERT，其输出特征可以用于 ASR、说话人识别等多种下游任务。

### 3.2 HuBERT
*   **改进**: 引入了离线聚类步骤，生成伪标签 (Pseudo-labels)。
*   **任务**: 预测 Mask 帧所属的聚类中心 ID。这更像 BERT 的 MLM 任务（分类问题）。

### 3.3 Data2Vec
*   **统一架构**: Meta 提出的统一模态框架。无论是文本、图像还是语音，都使用同样的 Masked Prediction 任务，通过预测 Teacher 模型的输出来进行自蒸馏。
