# Encoder-Only 架构：从 BERT 到 Wav2Vec 2.0

Encoder-Only 架构仅使用 Transformer 的 Encoder 部分。其核心特征是**双向注意力（Bidirectional Attention）**，即每个 Token 都能看到整个序列的信息。

## 1. 文本领域的代表：BERT

### 1.1 结构特点
*   **双向性**: Self-Attention 没有任何 Mask，位置 $i$ 可以关注到位置 $j$（无论 $j>i$ 还是 $j<i$）。
*   **输入**: Token Embeddings + Segment Embeddings + Position Embeddings。

### 1.2 预训练目标
*   **MLM (Masked Language Modeling)**: 随机 Mask 掉 15% 的 Token，让模型根据上下文预测被 Mask 的词。
    *   输入: `The cat is [MASK] on the mat.`
    *   预测: `sleeping`

### 1.3 适用场景
*   理解任务（Understanding）：文本分类、命名实体识别（NER）、问答（SQuAD）。
*   **不适合生成**：因为无法像 GPT 那样逐词生成。

## 2. 语音领域的代表：Wav2Vec 2.0 & HuBERT

在语音领域，Encoder-Only 架构是**自监督学习**（Self-Supervised Learning, SSL）的主流选择。

### 2.1 核心挑战
语音是连续信号，没有天然的“词”或“Token”。无法直接像 BERT 那样做 Softmax 分类预测。

### 2.2 Wav2Vec 2.0
*   **结构**: CNN 特征提取器 + Transformer Encoder。
*   **量化模块 (Quantization)**: 将连续的 CNN 特征离散化为有限的码本（Codebook）向量。
*   **对比学习 (Contrastive Loss)**:
    *   Mask 掉一部分 Transformer 的输出。
    *   要求模型预测被 Mask 位置的量化向量（正样本），同时与 distractors（负样本）区分开。

### 2.3 HuBERT (Hidden Unit BERT)
*   **离线聚类**: 先用 K-Means 对 MFCC 特征聚类，得到伪标签（Pseudo-labels）。
*   **预测目标**: 类似于 BERT，Mask 掉输入，让模型预测被 Mask 帧对应的聚类 ID。
*   **迭代优化**: 用第一轮模型的中间层特征重新聚类，训练第二轮模型。

### 2.4 为什么语音偏爱 Encoder-Only？
语音识别（ASR）本质上是一个**序列标注**或**序列转录**问题。Encoder 能够提取极其丰富的声学特征（音素、韵律、说话人信息），这些特征是下游任务（ASR, 说话人识别, 情感分析）的基础。
