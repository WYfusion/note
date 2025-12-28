# 绝对位置编码：Sinusoidal 与 Learnable

Transformer 的 Self-Attention 机制本质上是**集合运算**，不具备序列顺序的感知能力。为了让模型理解“先后顺序”，必须显式注入位置信息。

## 1. Sinusoidal Positional Encoding (正弦位置编码)

这是 Google 在《Attention Is All You Need》原论文中提出的方案。

### 1.1 公式
对于位置 $pos$ 和维度索引 $2i$ 或 $2i+1$：
$$ PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right) $$
$$ PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right) $$

### 1.2 为什么是正弦/余弦？
*   **相对位置线性关系**: 作者希望 $PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性函数。
    *   由三角公式 $\sin(\alpha+\beta) = \sin\alpha\cos\beta + \cos\alpha\sin\beta$，确实满足线性变换性质。
*   **有界性**: 值域在 $[-1, 1]$ 之间，不会像直接输入整数 $pos$ 那样导致数值爆炸。

### 1.3 注入方式
直接**相加**到 Input Embedding 上：
$$ X = \text{Embedding}(Input) + PE $$

### 1.4 局限性
虽然理论上支持无限长度，但在实际训练中，模型很难泛化到比训练语料更长的序列（外推性差）。

## 2. Learnable Positional Embedding (可学习位置编码)

BERT, GPT-2, GPT-3 采用了这种简单粗暴的方法。

### 2.1 原理
直接初始化一个大小为 $[MaxLen, d_{model}]$ 的矩阵 $W_{pos}$，作为模型参数的一部分进行训练。
*   位置 0 的向量是 $W_{pos}[0]$。
*   位置 511 的向量是 $W_{pos}[511]$。

### 2.2 缺点
*   **长度限制**: 必须预先定义最大长度（如 BERT 的 512）。超过这个长度，模型就无法处理了（直接报错或截断）。
*   **数据稀疏**: 靠后的位置（如 2047）在训练数据中出现频率低，可能训练不充分。

## 3. 语音模型中的位置编码

### 3.1 卷积位置编码 (Convolutional Positional Embedding)
在语音模型（如 Wav2Vec 2.0, HuBERT）中，通常不使用上述两种编码。
*   **原因**: 语音特征（Spectrogram）本身具有很强的局部相关性。
*   **做法**: 在 Transformer 之前，先通过几层 **1D 卷积层**。
*   **效果**: 卷积操作天然带有相对位置信息（因为它只看局部窗口）。这被称为“隐式位置编码”。

### 3.2 Whisper 的做法
Whisper 使用了 Sinusoidal 编码，但它将音频特征的时间轴限制在 30 秒（3000 帧）。超过 30 秒的音频会被切片处理。这种硬编码的绝对位置限制了其处理超长连续语音的能力。
