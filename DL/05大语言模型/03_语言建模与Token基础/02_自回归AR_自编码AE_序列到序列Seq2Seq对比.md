---
tags:
  - LLM/语言建模
aliases:
  - 序列模型对比
  - AR与AE与Seq2Seq
created: 2025-01-01
updated: 2026-03-28
---

# 自回归（AR）、自编码（AE）与序列到序列（Seq2Seq）对比

> [!abstract] 摘要
> 理解大语言模型架构的关键一步是区分三种序列建模范式：自回归（AR）、自编码（AE）和序列到序列（Seq2Seq）。每种范式决定了模型的设计选择、训练目标和适用场景，它们共同构成了现代 LLM 架构的理论基础。

## 1. 三大范式总览

```
┌─────────────────────────────────────────────────────────────┐
│                                                         │
│   ┌───────────────┐         ┌───────────────┐         │
│   │   自回归 AR  │         │   自编码 AE    │         │
│   └───────────────┘         └───────────────┘         │
│         ↓                        ↓                     │
│   ┌───────────────┐         ┌───────────────┐         │
│   │  Seq2Seq      │         │  Seq2Seq      │         │
│   └───────────────┘         └───────────────┘         │
│         ↓                        ↓                     │
└─────────────────────────────────────────────────────────────┘
```

| 范式 | 核心思想 | 典型架构 | 输出类型 | 代表模型 |
|------|----------|----------|----------|----------|
| **AR (自回归)** | 预测下一个 | 生成 | [[../04_Transformer核心结构/模型家族/02_Decoder_Only_GPT_语音模型|GPT]], Llama |
| **AE (自编码)** | 重构输入 | 分类/表示 | [[../04_Transformer核心结构/模型家族/01_Encoder_Only_BERT_Wav2Vec|BERT]], Wav2Vec |
| **Seq2Seq** | 输入→输出 | 生成/转换 | [[../04_Transformer核心结构/模型家族/03_Encoder_Decoder_T5_Whisper|T5]], Whisper |

---

## 2. 自回归（Auto-Regressive, AR）

### 2.1 核心思想

> [!important] 因果预测
> **自回归模型**根据过去的预测未来，具有**因果性（Causal）**：第 $t$ 步只能看到 $t$ 之前的全部信息，不能看到 $t$ 之后的信息。

### 2.2 数学形式

对于序列 $X = (x_1, x_2, \dots, x_T)$：

$$P(X) = \prod_{t=1}^{T} P(x_t | x_{1:t-1})$$

**等价形式（链式法则）**：
$$P(X) = P(x_1) \cdot P(x_2 | x_1) \cdot P(x_3 | x_1, x_2) \cdots P(x_T | x_{1:T-1})$$

### 2.3 训练目标

使用 [[01_语言建模目标_MLE与交叉熵困惑度#MLE|最大似然估计]]或交叉熵损失：

$$\mathcal{L}_{AR} = - \sum_{t=1}^{T} \log P(x_t | x_{<t}; \theta)$$

### 2.4 特点总结

| 特点 | 说明 |
|------|------|
| **因果性** | 只看过去，不看未来 |
| **串行生成** | 逐 token 生成 |
| **概率模型** | 输出概率分布，可采样多样性 |
| **适用架构** | [[../04_Transformer核心结构/02_注意力机制/02_AttentionMask_Causal_Padding_CrossAttention|Causal Mask]], Decoder-only |

> [!note] AR 是现代 LLM 的基础
> 所有主流文本生成模型（GPT、Llama、Qwen、DeepSeek）都是 AR 模型。

---

## 3. 自编码（Auto-Encoding, AE）

### 3.1 核心思想

> [!important] 重构表示
> **自编码模型**试图重构输入本身，关注编码-解码流程。模型先学习将输入压缩成低维表示（编码），再从表示中恢复原始输入（解码）。

### 3.2 架构

```
输入 X
  ↓
Encoder (压缩)
  ↓
潜在表示 Z (低维)
  ↓
Decoder (重构)
  ↓
输出 X' (应 ≈ X)
```

### 3.3 训练方式

#### Denoising Auto-Encoder（DAE）

> [!tip] 掩码自编码
> 将输入添加噪声，训练模型去噪并恢复原始输入。
> - 提高鲁棒性
> - [[../04_Transformer核心结构/模型家族/01_Encoder_Only_BERT_Wav2Vec|BERT]] 采用此策略（Masked LM）

### 3.4 特点总结

| 特点 | 说明 |
|------|------|
| **双向性** | Encoder 可看到整个序列，无因果限制 |
| **非生成** | 主要用于分类、表示学习 |
| **潜在空间** | 学习压缩表示，可用于下游任务 |
| **适用架构** | Encoder-only, Encoder-Decoder |

---

## 4. 序列到序列（Sequence-to-Sequence, Seq2Seq）

### 4.1 核心思想

> [!important] 输入到输出
> **Seq2Seq 模型**学习从一个序列到另一个序列的映射。输出序列长度可与输入不同（如翻译、摘要）。

### 4.2 架构

```
输入序列 (长度 T1)
        ↓
    Encoder (RNN/Transformer)
        ↓
    上下文向量 C (固定或动态)
        ↓
    Decoder (RNN/Transformer)
        ↓
输出序列 (长度 T2)
```

### 4.3 注意力机制的作用

在传统 RNN Seq2Seq 中，使用 **Bahdanau Attention** 解决长距离依赖：

$$c_t = \sum_{j=1}^{T_x} \alpha_{tj} h_j$$

其中 $\alpha_{tj}$ 是注意力权重：

$$\alpha_{tj} = \frac{\exp(e_{tj})}{\sum_{k} \exp(e_{tk})}$$

$$e_{tj} = v_a^T \tanh(W_a s_{t-1} + U_a h_j)$$

> [!note] 注意力演进
> 这为 [[02_Transformer与注意力革命|Transformer]] 的 Self-Attention 铺平了道路。

### 4.4 特点总结

| 特点 | 说明 |
|------|------|
| **变长输出** | 适合翻译、摘要等任务 |
| **长距离依赖** | Attention 解决了这个问题 |
| **中间表示** | 上下文向量作为信息瓶颈 |
| **适用架构** | Encoder-Decoder |

---

## 5. 深度对比

### 5.1 适用场景对比

| 任务场景 | 推荐范式 | 代表模型 |
|----------|----------|----------|
| **文本生成** | AR | [[../04_Transformer核心结构/模型家族/02_Decoder_Only_GPT_语音模型|GPT]], Llama, Qwen, DeepSeek |
| **对话系统** | AR | ChatGPT, DeepSeek-Chat, Claude |
| **文本分类** | AE | [[../04_Transformer核心结构/模型家族/01_Encoder_Only_BERT_Wav2Vec|BERT]], RoBERTa |
| **情感分析** | AE | 分类器 |
| **实体识别** | AE | NER 模型 |
| **机器翻译** | Seq2Seq | [[../04_Transformer核心结构/模型家族/03_Encoder_Decoder_T5_Whisper|T5]], mT5 |
| **摘要生成** | Seq2Seq | BART, PEGASUS |
| **语音识别** | AE (编码器) → Seq2Seq 解码 | [[../11_多模态与跨模态/语音语言模型SLM/索引_语音语言模型SLM|Whisper]], Wav2Vec |
| **语音合成** | AE (编码器) → AR 解码 | Tacotron, VITS |

### 5.2 架构选择决策树

```
是否需要生成不同长度的输出？
├─ 否 → 使用 AE（分类/表示）
└─ 是 → 使用 Seq2Seq 或 AR
    ├─ 输出长度与输入有关？→ Seq2Seq
    └─ 输出长度未知？→ AR
```

### 5.3 现代融合

> [!note] 现代模型的混合特性
> 现代 LLM 通常融合多种范式的特性：
> - **[[../04_Transformer核心结构/模型家族/01_Encoder_Only_BERT_Wav2Vec|BERT]]** 风格预训练（去噪 AE 思想）
> - **[[../04_Transformer核心结构/模型家族/02_Decoder_Only_GPT_语音模型|GPT]]** 风格推理（AR 思想）
> - 两者通过 [[03_预训练_指令微调_对齐的演进|指令微调]] 统一

---

## 相关链接

**所属模块**：[[索引_语言建模与Token基础]]

**前置知识**：
- [[01_语言建模目标_MLE与交叉熵困惑度|语言建模目标]] — MLE 与交叉熵基础

**相关主题**：
-[[索引_Transformer核心结构|Transformer 核心结构]]] — 三大范式的统一
- [[../04_Transformer核心结构/模型家族/索引_模型家族|模型家族]] — Encoder-only/Decoder-only/Encoder-Decoder 详细对比
[[00_缩放点积注意力_为什么是点积_为什么要除以根号dk|Scaled Dot-Product Attention]]]] — 注意力的数学基础
[[00_多头注意力为什么有效_MHA_MQA_GQA_MLA|Multi-Head Attention]]]] — 注意力如何统一不同范式

**延伸阅读**：
[[00_缩放点积注意力_为什么是点积_为什么要除以根号dk|Scaled Dot-Product Attention]]]] — 注意力的数学基础
- [[../11_多模态与跨模态/语音语言模型SLM/01_语音表征_离散化_Codec|语音表征]] — 语音领域的 AE 模型（Wav2Vec）


