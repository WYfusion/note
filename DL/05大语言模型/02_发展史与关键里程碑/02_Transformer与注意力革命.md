---
tags:
  - LLM/发展史
aliases:
  - Transformer革命
  - 注意力机制革命
created: 2025-01-01
updated: 2026-03-28
---

# Transformer 与注意力革命

> [!abstract] 摘要
> 2017年 Google 发表「Attention Is All You Need」，彻底抛弃 RNN/CNN 结构，完全基于注意力机制，开启了大模型时代。

## 1. 为什么需要 Transformer？

[[01_从统计语言模型到神经LM|RNN/LSTM]] 存在两个致命弱点：

### 1.1 无法并行

> [!important] 串行计算瓶颈
> 普通RNN计算 $h_t$ 必须等待 $h_{t-1}$，导致在GPU上训练效率极低，无法利用并行加速。

**根本原因**：递归结构的时序依赖。

### 1.2 长距离依赖

> [!warning] 信息衰减问题
> 即使使用[[01_从统计语言模型到神经LM#LSTM|LSTM 门控]]机制，信息在长序列传递中仍会逐渐衰减，难以捕捉远距离的语义关联。

**具体表现**：
- 长句首尾呼应难以建模
- 多轮对话上下文丢失
- 长文档理解能力不足

---

## 2. 核心组件：Scaled Dot-Product Attention

Transformer 通过 *[[01_缩放点积注意力_为什么是点积_为什么要除以根号dk|Self-Attention]]]**  彻底解决了上述问题。

### 2.1 Q, K, V 的定义

对于输入矩阵 $X$（每一行是一个词向量），通过三个线性变换得到：

$$Q = X W^Q, \quad K = X W^K, \quad V = X W^V$$

| 符号 | 含义 | 形状 |
|------|------|------|
| $Q$ (Query) | 查询向量：「想找什么」 | $[L, d_k]$ |
| $K$ (Key) | 键向量：「有什么特征」 | $[L, d_k]$ |
| $V$ (Value) | 值向量：「内容信息」 | $[L, d_v]$ |
| $d_k$ | Key 维度 | 通常 $d_k = d_v$ |

### 2.2 注意力计算公式

> [!important] 核心公式
> $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**四步解析**：
1. **相似度计算 ($QK^T$)**：计算每个 Query 和所有 Key 的点积
   - 形状：$[L, L]$ 的分数矩阵
   - 物理意义：衡量两个位置的关联强度

2. **缩放 ($\frac{1}{\sqrt{d_k}}$)**：
   > [!tip] 缩放原因
   > 当 $d_k$ 很大时，点积值会很大，Softmax 进入饱和区，梯度趋近0。
   > 除以 $\sqrt{d_k}$ 将方差拉回1，保证梯度流动稳定。

3. **归一化 (Softmax)**：将分数转换为概率分布，权重和为1

4. **加权求和 ($\cdot V$)**：根据注意力权重聚合 Value 信息

---

## 3. Multi-Head Attention (多头注意力)

> [!tip] 直观理解
> 「三个臭皮匠，顶个诸葛亮」——将 Embedding 空间切分为多个子空间，每个子空间独立计算 Attention，最后拼接融合。

### 3.1 公式

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_H)W^O$$

$$\text{where } \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**参数量分析**：
| 参数矩阵 | 形状 | 说明 |
|----------|------|------|
| $W_Q, W_K, W_V$ | $[d_{model}, d_{model}]$ | 逻辑上分头，实现时通常是一大矩阵 |
| $W_O$ | $[d_{model}, d_{model}]$ | 输出融合层 |

### 3.2 为什么多头有效？

> [!note] 多头优势
> 1. **多视角 (Diversity)**：不同头可以关注不同类型的模式
>    - Head 1：局部信息（相邻词）
>    - Head 2：长距离依赖（主谓一致）
>    - Head 3：语义指代
> 2. **鲁棒性**：即使某个头「走神」了，其他头还能补充信息

---

## 4. 位置编码 (Positional Encoding)

### 4.1 Self-Attention 的置换不变性

> [!warning] 顺序信息丢失
> Self-Attention 是**置换不变**的：如果打乱句子中词的顺序，Attention 输出结果完全一样。

**原因**：$QK^T$ 只计算词之间的关联，与位置无关。

### 4.2 正弦/余弦位置编码

原论文使用的**绝对位置编码**：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

> [!note] 线性可推性
> 这种编码允许模型通过线性变换学习**相对位置关系**（因为 $\sin(\alpha+\beta)$ 可以展开）。

> [!tip] 现代演进
> 当代 LLM 多使用[[03_RoPE_ALiBi_为什么主流模型偏向这两类方案|RoPE]]]（旋转位置编码），外推性更好。

---

## 5. 整体架构（Encoder-Decoder）

### 5.1 Encoder 结构

```
输入 X
  ↓
Token Embedding + Positional Encoding
  ↓
N × (Self-Attention + Feed-Forward + LayerNorm + Residual)
  ↓
输出：编码后的表示 H
```

**每层两个子层**：
1. **Multi-Head Self-Attention**：双向，能看到整个序列
2. **Position-wise Feed-Forward (FFN)**：$FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2$

**残差连接与层归一化**：
$$\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

### 5.2 Decoder 结构

```
输出 Y (已生成)
  ↓
Masked Self-Attention (只能看已生成的词)
  ↓
Encoder-Decoder Attention (Query 来自 Decoder, Key/Value 来自 Encoder)
  ↓
Feed-Forward + LayerNorm
  ↓
预测：下一个词的概率分布
```

**关键区别**：
| 特性 | Encoder | Decoder |
|------|---------|----------|
| Self-Attention | 双向（全部） | Masked（单向） |
| Cross-Attention | 无 | 有（关注 Encoder） |

---

## 6. 架构演变：三大分支

详见 [[../04_Transformer核心结构/模型家族/索引_模型家族|模型家族详解]]

### 6.1 Encoder-only (编码器架构)

> [!summary] 双向理解
> 只使用 Encoder，双向注意力，**适合理解任务**。

**代表模型**：
- **BERT**：Masked Language Modeling (MLM)
- **Wav2Vec 2.0**：语音自监督学习 #语音/ASR
- **RoBERTa**：BERT 的优化版本

**适用场景**：
- 文本分类
- 实体识别
- 情感分析
- 语音识别 ([[../11_多模态与跨模态/语音语言模型SLM/索引_语音语言模型SLM|语音大模型]])

### 6.2 Decoder-only (解码器架构)

> [!summary] 单向生成
> 只使用 Decoder，**Causal Mask**（单向注意力），**适合生成任务**。

**代表模型**：
- **GPT 系列**：GPT-3, GPT-4, GPT-4o
- **Llama 3**：Meta 开源，主流基座
- **VALL-E**：Microsoft 语音合成 #语音/TTS
- **Qwen 2.5**：阿里开源，中文能力强

**适用场景**：
- 文本生成
- 代码生成
- 语音合成 ([[../11_多模态与跨模态/语音语言模型SLM/01_语音表征_离散化_Codec|语音生成]])
- 对话系统

### 6.3 Encoder-Decoder (编解码架构)

> [!summary] 端到端序列
> 保留完整架构，**适合 Seq2Seq 任务**。

**代表模型**：
- **T5**：Text-to-Text 预训练
- **Whisper**：语音识别 #语音/ASR
- **BART**：去噪自编码器

**适用场景**：
- 机器翻译
- 语音识别 ([[../11_多模态与跨模态/语音语言模型SLM/01_语音表征_离散化_Codec|Whisper详解]])
- 摘要生成

---

## 7. Transformer 的成功原因

> [!important] 四大关键创新
> Transformer 的成功在于彻底解决了 RNN 的核心痛点：

1. **完全并行化**：无递归依赖，GPU 友好，使得大规模训练成为可能
2. **全局视野**：Self-Attention 直接捕捉任意距离的依赖，无视位置限制
3. **通用架构**：不限于文本，扩展到视觉 (ViT)、语音 ([[../11_多模态与跨模态/语音语言模型SLM|音频Token]])、多模态
4. **可扩展性**：Scaling Law 奠行，「堆算力、堆数据、堆参数」模型就会变强

> [!tip] 开启大模型时代
> 这些特性使得 Transformer 成为了从 GPT-2 (1.5B) 到 GPT-4 (估计万亿参数）演进的基石。

---

## 相关链接

**所属模块**：[[索引_发展史与关键里程碑]]

**前置知识**：
- [[01_从统计语言模型到神经LM]] — 理解 RNN/LSTM 的局限性

**相关主题**：
-[[索引_Transformer核心结构|Transformer 核心结构]]] — 详细技术解析
-[[01_缩放点积注意力_为什么是点积_为什么要除以根号dk|Scaled Dot-Product Attention]]] — 核心注意力详解
-[[03_多头注意力为什么有效_MHA_MQA_GQA_MLA|Multi-Head Attention]]] — 多头注意力详解
-[[03_RoPE_ALiBi_为什么主流模型偏向这两类方案|RoPE 相对位置编码]]] — 现代位置编码

**延伸阅读**：
-[[索引_高效Transformer与推理工程|高效注意力与优化]]] — FlashAttention 等推理加速
- [[../05_预训练_数据_规模化训练|索引_预训练_数据_规模化训练|预训练与规模化训练]] — 基于 Transformer 的训练范式



