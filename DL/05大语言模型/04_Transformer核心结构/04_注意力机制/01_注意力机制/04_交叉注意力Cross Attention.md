---
tags:
  - LLM/Transformer
  - 注意力机制/交叉注意力
  - 架构/EncoderDecoder
aliases:
  - Cross Attention
  - Cross-Attention
updated: 2026-03-29
---

# 交叉注意力（Cross Attention）

> [!abstract]
> 交叉注意力做的事很简单：Decoder 带着“我下一步需要什么”的问题，去 Encoder 提供的条件信息里做检索。

> [!note]
> 交叉注意力的思想在早期 seq2seq attention 中就已经出现，Transformer 在 2017 年把它标准化为 Decoder 中独立的一个子层。

## 交叉注意力到底解决了什么问题

如果 Decoder 只有 masked self-attention，它只能看到“已经生成过的目标前缀”，却看不到源序列本身。  
这意味着它知道自己“已经说到哪里”，但不知道“应该根据什么条件继续生成”。

交叉注意力的作用就是把这条条件通路补上：

- 翻译时，条件是源语言句子
- 摘要时，条件是原文表示
- 语音识别时，条件是语音编码特征
- 多模态生成时，条件可能是图像 patch、音频帧或检索文档

## 核心公式

设：

- $H^{dec}$ 是 Decoder 当前层输入
- $H^{enc}$ 是 Encoder 输出

则

$$
Q = H^{dec} W^Q,\quad K = H^{enc} W^K,\quad V = H^{enc} W^V
$$

交叉注意力写作：

$$
\text{CrossAttn}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

这里最关键的一点是：

- `Q` 来自 Decoder
- `K/V` 来自 Encoder

也就是“提问方”和“知识提供方”来自两个不同序列。

## 图上看最直观

![[Pasted image 20250315215541.png|400]]
![[Pasted image 20250315214850.png|650]]
![[Pasted image 20250315214529.png|1000]]

从图上可以把交叉注意力理解成一句话：  
**Decoder 当前状态拿着自己的 Query，去源序列编码后的 Key/Value 里寻找最相关的信息。**

## 与自注意力有什么本质区别

| 维度 | 自注意力 | 交叉注意力 |
| --- | --- | --- |
| Query 来源 | 当前序列 | Decoder 当前序列 |
| Key/Value 来源 | 同一序列 | 条件序列或源序列 |
| 主要作用 | 建模序列内部依赖 | 建立“条件到输出”的对齐关系 |
| 常见位置 | Encoder、Decoder | Encoder-Decoder 的 Decoder 子层 |

## 一个翻译场景下的直觉

假设 Encoder 编码的是英文句子 `I love AI`，Decoder 当前已经生成了“我”，下一步准备生成下一个中文 token。

此时：

- Decoder 的 Query 表示“我接下来需要一个与当前翻译进度最相关的源端信息”
- Encoder 的 Key/Value 则保留了 `I / love / AI` 的上下文表示

交叉注意力会根据 $QK^\top$ 的匹配结果，把与当前生成最相关的 Value 聚合回来。  
如果模型判断下一步最需要“love”对应的语义，那么它就会更强地读取该位置的信息。

这就是“对齐”的来源。它不是硬匹配，而是一个可微的软检索过程。

## 为什么它适合条件生成

### 1. 可以把源端信息按需取回

Decoder 不需要一次性把 Encoder 输出全部记死，而是每一步都可以重新检索。

### 2. 可重复堆叠

多层 Decoder 可以反复做交叉注意力，不断细化对源序列的读取方式。

### 3. 对多模态很自然

只要 Encoder 能把输入编码成一串表示，Decoder 就可以用同样的机制读取它。

## 掩码怎么配

> [!warning]
> 交叉注意力本身一般不需要 causal mask。  
> 因果约束属于 Decoder 前面的 masked self-attention，而不是 cross-attention 本身。

交叉注意力最常见的 mask 是：

- 源序列的 padding mask
- 指定区域可见性的 block mask
- 多源输入时的路由约束 mask

## 典型变体

- 多源交叉注意力：同时读取多个 Encoder 或多个检索源
- 检索增强交叉注意力：读取外部文档、记忆库或工具返回结果
- 轻量化交叉注意力：降低 Decoder 推理成本

![[Pasted image 20250315215652.png]]

## 相关双链

- [[索引_注意力机制]]
- [[03_掩码与因果性]]
- [[03_CrossAttentionMask如何约束条件信息流|Cross-Attention Mask 如何约束条件信息流]]
- [[02_Decoder_Block]]
- [[03_Encoder_Decoder_transformer流程]]
- [[02_EncoderDecoder数据流与CrossAttention位置|Encoder-Decoder 数据流与 Cross-Attention 位置]]

## 参考资料

- Bahdanau, D., et al. *Neural Machine Translation by Jointly Learning to Align and Translate*. ICLR, 2015.
- Vaswani, A., et al. *Attention Is All You Need*. NeurIPS, 2017.
