---
tags:
  - 机器学习
  - 深度学习
  - 自监督学习
  - BERT
  - Transformer
  - NLP
created: 2025-01-18
modified: 2025-01-18
---

> [!summary] 核心思想
> BERT 是基于 Transformer 编码器的双向预训练模型，通过大规模无监督预训练（掩码语言模型 MLM 和下一句预测 NSP）实现双向上下文建模，首次在 11 项 NLP 任务中刷新了 SOTA。

# BERT（Bidirectional Encoder Representations from Transformers）

## 概述

Google 于 2018 年发布的基于 Transformer 编码器的模型，首次实现双向上下文建模。通过大规模无监督预训练（如掩码语言模型 MLM 和下一句预测 NSP），BERT 在 11 项 NLP 任务中刷新了 SOTA。

BERT 是实现填空的作用，但是可以用在其他下游任务上，这个任务可以不一定和填空有关，甚至无关也可以。

### 架构特点

`BERT` 是 `Transformer` 的 `Encoder` 架构：
- 输入一排的向量
- 输出就有一排同长向量

---

## 1. BERT 在预训练中的实现

预训练的作用其实就是一个初始化的过程，为了更好地用于下游任务中去。

### 1.1 掩码语言模型（Masked Language Model, MLM）

#### 掩码策略

BERT 对输入序列中的每个 token 独立评估，以 **15% 的概率**决定是否掩码。注意这种策略是离散位置单个化的 token。

#### 替换方式

将被盖住的位置处换为新的 token，这 15% 的 token 中，并非所有都被替换为掩码符号，而是进一步分为三种情况：

1. **80% 的概率**替换为 `<MASK>` 符号（特定掩蔽 token）
2. **10% 的概率**替换为随机 token
3. **10% 的概率**保持原 token 不变（保留上下文信息以缓解预训练与微调的不匹配）

也即先随机选中被替换 token → 再随机挑选替换 token将被选中的 token 替换即可。

### 1.2 预测流程

1. 得到新的 seq 进入 BERT 中去得到另一个 seq
2. 将被掩码位置的输出向量（隐藏状态）经过一个**线性层（无激活函数）**
3. 映射到词表维度
4. **分类目标**：直接预测被掩码 token 在词表中的原始 ID
5. 计算预测概率分布与真实标签（one-hot 编码的原始 token ID）之间的交叉熵

简而言之就是做一个分类，分类目标是原被掩蔽的 token 所代表的特征向量。分类个数等于词表大小。

![[Pasted image 20250317193719.png|1200]]

### 1.3 下一句预测（Next Sentence Prediction, NSP）

- 预测两个句子是否是连续的
- 辅助任务，帮助模型理解句子间的关系

---

## 2. BERT 的应用

虽然 BERT 被设计时貌似只能做填空题，但是可以被分化为各式各样的任务。

### 2.1 Fine-tune 微调的好处

明显比随机初始化的 BERT 好训练得多，并且训练完成后依旧比随机初始化的 BERT 更好。

- 图中虚线的是随机初始化的
- 实线的是 Pre-train 后 fine-tune 的

![[Pasted image 20250318141135.png|600]]

### 2.2 微调示例

#### Case1：句子情感倾向

- 输入句子
- 输出情感分类（正面/负面）

#### Case2：句子内词性辨析

- 输入句子
- 输出每个词的词性标签

#### Case3：两句子关系辨别

- 前提句与结论句关系辨析
- 可以用作判定发言立场

#### Case4：问答模型

- 输入 Document 和 Question
- 输出 s、e 整数数字
- 答案是 Document 从 s 到 e 的部分

![[Pasted image 20250317212727.png|1000]]

---

## 3. BERT 与 De-noising Auto-encoder 的相似性

![[Pasted image 20250318133144.png|600]]

BERT 的掩码语言模型类似于去噪自编码器：
- 通过添加噪声（掩码）破坏输入
- 训练模型恢复原始输入
- 学习鲁棒的特征表示

---

## 4. BERT 为何有效

### 4.1 上下文相关的词嵌入

不同的句子下的相同 token 之间的含义很有可能不一样。例如水果苹果和苹果电脑，可见不同上下文下，苹果含义不同。

![[Pasted image 20250317223806.png|800]]
![[Pasted image 20250317223711.png|800]]

BERT 在学习的过程中就学到了上下文的信息。所以说 BERT 所抽出来的向量就是可以说是 **Contextualized word embedding**（上下文词嵌入）。

### 4.2 可解释性问题

当然，BERT 有效的可解释性比较差，可能很不相关的用了也会有效果。

### 4.3 跨语言表现

对于语言填空方面，也有奇效，甚至跨语言下也可以表现的很好，但是需要预训练的资料量够高。

### 4.4 假收敛现象

可能会出现"假收敛现象"。

![[Pasted image 20250317225326.png|600]]
![[Pasted image 20250317225749.png|600]]

---

## 5. BERT 的变体

### BART：双向编码器-自回归解码器的融合

#### 核心架构

- BART 采用**编码器-解码器架构**
- 编码器类似 BERT 的双向 Transformer
- 解码器类似 GPT 的单向自回归 Transformer
- 这种设计使其既能理解上下文（编码器），又能生成连贯文本（解码器）

#### 替换策略

- 特定掩蔽 token、随机 token 替换
- 删除 token
- 更换句子顺序
- 打乱不同句子 token
- 同时掩蔽不同句子的 token

### MASS：面向生成的统一掩码序列建模

#### 核心架构

- MASS 同样为**编码器-解码器结构**
- 但专注于**序列到序列的生成任务**

#### 掩码策略

- **连续跨度掩码**：对输入序列中连续的 k 个 token 进行掩码（k 为输入长度的 50%）
- **自回归预测**：编码器处理被掩码的输入，解码器自回归生成被掩码的片段，强化对长距离依赖的建模

### 三者对比

| **模型** | **核心特点** | **优势** |
|---------|-------------|---------|
| **BERT** | 随机单 token 掩码 | 平衡上下文学习与噪声鲁棒性，奠定双向理解基础 |
| **BART** | 多样化噪声模拟 | 融合生成与理解能力，适应复杂任务 |
| **MASS** | 长跨度掩码 | 以生成任务为核心，通过长跨度掩码强制模型学习序列全局依赖 |

![[Pasted image 20250317220008.png|600]]

---

## 相关链接

- [[Transformer]] - BERT 的基础架构
- [[自监督预训练/Self-supervised Learning Framework]] - 自监督学习框架
- [[00_transformer阅读指引]] - Transformer 学习指引
- [[Transformer/02_输入表示]] - 输入表示和分词

## 参考资料

- Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL*
- Lewis, M., et al. (2020). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. *ACL*
- Song, K., et al. (2019). MASS: Masked Sequence to Sequence Pre-training for Language Generation. *ICML*

