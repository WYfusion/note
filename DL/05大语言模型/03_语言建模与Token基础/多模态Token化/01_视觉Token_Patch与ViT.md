---
tags:
  - LLM/多模态
aliases:
  - 视觉Patch化
  - ViT Token
  - 图像Token
created: 2025-01-01
updated: 2026-03-28
---

# 视觉 Token：Patch 与 ViT

> [!abstract] 摘要
> 视觉 Token 是将连续的 2D 图像转换为离散的 1D 序列的关键技术。ViT (Vision Transformer) 通过图像 Patch 化，将图像转化为类似文本的 Token 序列，为多模态大模型奠定基础。

## 0. 统一概念：为什么需要视觉 Token？ 

要让 Transformer 处理图像，首先需要将连续的 2D 图像转换为离散的 1D 序列。**Patch (图块)** 是实现这一转换的核心概念。

## 1. Vision Transformer (ViT) 的诞生  

在 ViT (2020) 之前，CNN 是计算机视觉的绝对霸主。ViT 证明了：如果将图像切成块，直接喂给 Transformer，效果可以超越 CNN。

> [!important] ViT 的历史意义 
> **ViT (Vision Transformer)** 是 Google 在 2020 年提出的革命性模型，首次证明了 Transformer 在图像识别任务上能够超越 CNN。这项工作的关键创新是：将图像视为序列，而非网格结构。

## 2. Patch Embedding 流程 

假设输入图像大小为 $H \times W \times C$ (如 $224 \times 224 \times 3$)。

### 2.1 切块 (Patching) 

将图像切分为固定大小的 $P \times P$ 小块 (Patch)。通常 $P=16$。

*   Patch 数量 $N = \frac{H \times W}{P \times P} = \frac{224 \times 224}{16 \times 16} = 196$。
*   每个 Patch 包含的信息量：$P \times P \times C = 16 \times 16 \times 3 = 768$ 个像素值。

> [!example] Patch 计算示例
> ```python
> # 224x224 图像，16x16 Patch
> H, W, C = 224, 224, 3
> P = 16
> num_patches = (H // P) * (W // P)  # 196
> patch_dim = P * P * C  # 768
> ```

### 2.2 线性投影 (Linear Projection) 

将每个 Patch 展平 (Flatten) 为一个 1D 向量，然后通过一个线性层映射到 $D$ 维 (Embedding Dimension)。

*   这就得到了 $N$ 个向量，类似于 NLP 中的 $N$ 个 Word Embedding。

> [!tip] Patch Embedding 维度
> 在 ViT-Base 中，$D = 768$；在 ViT-Large 中，$D = 1024$。这个维度应该与 Transformer 的隐藏层维度一致。

### 2.3 位置编码与序列建模

> [!important] 位置编码的重要性 
> 图像是有空间结构的，Patch 的位置很重要。需要给每个 Patch Embedding 加上位置编码 (Positional Encoding)。

与 [[../04_Transformer核心结构/03_位置编码与长上下文/01_绝对位置编码_Sinusoidal_Learnable|绝对位置编码]] 类似，但 ViT 学习可学习的位置编码：

$$PE_{(i,j)} = \begin{cases}
\cos\left(\frac{i}{10000^{2j/d}}\right) & \text{if } 2j < d \\
\sin\left(\frac{i}{10000^{(2j+1)/d}}\right) & \text{if } 2j \geq d
\end{cases}$$

其中 $i$ 是 Patch 的序号，$(i,j)$ 表示 Patch 在空间中的位置。

### 2.4 [CLS] Token 的作用

类似于[[02_特殊Token_对齐与模板的影响|BERT 的 [CLS] Token]]]，在序列开头加一个可学习的 `[CLS]` Token，用于汇聚全局信息进行分类。

> [!note] [CLS] Token 的数学意义
> **[CLS] Token** 是可学习的参数向量，通过 Self-Attention 机制汇聚所有 Patch 的信息。在分类任务中，最终只使用 [CLS] 的输出表示整个图像：
> $$\text{Prediction} = \text{Linear}(\text{Transformer}(\text{[CLS]}))$$

## 3. 为什么 Patch 是”视觉单词”？ 

> [!important] Patch 与单词的类比 
> 在 NLP 与 CV 的统一框架中：

| 维度 | NLP | CV |
|------|-----|-----|
| **基本单元** | Word | Patch |
| **序列表示** | Sentence = Word₁, Word₂, ..., Wordₙ | Image = Patch₁, Patch₂, ..., Patchₙ |
| **信息表示** | Word Embedding | Patch Embedding |
| **位置信息** | Positional Encoding | 2D Positional Encoding |
| **全局信息** | [CLS] Token | [CLS] Token |

> [!example] 长距离依赖的捕获
> Transformer 的 Self-Attention 机制可以捕捉 Patch 之间的长距离依赖（例如：左上角的狗头和右下角的狗尾巴属于同一只狗）：
> $$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

这种机制使得 ViT 能够理解图像的全局结构，而不像 CNN 那样只能感受局部感受野。

## 4. ViT 在多模态大模型中的应用 #LLM/多模态

在 GPT-4V, LLaVA 等多模态模型中，视觉编码器 (Vision Encoder, 通常是 CLIP-ViT) 的作用就是将图像变成 Patch Token 序列。

### 4.1 多模态对齐机制 

**对齐 (Alignment)**：通过一个 Projector（线性层或 MLP），将视觉 Token 的维度变换为 LLM 的文本 Token 维度。

> [!important] 对齐的挑战
> - **维度不匹配**：视觉 Token（如 768 维）与文本 Token（如 4096 维）维度不同
> - **语义鸿沟**：视觉特征与文本特征属于不同模态
> - **信息密度差异**：视觉信息更密集，文本信息更稀疏

### 4.2 多模态输入结构 

**输入**：图像 Token 序列 + 文本 Token 序列 → LLM

```
[视觉编码器]        [Projector]        [LLM]
   图像  --------→  视觉Token --------→  文本Token
                              ↘
                              LLM (理解视觉内容)
```

### 4.3 从图像到语言的理解 

**结果**：LLM 像”看”文字一样”看”到了图像。

> [!note] 多模态能力评估
> - **视觉问答 (VQA)**：模型回答关于图像的问题
> - **图像描述 (Captioning)**：生成图像的文字描述
> - **视觉推理**：理解图像中的逻辑关系
> - **跨模态检索**：根据文字找图片，或根据图片找文字

### 4.4 主流视觉编码器对比

| 模型 | 维度 | 预训练数据 | 特点 |
|------|------|------------|------|
| **ViT-B/16** | 768 | ImageNet-21K | 基础版本，16x16 Patch |
| **ViT-L/14** | 1024 | JFT-300M | 大版本，14x14 Patch |
| **CLIP-ViT** | 512/768 | Web-scale 图文对 | 多模态预训练 |
| **DINO-ViT** | 384/768 | 无监督 | 自监督特征学习 |

---

## 5. 视觉 Token 的未来趋势 

### 5.1 高分辨率视觉 Token

> [!warning] 高分辨率挑战
> 传统 ViT 难以处理高分辨率图像，因为 Patch 数量爆炸：
> - 1080p 图像：`1920×1080/16² = 8100` 个 Patch
> - 4K 图像：`3840×2160/16² = 32400` 个 Patch

**解决方案**：
- **层次化 Patch**：大图像 → 大 Patch → 小 Patch
- **滑动窗口**：Patch 之间有重叠
- **稀疏注意力**：只关注相关区域

### 5.2 动态 Patch 大小

> [!tip] 可变 Patch 策略
```python
# 根据图像复杂度动态调整 Patch 大小
def adaptive_patch_size(image):
    # 简单区域：大 Patch（减少计算）
    # 复杂区域：小 Patch（保留细节）
    return dynamic_patches
```

### 5.3 3D 视觉 Token

对于视频和 3D 数据，需要引入时间维度：
- **视频**：时空 Patch = 2D 空间 Patch + 1D 时间切片
- **3D 点云**：将 3D 空间划分为体素 (Voxels)

---

## 相关链接

**所属模块**：[[索引_多模态Token化]]

**前置知识**：
- [[../../01_语言建模目标_MLE与交叉熵困惑度|语言建模目标]] — 理解序列建模基础
- [[../../Tokenizer与分词/01_BPE_WordPiece_Unigram|子词分词]] — 文本分词与 Patch 的类比

**相关主题**：
- [[../02_离散视觉Token_VQVAE与dVAE|离散视觉 Token]] — 基于学习的视觉 Token 化
- [[../03_音频Token_AudioCodec与Whisper|音频 Token (Codec)]] — 其他模态的 Token 化
[[00_缩放点积注意力_为什么是点积_为什么要除以根号dk|缩放点积注意力]]]] — Self-Attention 机制

**延伸阅读**：
- [[../../11_多模态与跨模态/索引_多模态与跨模态|多模态大模型]] — 完整多模态架构
- [[../../11_多模态与跨模态/模型架构/CLIP与FLAME|CLIP模型]] — 图文对齐的先驱



