# 视觉 Token：Patch 与 ViT

要让 Transformer 处理图像，首先需要将连续的 2D 图像转换为离散的 1D 序列。**Patch (图块)** 是实现这一转换的核心概念。

## 1. Vision Transformer (ViT) 的诞生

在 ViT (2020) 之前，CNN 是计算机视觉的绝对霸主。ViT 证明了：如果将图像切成块，直接喂给 Transformer，效果可以超越 CNN。

## 2. Patch Embedding 流程

假设输入图像大小为 $H \times W \times C$ (如 $224 \times 224 \times 3$)。

### 2.1 切块 (Patching)
将图像切分为固定大小的 $P \times P$ 小块 (Patch)。通常 $P=16$。
*   Patch 数量 $N = \frac{H \times W}{P \times P} = \frac{224 \times 224}{16 \times 16} = 196$。
*   每个 Patch 包含的信息量：$P \times P \times C = 16 \times 16 \times 3 = 768$ 个像素值。

### 2.2 线性投影 (Linear Projection)
将每个 Patch 展平 (Flatten) 为一个 1D 向量，然后通过一个线性层映射到 $D$ 维 (Embedding Dimension)。
*   这就得到了 $N$ 个向量，类似于 NLP 中的 $N$ 个 Word Embedding。

### 2.3 添加位置编码
图像是有空间结构的，Patch 的位置很重要。需要给每个 Patch Embedding 加上位置编码 (Positional Encoding)。

### 2.4 加上 [CLS] Token
类似于 BERT，在序列开头加一个可学习的 `[CLS]` Token，用于汇聚全局信息进行分类。

## 3. 为什么 Patch 是“视觉单词”？

*   在 NLP 中，句子 = 单词的序列。
*   在 ViT 中，图像 = Patch 的序列。
*   Transformer 的 Self-Attention 机制可以捕捉 Patch 之间的长距离依赖（例如：左上角的狗头和右下角的狗尾巴属于同一只狗）。

## 4. 进阶：多模态大模型 (MLLM) 中的应用

在 GPT-4V, LLaVA 等多模态模型中，视觉编码器 (Vision Encoder, 通常是 CLIP-ViT) 的作用就是将图像变成 Patch Token 序列。
*   **对齐 (Alignment)**: 通过一个 Projector (线性层或 MLP)，将视觉 Token 的维度变换为 LLM 的文本 Token 维度。
*   **输入**: 图像 Token 序列 + 文本 Token 序列 -> LLM。
*   **结果**: LLM 像“看”文字一样“看”到了图像。
