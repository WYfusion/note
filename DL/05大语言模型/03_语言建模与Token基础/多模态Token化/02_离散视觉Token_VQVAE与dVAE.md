---
tags:
  - LLM/多模态
  - LLM/生成
aliases:
  - 离散视觉表示
  - VQ-VAE
  - dVAE
  - 向量量化
created: 2025-01-01
updated: 2026-03-28
---

# 离散视觉 Token：VQ-VAE 与 dVAE

> [!abstract] 摘要
> 离散视觉 Token 是将连续图像表示为离散 ID 序列的技术，为自回归图像生成奠定基础。VQ-VAE 和 dVAE 通过向量量化实现图像的离散化，使得图像可以像文本一样被 GPT 等语言模型处理。

## 0. 统一概念：为什么需要离散视觉 Token？

ViT 的 Patch Embedding 是连续的向量。但在某些生成任务（如 DALL-E 1, VQGAN）中，我们需要将图像表示为**离散的 ID 序列**，就像文本 Token 一样。这需要用到 **向量量化 (Vector Quantization)**。

## 1. 为什么要离散化？

> [!important] 离散化的革命性意义
> 如果图像能变成一串整数 ID (如 `[352, 1024, 88, ...]`)，我们就可以直接用 GPT 这种强大的 Transformer 来预测下一个像素块，从而生成图像！

**离散化的优势**：
- **统一表示**：图像与文本共享相同的 Token 表示
- **生成能力**：利用语言模型的强大生成能力
- **压缩比**：大幅减少存储和计算需求
- **语义一致性**：离散 Token 能更好地捕捉语义信息

> [!note] 自回归图像生成流程
> ```
> 文本提示 → [文本编码] → [GPT] → [图像 Token 序列] → [解码器] → 图像
>                                     ↑
>                             预测下一个像素块
> ```

## 2. VQ-VAE (Vector Quantized VAE)

### 2.1 码本 (Codebook)

> [!important] 码本的核心思想
> 核心思想是维护一个固定的”词表”（Codebook），包含 $K$ 个向量 $e_1, \dots, e_K$。

**码本的特点**：
- 固定大小：通常 $K = 512 \sim 8192$
- 向量维度：与 Encoder 输出维度相同
- 可学习：通过训练更新码本向量

> [!example] 码本示例
> ```python
> # 码本结构
> codebook = torch.nn.Embedding(K, D)  # K个向量，每个维度D
> # 查找最近邻
> distances = torch.cdist(z_e, codebook.weight)  # 计算距离
> indices = torch.argmin(distances, dim=-1)      # 找最近索引
> ```

### 2.2 编码-量化-解码流程

> [!important] VQ-VAE 三阶段流程

1.  **Encoder**: 将图像 $x$ 编码为特征图 $z_e(x)$。
2.  **Quantization (量化)**: 对于特征图中的每个向量，在 Codebook 中找到与它距离最近的向量 $e_k$，并用 $k$ (索引 ID) 替换它。
    $$ z_q(x) = e_k, \text{ where } k = \arg\min_j \|z_e(x) - e_j\|_2 $$
3.  **Decoder**: 根据量化后的向量 $z_q(x)$ 重建图像。

> [!note] 损失函数组成
> VQ-VAE 的总损失包含三个部分：
> $$\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \mathcal{L}_{\text{vq}} + \gamma \mathcal{L}_{\text{commit}}$$
> - **重建损失**：像素级 MSE
> - **码本损失**：码本向量的 L2 正则化
> - **承诺损失**：鼓励 Encoder 输出接近码本

### 2.3 梯度直通估计 (Straight-Through Estimator)

> [!warning] 量化的不可导性
> 量化操作 ($\arg\min$) 是不可导的。训练时，我们将 Decoder 的梯度直接复制给 Encoder，跳过量化层。

**STE 的实现**：
```python
def straight_through(z_e, codebook):
    # 前向：找到最近邻
    indices = torch.argmin(torch.cdist(z_e, codebook.weight), dim=-1)
    z_q = codebook(indices)

    # 反向：梯度直通
    z_q = z_e + (z_q - z_e).detach()
    return z_q, indices
```

> [!tip] STE 的直观理解
> **STE** 就像”把信投进信箱，但回执时直接用原始信件”：
> - 前向：量化到最近码本向量
> - 反向：假装没发生量化，直接传递梯度

## 3. dVAE (Discrete VAE) 与 DALL-E 1 #LLM/生成

> [!important] DALL-E 1 的架构创新
> OpenAI 在 DALL-E 1 中使用了 dVAE，实现了文本到图像的自回归生成。

### 3.1 dVAE 的具体实现

> [!tip] dVAE 的压缩策略
> *   将 $256 \times 256$ 的图像压缩为 $32 \times 32$ 的 Grid。
> *   每个 Grid 对应一个 Token ID (词表大小 8192)。
> *   一张图变成了 $32 \times 32 = 1024$ 个 Token。

**压缩比计算**：
- 原始图像：$256 \times 256 \times 3 = 196,608$ 个像素
- Token 序列：$32 \times 32 = 1,024$ 个 Token
- 压缩比：$196,608 / 1,024 = 192:1$

### 3.2 自回归图像生成

> [!important] 文本到图像的生成流程
> **生成**：文本 Token + 图像 Token 拼接，训练 GPT 预测下一个 Token。

```
输入: "a cat wearing a hat"
       ↓
[文本编码] → [GPT-3] → [图像 Token 序列] → [dVAE 解码] → 输出图像
       ↑                     ↑
   文本提示(256 tokens)   预测下一个 Token
```

> [!note] DALL-E 1 的局限性
> - **低分辨率**：生成的图像只有 256×256
> - **质量一般**：不如后来的扩散模型
> - **训练成本高**：需要大量计算资源

### 3.3 与 VQ-VAE 的区别

| 特性 | VQ-VAE | dVAE (DALL-E 1) |
|------|---------|-----------------|
| **码本大小** | 512-2048 | 8192 |
| **压缩率** | 中等 | 高 (192:1) |
| **应用** | 一般图像编码 | 高分辨率图像生成 |
| **训练策略** | 单独训练 | 与 GPT 联合训练 |

## 4. VQGAN：对抗训练的改进 #LLM/生成

> [!important] VQGAN 的创新点
> VQGAN (Vector-Quantized GAN) 进一步引入了**对抗损失 (Adversarial Loss)** 和 **感知损失 (Perceptual Loss)**，使得重建的图像更加清晰、逼真。

### 4.1 VQGAN 的损失函数组成

| 损失类型 | 作用 | 公式 |
|----------|------|------|
| **重建损失** | 像素级保真度 | $\mathcal{L}_{\text{recon}} = \|x - \hat{x}\|_1$ |
| **对抗损失** | 提升视觉质量 | $\mathcal{L}_{\text{adv}} = -\mathbb{E}[\log D(\hat{x})]$ |
| **感知损失** | 保持语义一致性 | $\mathcal{L}_{\text{perc}} = \| \phi(x) - \phi(\hat{x}) \|_2$ |
| **码本损失** | 保持码本稳定 | $\mathcal{L}_{\text{code}} = \| \text{sg}(z_e) - e_k \|_2$ |

> [!note] 感知损失的作用
> **感知损失** 使用预训练的 CNN（如 VGG）提取特征，确保重建图像在语义层面相似，而不仅仅是像素级相似。

### 4.2 VQGAN 与 Stable Diffusion

> [!important] VQGAN 的重要地位
> VQGAN 是 Stable Diffusion (Latent Diffusion) 的基础组件，提供了高质量的图像编码器。

**Stable Diffusion 中的角色**：
- **编码器**：将图像压缩到潜空间
- **码本**：定义潜空间的离散表示
- **解码器**：从潜空间重建图像

---

## 5. 离散视觉 Token 方法对比

### 5.1 三种主要方法总结

| 方法 | 表示形式 | 码本大小 | 压缩比 | 适用场景 | 代表模型 |
|------|----------|----------|--------|----------|----------|
| **ViT Patch** | 连续向量 | 无 | 1:1 | 判别任务, 多模态理解 | ViT, CLIP |
| **VQ-VAE / dVAE** | 离散 ID | 512-8192 | 32:1 - 192:1 | 自回归图像生成 | DALL-E 1, Parti |
| **VAE (SD)** | 连续分布 | 无 | 8:1 | 扩散模型生成 | Stable Diffusion |

### 5.2 选择策略指南

> [!important] 如何选择视觉 Token 方法

**选择 ViT Patch 的场景**：
- ✅ 需要高精度图像理解
- ✅ 多模态任务（如 VQA, 图文检索）
- ✅ 计算资源充足
- ❌ 不需要生成图像

**选择 VQ-VAE/dVAE 的场景**：
- ✅ 需要自回归生成
- ✅ 与文本序列统一处理
- ✅ 需要离散表示
- ❌ 对图像质量要求极高

**选择 VAE (SD) 的场景**：
- ✅ 高质量图像生成
- ✅ 扩散模型生态
- ✅ 计算资源充足
- ❌ 不需要自回归生成

### 5.3 性能对比指标

| 指标 | ViT Patch | VQ-VAE | VAE (SD) |
|------|-----------|---------|----------|
| **FID 分数** | 低（判别任务） | 中等 | 最低（生成任务） |
| **训练速度** | 快 | 中等 | 慢 |
| **推理速度** | 快 | 快 | 中等 |
| **内存占用** | 高 | 中等 | 高 |
| **生成可控性** | 低 | 高（自回归） | 中等 |

> [!tip] FID 分数
> **FID (Fréchet Inception Distance)** 是评估生成图像质量的指标，分数越低表示生成质量越好。

---

## 6. 最新进展：混合 Token 方法

### 6.1 TokenPing-128

> [!note] 混合 Token 的创新
> 结合了连续和离散表示的优点：
> - 使用 128 个码本向量
> - 结合 Patch 和 Residual Token
- 支持高分辨率图像生成

### 6.2 Multi-Codebook VQ-VAE

> [!example] 多码本策略
> ```python
> # 使用多个码本捕获不同粒度的特征
> codebook_fine = Embedding(512, 64)   # 细粒度特征
> codebook_medium = Embedding(256, 128) # 中粒度特征
> codebook_coarse = Embedding(128, 256)  # 粗粒度特征
> ```

---

## 相关链接

**所属模块**：[[索引_多模态Token化]]

**前置知识**：
- [[../../01_语言建模目标_MLE与交叉熵困惑度|语言建模目标]] — 理解自回归建模
- [[../../Tokenizer与分词/01_BPE_WordPiece_Unigram|子词分词]] — 离散化原理

**相关主题**：
- [[../01_视觉Token_Patch与ViT|视觉 Token (ViT)]] — 连续视觉表示
- [[../03_音频Token_AudioCodec与Whisper|音频 Token (Codec)]] — 其他模态的 Token 化
- [[../../04_Transformer核心结构/模型家族/03_Encoder_Decoder_T5_Whisper|Whisper 架构]] — Encoder-Decoder 模型

**延伸阅读**：
- [[../../11_多模态与跨模态/索引_多模态与跨模态|多模态大模型]] — 完整多模态架构
- [[../../11_多模态与跨模态/生成模型/DALL-E与Midjourney|图像生成模型]] — 实际应用案例

