# 离散视觉 Token：VQ-VAE 与 dVAE

ViT 的 Patch Embedding 是连续的向量。但在某些生成任务（如 DALL-E 1, VQGAN）中，我们需要将图像表示为**离散的 ID 序列**，就像文本 Token 一样。这需要用到 **向量量化 (Vector Quantization)**。

## 1. 为什么要离散化？

如果图像能变成一串整数 ID (如 `[352, 1024, 88, ...]`)，我们就可以直接用 GPT 这种强大的 Transformer 来预测下一个像素块，从而生成图像！

## 2. VQ-VAE (Vector Quantized VAE)

### 2.1 码本 (Codebook)
核心思想是维护一个固定的“词表”（Codebook），包含 $K$ 个向量 $e_1, \dots, e_K$。

### 2.2 编码过程
1.  **Encoder**: 将图像 $x$ 编码为特征图 $z_e(x)$。
2.  **Quantization (量化)**: 对于特征图中的每个向量，在 Codebook 中找到与它距离最近的向量 $e_k$，并用 $k$ (索引 ID) 替换它。
    $$ z_q(x) = e_k, \text{ where } k = \arg\min_j \|z_e(x) - e_j\|_2 $$
3.  **Decoder**: 根据量化后的向量 $z_q(x)$ 重建图像。

### 2.3 梯度直通估计 (Straight-Through Estimator)
量化操作 ($\arg\min$) 是不可导的。训练时，我们将 Decoder 的梯度直接复制给 Encoder，跳过量化层。

## 3. dVAE (Discrete VAE) 与 DALL-E 1

OpenAI 在 DALL-E 1 中使用了 dVAE。
*   将 $256 \times 256$ 的图像压缩为 $32 \times 32$ 的 Grid。
*   每个 Grid 对应一个 Token ID (词表大小 8192)。
*   一张图变成了 $32 \times 32 = 1024$ 个 Token。
*   **生成**: 文本 Token + 图像 Token 拼接，训练 GPT 预测下一个 Token。

## 4. VQGAN

VQGAN 进一步引入了**对抗损失 (Adversarial Loss)** 和 **感知损失 (Perceptual Loss)**，使得重建的图像更加清晰、逼真。它是 Stable Diffusion (Latent Diffusion) 的基础组件。

## 5. 总结

| 方法 | 表示形式 | 适用场景 |
| :--- | :--- | :--- |
| **ViT Patch** | 连续向量 | 判别任务 (分类), 多模态理解 (LLaVA) |
| **VQ-VAE / dVAE** | 离散 ID | 自回归图像生成 (DALL-E 1, Parti) |
| **VAE (SD)** | 连续分布 | 扩散模型生成 (Stable Diffusion) |
