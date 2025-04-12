在$\text{VIT}$论文里比较值得学习的地方有两点

- 一个是对图片的预处理成$\text{image token}$的$\text{Embedding Patched}$

- 另一个则是$\text{Transformer}$模块里的多头注意力模块

$\text{word embedding}$是针对$\text{context}$进行编码，便于使机器进行学习的方法，而$\text{Embedding patch}$则是针对$\text{image}$进行**编码**，便于机器学习的方法。

同理，$\text{Embedding patch}$也是主要做两个事：

1. 将图片像分词一样划分
2. 将分好的图片（我们这里称为$\text{Patch}$）进行$\text{N(embedded\_dim)}$维空间的映射。

最直观的方法是将二维的图片直接拉成一维的向量，如$28\times 28$的图片，拉成$1\times 784$长度的向量，将784维的向量当成$\text{context}$,然后去做$\text{word embedding}$。但这种方法问题就在与消耗太大，$\text{NLP}$领域处理一个$14\times 14=196$长度的句子已经算是比较费时的事情，更何况只是$28\times 28$的图片，照现在CV领域处理图片都基本在$224\times 224$往上，明显不行。因此采用折中的方法：

- 将图片先分割成一个个（$\text{Patch}\times \text{Patch}$）的小块，这么我们设$\text{Patch}$为$\text{7}$，那么一个$28\times 28$的图片，就可以被划分成$16$个$7\times 7$的图片了。宽和高各$4$个小块，所以有$16$个，视为$16$个$\text{context}$

- 我们再像$\text{context}$一样构建一个$(\text{Patch}\times \text{embedding\_dim})$的权重矩阵，再让得到的分割矩阵$(16\times 49)$左乘其权重矩阵$(49\times \text{embedding\_dim})$得到一个$(16\times \text{embedded\_dim})$的映射为高维的矩阵，对于$\text{transformer}$来说就变的可以处理了。

### Embedding Patched实现

直接用一个卷积层即可实现第一步，设卷积层的$\text{stride}$和$\text{kernel}$大小为$\text{Patch}$，$\text{out\_channels}$为高维维度数，再将其展平并转置。

```python
import torch.nn as nn
class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)

        x = self.norm(x)
        return x
```

输出的特征图形状是 `[B, embed_dim, H_p, W_p]`

**Flatten 操作**: `x = self.proj(x).flatten(2)`

- 将 `H_p` 和 `W_p` 展开为一个维度，形状从 `[B, embed_dim, H_p, W_p]` 变为 `[B, embed_dim, HW]`，这里的 HW=Hp×WpHW = H_p \times W_pHW=Hp×Wp 表示总的 patch 数量。

**Transpose 操作**: `x.transpose(1, 2)`

- 调整维度的顺序，从 `[B, embed_dim, HW]` 变为 `[B, HW, embed_dim]`。
- 这样，每个 patch 的特征向量（长度为 `embed_dim`）成为序列中的一个元素，序列长度为 `HW`。

`[B, HW, embed_dim]`这种形状是 Transformer 模型的标准输入格式：

- 第零维度 (`B`): Batch size，表示一次处理的样本数。
- 第一维度 (`HW`): 序列长度，表示有多少个 patch。
- 第二维度 (`embed_dim`): 特征维度，表示每个 patch 的嵌入特征。

例如，在 Vision Transformer (ViT) 中，每个 patch 被看作一个 "token"，类似于 NLP 中的单词序列。转置操作确保这些 tokens 的表示形式与 Transformer 的输入格式一致。