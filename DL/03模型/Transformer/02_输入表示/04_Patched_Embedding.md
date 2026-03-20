## 核心想法

- 将连续模态（图像、语音特征或长文本片段）切成固定大小的 patch，再线性映射到 $d_{model}$ 维，得到一串“patch tokens”。
- 优点：降低序列长度、保持局部结构、与标准 Transformer 输入格式对齐。

## 典型实现（Vision Transformer）

1) 划分补丁：输入图像 $X \in \mathbb{R}^{B \times C \times H \times W}$，设 patch 大小 $P$，得到 $\frac{H}{P}\frac{W}{P}$ 个 patch。  
2) 线性映射：用卷积实现投影，核与步长均为 $P$：
```python
proj = nn.Conv2d(in_c, embed_dim, kernel_size=P, stride=P)
tokens = proj(x).flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
```
3) 加位置：为 patch 网格添加可学习或二维展开的位置信息；可再加 `[CLS]`/任务 token。

## 数学形式

- 单个 patch 拉平成向量 $p \in \mathbb{R}^{P^2 C}$，嵌入为 $e = p W + b$，其中 $W \in \mathbb{R}^{P^2 C \times d_{model}}$。
- 使用卷积可视为共享权重的线性映射，参数与上述等价。

## 变体与实践

- 重叠 patch：提高平滑性，代价是序列变长。
- 分辨率变化：可对位置编码插值；或用相对位置（RoPE/ALiBi）减轻外推误差。
- 文本长序列分块：按固定长度截断为块后映射，常用于滑窗/块注意力模型（与局部注意力一起使用）。
