# Conv-TasNet 卷积时域分离网络

## 1. 概述

### 1.1 背景

**Conv-TasNet (Convolutional Time-domain Audio Separation Network)** 是 TasNet 的改进版本，用**时序卷积网络 (TCN)** 替代 LSTM，由 Luo 和 Mesgarani 于 2019 年提出。

### 1.2 主要改进

| 特性 | TasNet | Conv-TasNet |
|------|--------|-------------|
| 分离网络 | LSTM | TCN |
| 计算效率 | 低 | 高 |
| 并行化 | 困难 | 容易 |
| 长序列建模 | 一般 | 好 |
| 参数量 | 大 | 小 |

### 1.3 性能提升

- SI-SNRi: 10 dB → 15+ dB
- 参数量减少 ~3x
- 推理速度提升 ~10x

---

## 2. 模型架构

### 2.1 整体结构

```
混合信号 x(t)
    ↓
[编码器 Encoder] (1-D Conv)
    ↓
混合表示 W ∈ R^(N×K)
    ↓
[分离网络 Separation Network]
├── LayerNorm
├── 1×1 Conv (bottleneck)
├── TCN Blocks × R
│   └── 1-D Conv Blocks × X
└── 1×1 Conv + Softmax (masks)
    ↓
掩码 M₁, M₂, ..., Mₙ
    ↓
源表示 Dᵢ = Mᵢ ⊙ W
    ↓
[解码器 Decoder] (1-D TransConv)
    ↓
分离信号 ŝ₁(t), ŝ₂(t), ...
```

### 2.2 编码器

与 TasNet 相同，使用 1-D 卷积：

$$\mathbf{W} = \text{ReLU}(\text{Conv1D}(\mathbf{x}))$$

参数：
- 滤波器数量：$N$
- 核大小：$L$
- 步长：$L/2$

### 2.3 分离网络 - TCN

**Temporal Convolutional Network** 由多个堆叠的扩张卷积块组成。

**关键组件**：
1. **Bottleneck 层**：降维
2. **TCN 块**：堆叠的扩张卷积
3. **输出层**：生成掩码

### 2.4 解码器

使用转置卷积重建时域信号：

$$\hat{\mathbf{s}}_i = \text{ConvTranspose1D}(\mathbf{M}_i \odot \mathbf{W})$$

---

## 3. TCN 详解

### 3.1 扩张卷积 (Dilated Convolution)

标准卷积的感受野有限，扩张卷积通过在卷积核中插入空洞来扩大感受野。

**扩张因子 $d$**：
$$y[t] = \sum_{k=0}^{K-1} w[k] \cdot x[t - d \cdot k]$$

**感受野**：
$$\text{RF} = 1 + (K-1) \cdot d$$

### 3.2 指数增长的扩张因子

在 Conv-TasNet 中，扩张因子指数增长：
$$d = 2^i, \quad i = 0, 1, 2, \ldots, X-1$$

**总感受野**：
$$\text{RF}_{\text{total}} = 1 + 2(K-1) \sum_{i=0}^{X-1} 2^i = 1 + 2(K-1)(2^X - 1)$$

### 3.3 1-D Conv Block

每个卷积块的结构：

```
输入
  ↓
1×1 Conv (扩展通道)
  ↓
PReLU
  ↓
LayerNorm
  ↓
Depthwise Conv (扩张卷积)
  ↓
PReLU
  ↓
LayerNorm
  ↓
1×1 Conv (压缩通道)
  ↓
(+ 残差连接)
  ↓
输出
```

### 3.4 数学表示

设输入为 $\mathbf{h}^{(l)} \in \mathbb{R}^{B \times K}$，第 $l$ 个块的输出：

$$\mathbf{z}^{(l)} = \text{Conv}_{1\times1}^{\text{expand}}(\mathbf{h}^{(l)}) \in \mathbb{R}^{H \times K}$$
$$\mathbf{z}^{(l)} = \text{PReLU}(\text{LN}(\mathbf{z}^{(l)}))$$
$$\mathbf{z}^{(l)} = \text{DWConv}_{d_l}(\mathbf{z}^{(l)})$$
$$\mathbf{z}^{(l)} = \text{PReLU}(\text{LN}(\mathbf{z}^{(l)}))$$
$$\mathbf{o}^{(l)} = \text{Conv}_{1\times1}^{\text{compress}}(\mathbf{z}^{(l)}) \in \mathbb{R}^{B \times K}$$
$$\mathbf{h}^{(l+1)} = \mathbf{h}^{(l)} + \mathbf{o}^{(l)}$$

---

## 4. 掩码估计

### 4.1 Softmax 掩码

确保掩码和为 1（能量守恒）：

$$\mathbf{M}_i = \frac{\exp(\mathbf{O}_i)}{\sum_{j=1}^{C} \exp(\mathbf{O}_j)}$$

其中 $\mathbf{O}_i$ 是网络输出。

### 4.2 Sigmoid 掩码

允许掩码独立：

$$\mathbf{M}_i = \sigma(\mathbf{O}_i)$$

### 4.3 ReLU 掩码

非负掩码：

$$\mathbf{M}_i = \text{ReLU}(\mathbf{O}_i)$$

---

## 5. 模型配置

### 5.1 超参数

| 参数 | 符号 | 典型值 |
|------|------|--------|
| 编码器滤波器数 | $N$ | 256-512 |
| 编码器核大小 | $L$ | 16-40 |
| Bottleneck 通道数 | $B$ | 128-256 |
| 卷积块通道数 | $H$ | 256-512 |
| 卷积核大小 | $P$ | 3 |
| 每个重复的块数 | $X$ | 8 |
| 重复次数 | $R$ | 3-4 |
| 源数量 | $C$ | 2-3 |

### 5.2 参数量估计

$$\text{Params} \approx N \cdot L + B \cdot N + R \cdot X \cdot (2BH + HP + B) + B \cdot C \cdot N$$

典型配置约 5-8M 参数。

---

## 6. 完整实现

### 6.1 编码器

```python
class Encoder(nn.Module):
    def __init__(self, N, L):
        super().__init__()
        self.conv = nn.Conv1d(1, N, L, stride=L//2, bias=False)
        
    def forward(self, x):
        # x: [B, T]
        x = x.unsqueeze(1)  # [B, 1, T]
        return F.relu(self.conv(x))  # [B, N, K]
```

### 6.2 1-D Conv Block

```python
class ConvBlock(nn.Module):
    def __init__(self, B, H, P, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(B, H, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = nn.GroupNorm(1, H)  # 等价于 LayerNorm
        
        # Depthwise Conv
        padding = (P - 1) * dilation // 2
        self.dwconv = nn.Conv1d(H, H, P, dilation=dilation, 
                                padding=padding, groups=H)
        self.prelu2 = nn.PReLU()
        self.norm2 = nn.GroupNorm(1, H)
        
        self.conv2 = nn.Conv1d(H, B, 1)
        
    def forward(self, x):
        # x: [B, B_ch, K]
        residual = x
        
        x = self.conv1(x)
        x = self.prelu1(self.norm1(x))
        
        x = self.dwconv(x)
        x = self.prelu2(self.norm2(x))
        
        x = self.conv2(x)
        
        return x + residual
```

### 6.3 TCN

```python
class TCN(nn.Module):
    def __init__(self, B, H, P, X, R):
        super().__init__()
        self.blocks = nn.ModuleList()
        
        for r in range(R):
            for x in range(X):
                dilation = 2 ** x
                self.blocks.append(ConvBlock(B, H, P, dilation))
                
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
```

### 6.4 分离网络

```python
class SeparationNet(nn.Module):
    def __init__(self, N, B, H, P, X, R, C):
        super().__init__()
        self.norm = nn.GroupNorm(1, N)
        self.bottleneck = nn.Conv1d(N, B, 1)
        self.tcn = TCN(B, H, P, X, R)
        self.mask_conv = nn.Conv1d(B, N * C, 1)
        self.C = C
        self.N = N
        
    def forward(self, w):
        # w: [B, N, K]
        batch_size = w.shape[0]
        
        x = self.norm(w)
        x = self.bottleneck(x)  # [B, B_ch, K]
        x = self.tcn(x)  # [B, B_ch, K]
        x = self.mask_conv(x)  # [B, N*C, K]
        
        # Reshape to [B, C, N, K]
        x = x.view(batch_size, self.C, self.N, -1)
        
        # Softmax over sources
        masks = F.softmax(x, dim=1)
        
        return masks
```

### 6.5 解码器

```python
class Decoder(nn.Module):
    def __init__(self, N, L):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(N, 1, L, stride=L//2, bias=False)
        
    def forward(self, d):
        # d: [B, N, K]
        return self.deconv(d).squeeze(1)  # [B, T]
```

### 6.6 完整 Conv-TasNet

```python
class ConvTasNet(nn.Module):
    def __init__(self, N=256, L=20, B=256, H=512, P=3, X=8, R=3, C=2):
        super().__init__()
        self.encoder = Encoder(N, L)
        self.separator = SeparationNet(N, B, H, P, X, R, C)
        self.decoder = Decoder(N, L)
        self.C = C
        
    def forward(self, x):
        # x: [B, T]
        
        # 编码
        w = self.encoder(x)  # [B, N, K]
        
        # 分离
        masks = self.separator(w)  # [B, C, N, K]
        
        # 解码
        outputs = []
        for i in range(self.C):
            d = masks[:, i] * w  # [B, N, K]
            s = self.decoder(d)  # [B, T]
            outputs.append(s)
        
        return torch.stack(outputs, dim=1)  # [B, C, T]
```

---

## 7. 训练

### 7.1 损失函数

使用 SI-SNR 损失 + PIT：

```python
def si_snr_loss(estimate, target, eps=1e-8):
    # 零均值
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    
    # 投影
    dot = (estimate * target).sum(dim=-1, keepdim=True)
    s_target = dot * target / (target.pow(2).sum(dim=-1, keepdim=True) + eps)
    
    # 噪声
    e_noise = estimate - s_target
    
    # SI-SNR
    si_snr = 10 * torch.log10(
        s_target.pow(2).sum(dim=-1) / (e_noise.pow(2).sum(dim=-1) + eps) + eps
    )
    
    return -si_snr.mean()
```

### 7.2 训练配置

```python
# 模型
model = ConvTasNet(N=256, L=20, B=256, H=512, P=3, X=8, R=3, C=2)

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 学习率调度
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

# 训练循环
for epoch in range(epochs):
    for batch in dataloader:
        mixture, sources = batch
        
        # 前向传播
        estimates = model(mixture)
        
        # PIT 损失
        loss = pit_si_snr_loss(estimates, sources)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
```

---

## 8. 性能分析

### 8.1 WSJ0-2mix 结果

| 模型 | SI-SNRi (dB) | SDRi (dB) | 参数量 |
|------|--------------|-----------|--------|
| TasNet | 10.8 | 11.1 | 23.6M |
| Conv-TasNet | 15.3 | 15.6 | 5.1M |
| Conv-TasNet (large) | 15.6 | 15.9 | 8.8M |

### 8.2 计算效率

| 模型 | RTF (实时因子) | GPU 内存 |
|------|----------------|----------|
| TasNet | ~0.5 | ~4GB |
| Conv-TasNet | ~0.05 | ~1GB |

### 8.3 消融实验

| 配置 | SI-SNRi |
|------|---------|
| 完整模型 | 15.3 |
| 无残差连接 | 13.8 |
| 无 LayerNorm | 14.2 |
| X=4 (减少块数) | 14.5 |
| R=2 (减少重复) | 14.8 |

---

## 9. 变体与扩展

### 9.1 Causal Conv-TasNet

用于实时处理，使用因果卷积：

```python
# 因果卷积：只使用过去的信息
padding = (P - 1) * dilation
self.dwconv = nn.Conv1d(H, H, P, dilation=dilation, padding=padding)
# 截断未来信息
x = x[:, :, :-padding]
```

### 9.2 Multi-scale Conv-TasNet

使用多尺度编码器：

```python
class MultiScaleEncoder(nn.Module):
    def __init__(self, N, L_list):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(1, N // len(L_list), L, stride=L//2)
            for L in L_list
        ])
```

### 9.3 Attention Conv-TasNet

在 TCN 中加入注意力机制：

```python
class AttentionBlock(nn.Module):
    def __init__(self, B):
        super().__init__()
        self.attention = nn.MultiheadAttention(B, num_heads=8)
```
