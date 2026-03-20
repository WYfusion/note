# Dual-Path RNN 双路径循环网络

## 1. 概述

### 1.1 背景

**Dual-Path RNN (DPRNN)** 是一种处理长序列语音分离的高效架构，由 Luo 等人于 2020 年提出。它解决了 Conv-TasNet 在处理长音频时感受野不足的问题。

### 1.2 核心问题

**长序列挑战**：
- 语音信号可能长达数秒甚至数分钟
- 采样率 16kHz 时，1秒 = 16000 个样本
- 传统 RNN 难以建模如此长的依赖关系
- TCN 需要非常深的网络才能覆盖足够的感受野

### 1.3 解决方案

**双路径策略**：
1. 将长序列分割成短块 (chunks)
2. **块内路径 (Intra-chunk)**：建模块内的局部依赖
3. **块间路径 (Inter-chunk)**：建模块间的全局依赖

---

## 2. 模型架构

### 2.1 整体结构

```
混合信号 x(t)
    ↓
[编码器 Encoder]
    ↓
混合表示 W ∈ R^(N×L)
    ↓
[分割 Segmentation]
    ↓
3D 张量 ∈ R^(N×K×S)
    ↓
[Dual-Path RNN Blocks × B]
├── Intra-chunk RNN (沿 K 维度)
└── Inter-chunk RNN (沿 S 维度)
    ↓
[重叠相加 Overlap-Add]
    ↓
掩码 M₁, M₂, ...
    ↓
[解码器 Decoder]
    ↓
分离信号 ŝ₁(t), ŝ₂(t), ...
```

### 2.2 分割操作

将 1D 序列分割成重叠的块：

**输入**：$\mathbf{W} \in \mathbb{R}^{N \times L}$

**分割**：
- 块大小：$K$
- 跳跃大小：$P = K/2$（50% 重叠）
- 块数量：$S = \lceil (L - K) / P \rceil + 1$

**输出**：$\mathbf{T} \in \mathbb{R}^{N \times K \times S}$

```python
def segment(x, K, P):
    # x: [B, N, L]
    B, N, L = x.shape
    
    # 填充
    pad_len = (K - P) - (L - K) % P
    x = F.pad(x, (0, pad_len))
    
    # 分割
    S = (x.shape[-1] - K) // P + 1
    segments = x.unfold(-1, K, P)  # [B, N, S, K]
    segments = segments.permute(0, 1, 3, 2)  # [B, N, K, S]
    
    return segments
```

### 2.3 重叠相加

将处理后的块重建为序列：

```python
def overlap_add(segments, P):
    # segments: [B, N, K, S]
    B, N, K, S = segments.shape
    
    # 输出长度
    L = P * (S - 1) + K
    output = torch.zeros(B, N, L)
    
    for s in range(S):
        start = s * P
        output[:, :, start:start+K] += segments[:, :, :, s]
    
    return output
```

---

## 3. Dual-Path RNN Block

### 3.1 块结构

```
输入 T ∈ R^(N×K×S)
    ↓
[Intra-chunk RNN]
├── 沿 K 维度处理
├── 每个块独立
└── 建模局部依赖
    ↓
[Inter-chunk RNN]
├── 沿 S 维度处理
├── 每个位置独立
└── 建模全局依赖
    ↓
输出 T' ∈ R^(N×K×S)
```

### 3.2 Intra-chunk RNN

处理每个块内部的序列：

$$\mathbf{T}[:, :, s] \xrightarrow{\text{RNN}} \mathbf{T}'[:, :, s], \quad \forall s$$

```python
class IntraRNN(nn.Module):
    def __init__(self, N, hidden_size):
        super().__init__()
        self.rnn = nn.LSTM(N, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, N)
        self.norm = nn.GroupNorm(1, N)
        
    def forward(self, x):
        # x: [B, N, K, S]
        B, N, K, S = x.shape
        
        # 重排为 [B*S, K, N]
        x = x.permute(0, 3, 2, 1).reshape(B * S, K, N)
        
        # RNN
        residual = x
        x, _ = self.rnn(x)  # [B*S, K, hidden*2]
        x = self.fc(x)  # [B*S, K, N]
        x = x + residual
        
        # 重排回 [B, N, K, S]
        x = x.reshape(B, S, K, N).permute(0, 3, 2, 1)
        x = self.norm(x)
        
        return x
```

### 3.3 Inter-chunk RNN

处理跨块的序列：

$$\mathbf{T}[:, k, :] \xrightarrow{\text{RNN}} \mathbf{T}'[:, k, :], \quad \forall k$$

```python
class InterRNN(nn.Module):
    def __init__(self, N, hidden_size):
        super().__init__()
        self.rnn = nn.LSTM(N, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, N)
        self.norm = nn.GroupNorm(1, N)
        
    def forward(self, x):
        # x: [B, N, K, S]
        B, N, K, S = x.shape
        
        # 重排为 [B*K, S, N]
        x = x.permute(0, 2, 3, 1).reshape(B * K, S, N)
        
        # RNN
        residual = x
        x, _ = self.rnn(x)  # [B*K, S, hidden*2]
        x = self.fc(x)  # [B*K, S, N]
        x = x + residual
        
        # 重排回 [B, N, K, S]
        x = x.reshape(B, K, S, N).permute(0, 3, 1, 2)
        x = self.norm(x)
        
        return x
```

### 3.4 完整 Dual-Path Block

```python
class DualPathBlock(nn.Module):
    def __init__(self, N, hidden_size):
        super().__init__()
        self.intra_rnn = IntraRNN(N, hidden_size)
        self.inter_rnn = InterRNN(N, hidden_size)
        
    def forward(self, x):
        # x: [B, N, K, S]
        x = self.intra_rnn(x)
        x = self.inter_rnn(x)
        return x
```

---

## 4. 数学分析

### 4.1 感受野分析

**Intra-chunk**：
- 每个块内完全连接
- 感受野 = $K$

**Inter-chunk**：
- 跨所有块连接
- 感受野 = $S \times P = L$（整个序列）

**总感受野**：通过交替处理，每个位置可以访问整个序列。

### 4.2 计算复杂度

**Intra-chunk RNN**：
$$O(S \times K \times N \times H)$$

**Inter-chunk RNN**：
$$O(K \times S \times N \times H)$$

**总复杂度**：
$$O(2 \times B \times K \times S \times N \times H) = O(B \times L \times N \times H)$$

与序列长度 $L$ 线性相关！

### 4.3 与其他方法对比

| 方法 | 感受野 | 复杂度 |
|------|--------|--------|
| LSTM | $L$ | $O(L \times N \times H)$ |
| TCN | $O(2^D)$ | $O(D \times L \times N)$ |
| DPRNN | $L$ | $O(L \times N \times H)$ |
| Transformer | $L$ | $O(L^2 \times N)$ |

DPRNN 在保持全局感受野的同时，复杂度与 LSTM 相当。

---

## 5. 完整模型实现

### 5.1 编码器和解码器

```python
class Encoder(nn.Module):
    def __init__(self, N, L):
        super().__init__()
        self.conv = nn.Conv1d(1, N, L, stride=L//2, bias=False)
        
    def forward(self, x):
        return F.relu(self.conv(x.unsqueeze(1)))


class Decoder(nn.Module):
    def __init__(self, N, L):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(N, 1, L, stride=L//2, bias=False)
        
    def forward(self, x):
        return self.deconv(x).squeeze(1)
```

### 5.2 分离网络

```python
class SeparationNet(nn.Module):
    def __init__(self, N, hidden_size, num_blocks, K, num_sources):
        super().__init__()
        self.K = K
        self.P = K // 2
        self.num_sources = num_sources
        
        self.norm = nn.GroupNorm(1, N)
        self.bottleneck = nn.Conv1d(N, N, 1)
        
        self.blocks = nn.ModuleList([
            DualPathBlock(N, hidden_size) for _ in range(num_blocks)
        ])
        
        self.mask_conv = nn.Conv2d(N, N * num_sources, 1)
        
    def forward(self, w):
        # w: [B, N, L]
        B, N, L = w.shape
        
        # 预处理
        x = self.norm(w)
        x = self.bottleneck(x)
        
        # 分割
        x = segment(x, self.K, self.P)  # [B, N, K, S]
        
        # Dual-Path 处理
        for block in self.blocks:
            x = block(x)
        
        # 生成掩码
        x = self.mask_conv(x)  # [B, N*C, K, S]
        x = x.view(B, self.num_sources, N, self.K, -1)  # [B, C, N, K, S]
        
        # 重叠相加
        masks = []
        for c in range(self.num_sources):
            m = overlap_add(x[:, c], self.P)[:, :, :L]  # [B, N, L]
            masks.append(m)
        
        masks = torch.stack(masks, dim=1)  # [B, C, N, L]
        masks = F.relu(masks)
        
        return masks
```

### 5.3 完整 DPRNN

```python
class DPRNN(nn.Module):
    def __init__(self, N=64, L=2, hidden_size=128, num_blocks=6, 
                 K=250, num_sources=2):
        super().__init__()
        self.encoder = Encoder(N, L)
        self.separator = SeparationNet(N, hidden_size, num_blocks, K, num_sources)
        self.decoder = Decoder(N, L)
        self.num_sources = num_sources
        
    def forward(self, x):
        # x: [B, T]
        
        # 编码
        w = self.encoder(x)  # [B, N, L]
        
        # 分离
        masks = self.separator(w)  # [B, C, N, L]
        
        # 解码
        outputs = []
        for c in range(self.num_sources):
            d = masks[:, c] * w
            s = self.decoder(d)
            outputs.append(s)
        
        return torch.stack(outputs, dim=1)  # [B, C, T]
```

---

## 6. 训练配置

### 6.1 超参数

| 参数 | 符号 | 典型值 |
|------|------|--------|
| 编码器维度 | $N$ | 64 |
| 编码器核大小 | $L$ | 2 |
| 隐藏层大小 | $H$ | 128 |
| Dual-Path 块数 | $B$ | 6 |
| 块大小 | $K$ | 250 |
| 源数量 | $C$ | 2 |

### 6.2 训练策略

```python
# 模型
model = DPRNN(N=64, L=2, hidden_size=128, num_blocks=6, K=250, num_sources=2)

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 学习率调度
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=2
)

# 梯度裁剪
max_grad_norm = 5.0
```

### 6.3 动态混合训练

为了提高泛化能力，可以在训练时动态生成混合：

```python
def dynamic_mixing(sources, snr_range=(-5, 5)):
    # 随机 SNR
    snr = torch.rand(1) * (snr_range[1] - snr_range[0]) + snr_range[0]
    scale = 10 ** (snr / 20)
    
    # 缩放第二个源
    sources[1] = sources[1] * scale
    
    # 混合
    mixture = sources.sum(dim=0)
    
    return mixture, sources
```

---

## 7. 性能分析

### 7.1 WSJ0-2mix 结果

| 模型 | SI-SNRi (dB) | 参数量 |
|------|--------------|--------|
| Conv-TasNet | 15.3 | 5.1M |
| DPRNN | 18.8 | 2.6M |
| DPRNN (large) | 19.0 | 3.6M |

### 7.2 长序列性能

| 序列长度 | Conv-TasNet | DPRNN |
|----------|-------------|-------|
| 4s | 15.3 | 18.8 |
| 8s | 14.8 | 18.5 |
| 16s | 13.5 | 18.2 |

DPRNN 在长序列上性能下降更小。

### 7.3 实时性

| 模型 | RTF (CPU) | RTF (GPU) |
|------|-----------|-----------|
| Conv-TasNet | 0.8 | 0.05 |
| DPRNN | 1.2 | 0.08 |

---

## 8. 变体与扩展

### 8.1 Causal DPRNN

用于实时处理：

```python
class CausalInterRNN(nn.Module):
    def __init__(self, N, hidden_size):
        super().__init__()
        # 单向 LSTM
        self.rnn = nn.LSTM(N, hidden_size, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, N)
```

### 8.2 DPRNN-TasNet

结合 Conv-TasNet 的编码器：

```python
class DPRNNTasNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ConvTasNetEncoder(N=256, L=20)
        self.separator = DPRNNSeparator(N=256, ...)
        self.decoder = ConvTasNetDecoder(N=256, L=20)
```

### 8.3 Multi-scale DPRNN

使用多尺度块大小：

```python
class MultiScaleDPRNN(nn.Module):
    def __init__(self, K_list=[125, 250, 500]):
        super().__init__()
        self.separators = nn.ModuleList([
            DPRNNSeparator(K=K) for K in K_list
        ])
```

---

## 9. 与其他方法对比

| 特性 | Conv-TasNet | DPRNN | SepFormer |
|------|-------------|-------|-----------|
| 架构 | TCN | RNN | Transformer |
| 感受野 | 有限 | 全局 | 全局 |
| 长序列 | 差 | 好 | 好 |
| 计算效率 | 高 | 中 | 低 |
| 参数量 | 5M | 2.6M | 26M |
| SI-SNRi | 15.3 | 18.8 | 20.4 |
