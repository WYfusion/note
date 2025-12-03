# SepFormer Transformer 分离网络

## 1. 概述

### 1.1 背景

**SepFormer (Separation Transformer)** 是基于 Transformer 的语音分离模型，由 Subakan 等人于 2021 年提出。它将 DPRNN 中的 RNN 替换为 Transformer，实现了当时最先进的分离性能。

### 1.2 核心创新

- **双路径 Transformer**：结合 DPRNN 的分块策略和 Transformer 的强大建模能力
- **高效注意力**：通过分块处理降低 Transformer 的计算复杂度
- **位置编码**：适应语音信号的时序特性

### 1.3 性能

在 WSJ0-2mix 上达到 **22.3 dB SI-SNRi**，超越所有之前的方法。

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
[SepFormer Blocks × B]
├── Intra-Transformer (块内)
└── Inter-Transformer (块间)
    ↓
[重叠相加 Overlap-Add]
    ↓
掩码 M₁, M₂, ...
    ↓
[解码器 Decoder]
    ↓
分离信号 ŝ₁(t), ŝ₂(t), ...
```

### 2.2 与 DPRNN 的对比

| 组件 | DPRNN | SepFormer |
|------|-------|-----------|
| Intra 处理 | Bi-LSTM | Transformer |
| Inter 处理 | Bi-LSTM | Transformer |
| 位置编码 | 隐式 | 显式 |
| 注意力 | 无 | 多头自注意力 |

---

## 3. Transformer 基础

### 3.1 自注意力机制

**Scaled Dot-Product Attention**：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $Q \in \mathbb{R}^{L \times d_k}$：查询矩阵
- $K \in \mathbb{R}^{L \times d_k}$：键矩阵
- $V \in \mathbb{R}^{L \times d_v}$：值矩阵
- $d_k$：键的维度

### 3.2 多头注意力

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

其中：
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### 3.3 Transformer 编码器层

```
输入 X
    ↓
[Multi-Head Attention]
    ↓
(+ 残差 + LayerNorm)
    ↓
[Feed-Forward Network]
    ↓
(+ 残差 + LayerNorm)
    ↓
输出
```

**Feed-Forward Network**：
$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

---

## 4. SepFormer Block

### 4.1 Intra-Transformer

处理每个块内部的序列（沿 K 维度）：

```python
class IntraTransformer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: [B, N, K, S]
        B, N, K, S = x.shape
        
        # 重排为 [B*S, K, N]
        x = x.permute(0, 3, 2, 1).reshape(B * S, K, N)
        
        # 位置编码
        x = self.pos_enc(x)
        
        # Transformer
        residual = x
        x = self.transformer(x)
        x = self.norm(x + residual)
        
        # 重排回 [B, N, K, S]
        x = x.reshape(B, S, K, N).permute(0, 3, 2, 1)
        
        return x
```

### 4.2 Inter-Transformer

处理跨块的序列（沿 S 维度）：

```python
class InterTransformer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: [B, N, K, S]
        B, N, K, S = x.shape
        
        # 重排为 [B*K, S, N]
        x = x.permute(0, 2, 3, 1).reshape(B * K, S, N)
        
        # 位置编码
        x = self.pos_enc(x)
        
        # Transformer
        residual = x
        x = self.transformer(x)
        x = self.norm(x + residual)
        
        # 重排回 [B, N, K, S]
        x = x.reshape(B, K, S, N).permute(0, 3, 1, 2)
        
        return x
```

### 4.3 完整 SepFormer Block

```python
class SepFormerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, 
                 num_intra_layers, num_inter_layers):
        super().__init__()
        self.intra_transformer = IntraTransformer(
            d_model, nhead, dim_feedforward, num_intra_layers
        )
        self.inter_transformer = InterTransformer(
            d_model, nhead, dim_feedforward, num_inter_layers
        )
        
    def forward(self, x):
        x = self.intra_transformer(x)
        x = self.inter_transformer(x)
        return x
```

---

## 5. 位置编码

### 5.1 正弦位置编码

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # x: [B, L, D]
        return x + self.pe[:, :x.size(1)]
```

### 5.2 可学习位置编码

```python
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

---

## 6. 完整模型实现

### 6.1 编码器和解码器

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

### 6.2 分离网络

```python
class SeparationNet(nn.Module):
    def __init__(self, N, d_model, nhead, dim_feedforward,
                 num_intra_layers, num_inter_layers, num_blocks,
                 K, num_sources):
        super().__init__()
        self.K = K
        self.P = K // 2
        self.num_sources = num_sources
        
        self.norm = nn.LayerNorm(N)
        self.linear = nn.Linear(N, d_model)
        
        self.blocks = nn.ModuleList([
            SepFormerBlock(d_model, nhead, dim_feedforward,
                          num_intra_layers, num_inter_layers)
            for _ in range(num_blocks)
        ])
        
        self.output_linear = nn.Linear(d_model, N * num_sources)
        
    def forward(self, w):
        # w: [B, N, L]
        B, N, L = w.shape
        
        # 分割
        x = segment(w, self.K, self.P)  # [B, N, K, S]
        S = x.shape[-1]
        
        # 线性变换
        x = x.permute(0, 2, 3, 1)  # [B, K, S, N]
        x = self.norm(x)
        x = self.linear(x)  # [B, K, S, d_model]
        x = x.permute(0, 3, 1, 2)  # [B, d_model, K, S]
        
        # SepFormer 块
        for block in self.blocks:
            x = block(x)
        
        # 输出
        x = x.permute(0, 2, 3, 1)  # [B, K, S, d_model]
        x = self.output_linear(x)  # [B, K, S, N*C]
        x = x.view(B, self.K, S, self.num_sources, N)
        x = x.permute(0, 3, 4, 1, 2)  # [B, C, N, K, S]
        
        # 重叠相加
        masks = []
        for c in range(self.num_sources):
            m = overlap_add(x[:, c], self.P)[:, :, :L]
            masks.append(m)
        
        masks = torch.stack(masks, dim=1)  # [B, C, N, L]
        masks = F.relu(masks)
        
        return masks
```

### 6.3 完整 SepFormer

```python
class SepFormer(nn.Module):
    def __init__(self, N=256, L=16, d_model=256, nhead=8,
                 dim_feedforward=1024, num_intra_layers=8,
                 num_inter_layers=8, num_blocks=2, K=250, num_sources=2):
        super().__init__()
        self.encoder = Encoder(N, L)
        self.separator = SeparationNet(
            N, d_model, nhead, dim_feedforward,
            num_intra_layers, num_inter_layers, num_blocks,
            K, num_sources
        )
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

## 7. 计算复杂度分析

### 7.1 标准 Transformer

自注意力复杂度：$O(L^2 \cdot d)$

对于长序列（$L = 16000$），这是不可接受的。

### 7.2 SepFormer 的复杂度

**Intra-Transformer**：
$$O(S \times K^2 \times d) = O\left(\frac{L}{P} \times K^2 \times d\right)$$

**Inter-Transformer**：
$$O(K \times S^2 \times d) = O\left(K \times \frac{L^2}{P^2} \times d\right)$$

当 $K \ll L$ 时，复杂度大大降低。

### 7.3 典型配置

| 参数 | 值 |
|------|-----|
| $L$ | 16000 (1秒@16kHz) |
| $K$ | 250 |
| $P$ | 125 |
| $S$ | 128 |

Intra 复杂度：$O(128 \times 250^2) = O(8M)$
Inter 复杂度：$O(250 \times 128^2) = O(4M)$

相比标准 Transformer 的 $O(256M)$，降低了约 20 倍。

---

## 8. 训练配置

### 8.1 超参数

| 参数 | 典型值 |
|------|--------|
| 编码器维度 $N$ | 256 |
| 编码器核大小 $L$ | 16 |
| Transformer 维度 $d$ | 256 |
| 注意力头数 | 8 |
| FFN 维度 | 1024 |
| Intra 层数 | 8 |
| Inter 层数 | 8 |
| SepFormer 块数 | 2 |
| 块大小 $K$ | 250 |

### 8.2 训练策略

```python
# 模型
model = SepFormer(
    N=256, L=16, d_model=256, nhead=8,
    dim_feedforward=1024, num_intra_layers=8,
    num_inter_layers=8, num_blocks=2, K=250, num_sources=2
)

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-4)

# 学习率调度 - Warmup + Decay
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=10000, num_training_steps=200000
)

# 梯度裁剪
max_grad_norm = 5.0
```

### 8.3 数据增强

```python
def augment(mixture, sources):
    # 速度扰动
    speed_factor = random.uniform(0.95, 1.05)
    mixture = torchaudio.functional.speed(mixture, speed_factor)
    sources = [torchaudio.functional.speed(s, speed_factor) for s in sources]
    
    # 随机增益
    gain = random.uniform(0.8, 1.2)
    mixture = mixture * gain
    sources = [s * gain for s in sources]
    
    return mixture, sources
```

---

## 9. 性能分析

### 9.1 WSJ0-2mix 结果

| 模型 | SI-SNRi (dB) | SDRi (dB) | 参数量 |
|------|--------------|-----------|--------|
| Conv-TasNet | 15.3 | 15.6 | 5.1M |
| DPRNN | 18.8 | 19.0 | 2.6M |
| SepFormer | 20.4 | 20.5 | 26M |
| SepFormer (large) | 22.3 | 22.4 | 26M |

### 9.2 消融实验

| 配置 | SI-SNRi |
|------|---------|
| 完整模型 | 20.4 |
| 无 Inter-Transformer | 17.2 |
| 无 Intra-Transformer | 16.8 |
| 1 个 SepFormer 块 | 19.1 |
| 4 层 Transformer | 19.5 |

### 9.3 计算效率

| 模型 | RTF (GPU) | 内存 |
|------|-----------|------|
| Conv-TasNet | 0.05 | 1GB |
| DPRNN | 0.08 | 1.5GB |
| SepFormer | 0.15 | 4GB |

---

## 10. 变体与扩展

### 10.1 Efficient SepFormer

使用线性注意力降低复杂度：

```python
class LinearAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.nhead = nhead
        self.d_head = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, L, D = x.shape
        
        Q = F.elu(self.q_proj(x)) + 1  # 非负
        K = F.elu(self.k_proj(x)) + 1
        V = self.v_proj(x)
        
        # 线性注意力: O(L*D^2) 而非 O(L^2*D)
        KV = torch.einsum('bld,ble->bde', K, V)
        Z = torch.einsum('bld,bd->bl', K, torch.ones(B, D))
        
        out = torch.einsum('bld,bde->ble', Q, KV) / (
            torch.einsum('bld,bd->bl', Q, Z).unsqueeze(-1) + 1e-6
        )
        
        return self.out_proj(out)
```

### 10.2 Causal SepFormer

用于实时处理：

```python
class CausalSepFormer(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # 使用因果注意力掩码
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        )
```

### 10.3 Multi-resolution SepFormer

使用多尺度处理：

```python
class MultiResSepFormer(nn.Module):
    def __init__(self, K_list=[125, 250, 500]):
        super().__init__()
        self.separators = nn.ModuleList([
            SepFormerSeparator(K=K) for K in K_list
        ])
        self.fusion = nn.Conv1d(len(K_list) * N, N, 1)
```

---

## 11. 总结

### 11.1 SepFormer 的优势

1. **强大的建模能力**：Transformer 的自注意力可以捕获长距离依赖
2. **并行计算**：比 RNN 更容易并行化
3. **灵活性**：可以轻松调整模型大小

### 11.2 局限性

1. **计算量大**：参数量和计算量都较大
2. **内存消耗**：注意力矩阵需要大量内存
3. **实时性**：难以用于实时应用

### 11.3 未来方向

- 更高效的注意力机制
- 流式处理支持
- 多模态融合（视听分离）
