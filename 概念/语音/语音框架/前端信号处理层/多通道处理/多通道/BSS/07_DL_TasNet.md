# TasNet 时域音频分离网络

## 1. 概述

### 1.1 背景

**TasNet (Time-domain Audio Separation Network)** 是第一个完全在时域进行端到端语音分离的深度学习模型，由 Luo 和 Mesgarani 于 2018 年提出。

### 1.2 创新点

- **时域处理**：直接处理波形，避免 STFT 的相位问题
- **可学习编码器**：用神经网络替代 STFT
- **端到端训练**：从混合信号直接到分离信号

### 1.3 与传统方法对比

| 特性 | 传统 BSS | TasNet |
|------|----------|--------|
| 域 | 频域 | 时域 |
| 特征提取 | STFT (固定) | 可学习编码器 |
| 相位处理 | 困难 | 隐式处理 |
| 训练方式 | 无监督 | 有监督 |

---

## 2. 模型架构

### 2.1 整体结构

```
混合信号 x(t)
    ↓
[编码器 Encoder]
    ↓
混合表示 W
    ↓
[分离网络 Separation Network]
    ↓
掩码 M₁, M₂, ..., Mₙ
    ↓
源表示 D₁ = M₁ ⊙ W, ...
    ↓
[解码器 Decoder]
    ↓
分离信号 ŝ₁(t), ŝ₂(t), ...
```

### 2.2 编码器 (Encoder)

将时域信号转换为高维表示：

$$\mathbf{W} = \text{ReLU}(\mathbf{U} * \mathbf{x})$$

其中：
- $\mathbf{x} \in \mathbb{R}^{1 \times L}$：输入信号
- $\mathbf{U} \in \mathbb{R}^{N \times L_w}$：编码器滤波器组
- $*$：卷积操作
- $N$：编码维度
- $L_w$：窗口长度

**类比 STFT**：
- $\mathbf{U}$ 类似于 STFT 的基函数
- $\mathbf{W}$ 类似于频谱表示
- 但 $\mathbf{U}$ 是可学习的

### 2.3 分离网络 (Separation Network)

估计每个源的掩码：

$$\mathbf{M}_i = f_\theta(\mathbf{W}), \quad i = 1, \ldots, C$$

其中 $C$ 是源的数量。

**原始 TasNet 使用 LSTM**：
```
W → LayerNorm → LSTM → FC → Sigmoid → M
```

### 2.4 解码器 (Decoder)

将分离后的表示转换回时域：

$$\hat{\mathbf{s}}_i = \mathbf{V}^T (\mathbf{M}_i \odot \mathbf{W})$$

其中：
- $\mathbf{V} \in \mathbb{R}^{N \times L_w}$：解码器滤波器组
- $\odot$：逐元素乘法

---

## 3. 数学推导

### 3.1 编码过程

输入信号 $\mathbf{x} \in \mathbb{R}^T$ 被分割成重叠的帧：

$$\mathbf{x}_k = \mathbf{x}[k \cdot S : k \cdot S + L_w], \quad k = 0, 1, \ldots, K-1$$

其中 $S$ 是步长，$K = \lfloor (T - L_w) / S \rfloor + 1$。

编码：
$$\mathbf{w}_k = \text{ReLU}(\mathbf{U} \mathbf{x}_k) \in \mathbb{R}^N$$

堆叠得到：
$$\mathbf{W} = [\mathbf{w}_0, \mathbf{w}_1, \ldots, \mathbf{w}_{K-1}] \in \mathbb{R}^{N \times K}$$

### 3.2 掩码估计

分离网络输出 $C$ 个掩码：
$$\mathbf{M}_i \in [0, 1]^{N \times K}, \quad i = 1, \ldots, C$$

使用 Sigmoid 激活确保掩码在 [0, 1] 范围内。

### 3.3 解码过程

分离后的表示：
$$\mathbf{D}_i = \mathbf{M}_i \odot \mathbf{W} \in \mathbb{R}^{N \times K}$$

解码每一帧：
$$\hat{\mathbf{s}}_{i,k} = \mathbf{V}^T \mathbf{d}_{i,k} \in \mathbb{R}^{L_w}$$

重叠相加重建：
$$\hat{s}_i[n] = \sum_k \hat{\mathbf{s}}_{i,k}[n - k \cdot S] \cdot w[n - k \cdot S]$$

其中 $w[\cdot]$ 是窗函数。

### 3.4 编码器-解码器约束

为了完美重建，需要：
$$\mathbf{V}^T \mathbf{U} = \mathbf{I}$$

或使用重叠相加条件。

---

## 4. 损失函数

### 4.1 尺度不变信噪比 (SI-SNR)

$$\text{SI-SNR}(\hat{s}, s) = 10 \log_{10} \frac{\|s_{\text{target}}\|^2}{\|e_{\text{noise}}\|^2}$$

其中：
$$s_{\text{target}} = \frac{\langle \hat{s}, s \rangle}{\|s\|^2} s$$
$$e_{\text{noise}} = \hat{s} - s_{\text{target}}$$

### 4.2 排列不变训练 (PIT)

由于源的顺序未知，使用 PIT 损失：

$$\mathcal{L}_{\text{PIT}} = \min_{\pi \in \mathcal{P}} \sum_{i=1}^{C} \mathcal{L}(\hat{s}_i, s_{\pi(i)})$$

其中 $\mathcal{P}$ 是所有排列的集合。

### 4.3 完整损失

$$\mathcal{L} = -\frac{1}{C} \max_{\pi \in \mathcal{P}} \sum_{i=1}^{C} \text{SI-SNR}(\hat{s}_i, s_{\pi(i)})$$

---

## 5. 训练细节

### 5.1 数据准备

**混合信号生成**：
$$x = \sum_{i=1}^{C} s_i$$

**数据增强**：
- 随机缩放
- 随机混合比例
- 添加噪声

### 5.2 超参数

| 参数 | 典型值 |
|------|--------|
| 编码维度 $N$ | 256-512 |
| 窗口长度 $L_w$ | 2-40 ms |
| 步长 $S$ | $L_w / 2$ |
| LSTM 层数 | 4 |
| LSTM 隐藏维度 | 500-1000 |

### 5.3 训练策略

- 优化器：Adam
- 学习率：$10^{-3}$，带衰减
- 批大小：根据 GPU 内存
- 梯度裁剪：防止梯度爆炸

---

## 6. 模型实现

### 6.1 编码器

```python
class Encoder(nn.Module):
    def __init__(self, N, L):
        super().__init__()
        # N: 编码维度, L: 窗口长度
        self.conv = nn.Conv1d(1, N, L, stride=L//2, bias=False)
        
    def forward(self, x):
        # x: [B, T]
        x = x.unsqueeze(1)  # [B, 1, T]
        w = F.relu(self.conv(x))  # [B, N, K]
        return w
```

### 6.2 分离网络 (LSTM)

```python
class SeparationNet(nn.Module):
    def __init__(self, N, hidden_size, num_layers, num_sources):
        super().__init__()
        self.norm = nn.LayerNorm(N)
        self.lstm = nn.LSTM(N, hidden_size, num_layers, 
                           batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, N * num_sources)
        self.num_sources = num_sources
        self.N = N
        
    def forward(self, w):
        # w: [B, N, K]
        B, N, K = w.shape
        
        # LayerNorm
        w = w.transpose(1, 2)  # [B, K, N]
        w = self.norm(w)
        
        # LSTM
        h, _ = self.lstm(w)  # [B, K, hidden*2]
        
        # FC + Sigmoid
        m = torch.sigmoid(self.fc(h))  # [B, K, N*C]
        m = m.view(B, K, self.num_sources, N)  # [B, K, C, N]
        m = m.permute(0, 2, 3, 1)  # [B, C, N, K]
        
        return m
```

### 6.3 解码器

```python
class Decoder(nn.Module):
    def __init__(self, N, L):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(N, 1, L, stride=L//2, bias=False)
        
    def forward(self, d):
        # d: [B, N, K]
        s = self.deconv(d)  # [B, 1, T]
        return s.squeeze(1)  # [B, T]
```

### 6.4 完整模型

```python
class TasNet(nn.Module):
    def __init__(self, N=256, L=40, hidden_size=500, 
                 num_layers=4, num_sources=2):
        super().__init__()
        self.encoder = Encoder(N, L)
        self.separator = SeparationNet(N, hidden_size, 
                                       num_layers, num_sources)
        self.decoder = Decoder(N, L)
        self.num_sources = num_sources
        
    def forward(self, x):
        # x: [B, T] 混合信号
        
        # 编码
        w = self.encoder(x)  # [B, N, K]
        
        # 分离
        masks = self.separator(w)  # [B, C, N, K]
        
        # 解码
        outputs = []
        for i in range(self.num_sources):
            d = masks[:, i] * w  # [B, N, K]
            s = self.decoder(d)  # [B, T]
            outputs.append(s)
        
        return torch.stack(outputs, dim=1)  # [B, C, T]
```

---

## 7. SI-SNR 损失实现

```python
def si_snr(estimate, target, eps=1e-8):
    """
    计算 SI-SNR
    
    参数:
        estimate: 估计信号 [B, T]
        target: 目标信号 [B, T]
    
    返回:
        si_snr: [B]
    """
    # 零均值
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    
    # s_target = <s', s> / ||s||^2 * s
    dot = (estimate * target).sum(dim=-1, keepdim=True)
    s_target_energy = (target ** 2).sum(dim=-1, keepdim=True) + eps
    proj = dot * target / s_target_energy
    
    # e_noise = s' - s_target
    noise = estimate - proj
    
    # SI-SNR
    si_snr = 10 * torch.log10(
        (proj ** 2).sum(dim=-1) / ((noise ** 2).sum(dim=-1) + eps) + eps
    )
    
    return si_snr


def pit_loss(estimates, targets):
    """
    排列不变训练损失
    
    参数:
        estimates: [B, C, T]
        targets: [B, C, T]
    
    返回:
        loss: 标量
    """
    from itertools import permutations
    
    B, C, T = estimates.shape
    
    # 计算所有排列的损失
    perms = list(permutations(range(C)))
    losses = []
    
    for perm in perms:
        loss = 0
        for i, j in enumerate(perm):
            loss -= si_snr(estimates[:, i], targets[:, j])
        losses.append(loss / C)
    
    # 选择最小损失
    losses = torch.stack(losses, dim=1)  # [B, num_perms]
    min_loss = losses.min(dim=1)[0]  # [B]
    
    return min_loss.mean()
```

---

## 8. 性能与局限

### 8.1 性能

在 WSJ0-2mix 数据集上：
- SI-SNRi: ~10 dB
- SDRi: ~10 dB

### 8.2 局限性

1. **计算量大**：LSTM 序列处理慢
2. **长序列困难**：LSTM 难以建模长距离依赖
3. **实时性差**：双向 LSTM 需要完整序列

### 8.3 后续改进

- **Conv-TasNet**：用 TCN 替代 LSTM
- **Dual-Path RNN**：处理长序列
- **SepFormer**：使用 Transformer
