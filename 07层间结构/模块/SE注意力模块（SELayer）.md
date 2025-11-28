SE（Squeeze-and-Excitation）注意力模块由 **SENet**（2018, Hu et al.）提出，用于建模通道间依赖关系。

---

## 一、核心思想

通过**全局信息建模**学习各通道的重要性权重，实现**通道注意力**机制。

**基本组成**：全局平均池化 + 两个全连接层

---

## 二、结构流程

```
输入 X ∈ R^(C×H×W)
      │
      ├─ Squeeze：全局平均池化 → z ∈ R^C
      │
      ├─ Excitation：FC → ReLU → FC → Sigmoid → s ∈ R^C
      │
      └─ Scale：X' = s ⊙ X（通道级缩放）
```

<img src="../../assets/image-20241114195044647.png" alt="SE模块示意图" style="zoom: 50%; display: block; margin: 0 auto;" />

---

## 三、数学形式

### 3.1 Squeeze（压缩）

对每个通道进行全局平均池化，获取通道描述符：

$$z_c = F_{sq}(u_c) = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} u_c(i, j)$$

其中 $u_c \in \mathbb{R}^{H \times W}$ 是第 $c$ 个通道的特征图。

### 3.2 Excitation（激励）

通过两个全连接层学习通道间非线性关系：

$$s = F_{ex}(z, W) = \sigma(W_2 \cdot \delta(W_1 \cdot z))$$

其中：
- $W_1 \in \mathbb{R}^{\frac{C}{r} \times C}$：降维，压缩比 $r$（通常 $r=16$）
- $\delta$：ReLU 激活函数
- $W_2 \in \mathbb{R}^{C \times \frac{C}{r}}$：升维
- $\sigma$：Sigmoid 激活函数

### 3.3 Scale（缩放）

用学到的权重对原特征进行通道级缩放：

$$\tilde{x}_c = F_{scale}(u_c, s_c) = s_c \cdot u_c$$

---

## 四、参数设计

| **参数**       | **标准 SE（SENet）** | **EfficientNet 中的 SE**   |
| ------------ | ---------------- | ------------------------ |
| **压缩比 r**    | 16               | 4                        |
| **FC1 输出维度** | 输入 SE 模块的通道数 / r | 输入 **MBConv 模块**的通道数 / 4 |
| **激活函数 1**   | ReLU             | **Swish**                |
| **激活函数 2**   | Sigmoid          | Sigmoid                  |

> **注意**：EfficientNet 中 FC1 的维度基于 MBConv 模块的**输入**通道数，而非 SE 模块的输入。

---

## 五、计算量分析

设输入通道数为 $C$，压缩比为 $r$：

| **操作**         | **参数量**                   |
| ---------------- | ---------------------------- |
| FC1              | $C \times \frac{C}{r}$       |
| FC2              | $\frac{C}{r} \times C$       |
| **总计**         | $\frac{2C^2}{r}$             |

相对于卷积层参数量较小，计算开销可接受。

---

## 六、PyTorch 实现

```python
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)
```

---

## 七、优势与应用

| **优势**             | **说明**                                       |
| -------------------- | ---------------------------------------------- |
| **轻量级**           | 参数量和计算量增加较少                         |
| **即插即用**         | 可嵌入任意 CNN 架构（ResNet、Inception 等）    |
| **性能提升显著**     | ImageNet 分类 Top-1 错误率降低约 0.5-1%        |
| **通道自适应**       | 自动学习通道重要性，增强有用特征、抑制无用特征 |
