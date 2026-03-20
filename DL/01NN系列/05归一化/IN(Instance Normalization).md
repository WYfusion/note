实例归一化
$$y=\frac{x-\mathrm{E}[x]}{\sqrt{\mathrm{Var}[x]+\epsilon}}\cdot\gamma+\beta$$
其中：

- $x$ 是输入
- $E[x]$ 是每个样本每个通道的均值
- $Var[x]$ 是每个样本每个通道的方差
- $ε$ 是为了数值稳定性添加的小常数
- $γ$ 和 $β$ 是可学习的缩放和偏移参数（当 affine=True 时）

**作用范围**：单独每一个样本、各个通道在空间维度(如高度和宽度)上归一化，每个通道有自己的均值和方差，保持通道间的相对关系，消除样本内单个通道的分布差异。

### 实例归一化:
- 主要用于计算机视觉任务
- 特别适合风格迁移(cyclegan)，因为它移除了每个实例的风格信息
- 在生成模型中很常用，如GAN和风格迁移

实例归一化和[[LN(LayerNormalization)|层归一化]]很像。

```python
import torch
import torch.nn as nn
# 创建一个简单的输入数据
batch_size = 2
channels = 3
height = 4
width = 4
x = torch.randn(batch_size, channels, height, width)

# 层归一化 - 需要指定特征维度
layer_norm = nn.LayerNorm([channels, height, width])
y_ln = layer_norm(x)
# 实例归一化 - 只需要指定通道数
instance_norm = nn.InstanceNorm2d(channels, affine=True)
y_in = instance_norm(x)

# 验证层归一化的结果
for i in range(batch_size):
    # 层归一化：每个样本应该有均值接近0，方差接近1
    mean_ln = y_ln[i].mean().item()
    var_ln = y_ln[i].var().item()
    print(f"样本 {i} 层归一化后：均值 = {mean_ln:.6f}, 方差 = {var_ln:.6f}")
    # 实例归一化：每个样本的每个通道应该有均值接近0，方差接近1
    for j in range(channels):
        mean_in = y_in[i, j].mean().item()
        var_in = y_in[i, j].var().item()
        print(f"样本 {i}, 通道 {j} 实例归一化后：均值 = {mean_in:.6f}, 方差 = {var_in:.6f}")

样本 0 层归一化后：均值 = 0.000000, 方差 = 1.021266
样本 0, 通道 0 实例归一化后：均值 = -0.000000, 方差 = 1.066658
样本 0, 通道 1 实例归一化后：均值 = 0.000000, 方差 = 1.066655
样本 0, 通道 2 实例归一化后：均值 = 0.000000, 方差 = 1.066649
样本 1 层归一化后：均值 = -0.000000, 方差 = 1.021266
样本 1, 通道 0 实例归一化后：均值 = -0.000000, 方差 = 1.066656
样本 1, 通道 1 实例归一化后：均值 = -0.000000, 方差 = 1.066654
样本 1, 通道 2 实例归一化后：均值 = 0.000000, 方差 = 1.066656
```