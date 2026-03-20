**作用**：通过计算时间维度上的均值（Mean）和标准差（Standard Deviation），并将两者拼接为最终特征。
**代码逻辑**：
- **输入**：三维张量 `(batch_size, channels, frames)`，表示批量语音特征（通道数 × 时间帧数）。
- **操作**：
    1. **计算均值**：沿时间轴（`dim=-1`）计算均值，得到形状 `(batch_size, channels)`。
    2. **计算标准差**：沿时间轴计算方差后开平方（添加 `1e-8` 防止数值不稳定），得到形状 `(batch_size, channels)`。
    3. **展平拼接**：将均值和标准差展平后拼接，输出 `(batch_size, 2*channels)`。
```python
    class TemporalStatsPool(nn.Module):
    """TSTP
    Temporal statistics pooling, concatenate mean and std, which is used in x-vector
    Comment: simple concatenation can not make full use of both statistics
    """
    def __init__(self):
        super(TemporalStatsPool, self).__init__()
    def forward(self, x):
        # The last dimension is the temporal axis
        pooling_mean = x.mean(dim=-1)
        pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-8)
        pooling_mean = pooling_mean.flatten(start_dim=1)
        pooling_std = pooling_std.flatten(start_dim=1)
        stats = torch.cat((pooling_mean, pooling_std), 1)
        return stats
```
