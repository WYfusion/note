**作用**：时间维度上的平均池化，提取全局时间特征。  
**来源**：论文《Multi-Task Learning with High-Order Statistics for X-vector based Text-Independent Speaker Verification》。  
**代码逻辑**：

- 输入：三维张量 `(batch, channels, frames)`。
- 操作：沿时间轴（`dim=2`）计算均值，输出 `(batch, channels)`。
- 展平：兼容二维输入。

**特点**：

- **简单高效**：仅计算时间维度上的平均值，计算复杂度低。
- **全局表征**：捕捉整体时间序列的静态特征，但忽略局部动态变化。
- **适用场景**：需快速提取全局特征的场景，如轻量级模型或辅助任务。
```python
class TemporalAveragePooling(nn.Module):
    def __init__(self):
        """TAP
        Link: https://arxiv.org/pdf/1903.12058.pdf
        """
        super(TemporalAveragePooling, self).__init__()
        
    def forward(self, x):
        """Computes Temporal Average Pooling Module
        Args:
            x (torch.Tensor): Input tensor (#batch, channels, frames).
        Returns:
            torch.Tensor: Output tensor (#batch, channels)
        """
        x = torch.mean(x, dim=2)
        # To be compatable with 2D input
        x = x.flatten(start_dim=1)
        return x
```
