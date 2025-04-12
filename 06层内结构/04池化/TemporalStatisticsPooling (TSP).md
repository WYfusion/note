**作用**：统计时间维度的均值和方差，提供更丰富的统计特征。  
**来源**：论文《X-vectors: Robust DNN Embeddings for Speaker Recognition》。  
**代码逻辑**：

- 输入：三维张量 `(batch, channels, frames)`。
- 操作：计算时间均值和方差，拼接为 `(batch, channels*2)`。

**特点**：

- **多统计量**：同时捕捉时间分布的集中趋势（均值）和离散程度（方差），增强特征表达能力。
- **固定权重**：对所有时间步平等对待，未区分重要性。
- **适用场景**：需兼顾时间分布位置与变化的场景，如语音说话人识别的基础池化层。
```python
class TemporalStatisticsPooling(nn.Module):
    def __init__(self):
        """TSP
        Link： http://www.danielpovey.com/files/2018_icassp_xvectors.pdf
        """
        super(TemporalStatisticsPooling, self).__init__()  

    def forward(self, x):
        """Computes Temporal Statistics Pooling Module
        Args:
            x (torch.Tensor): Input tensor (#batch, channels, frames).
        Returns:
            torch.Tensor: Output tensor (#batch, channels*2)
        """
        mean = torch.mean(x, dim=2)
        var = torch.var(x, dim=2)
        x = torch.cat((mean, var), dim=1)
        return x
```
