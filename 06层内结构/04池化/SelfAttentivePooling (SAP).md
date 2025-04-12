**作用**：通过自注意力机制动态加权时间步，突出关键信息。 
**代码逻辑**：

- 结构：两层 `Conv1d`（等效全连接层）生成注意力权重。
    - 第一层（`bottleneck_dim`）压缩维度，第二层恢复原维度。
    - 使用 `tanh` 和 `softmax` 生成归一化权重。
- 输出：加权平均后的特征向量 `(batch, channels)`。
**特点**：
- **动态聚焦**：自动学习不同时间步的重要性，提升对关键片段的敏感性。
- **无全局上下文**：仅依赖局部时间步信息分配权重。
- **适用场景**：需自适应关注重要时间段的语音任务（如情感识别、关键词检测）。
``` python
class SelfAttentivePooling(nn.Module):
    """SAP"""
    def __init__(self, in_dim, bottleneck_dim=128):
        # Use Conv1d with stride == 1 rather than Linear, then we don't need to transpose inputs.
        # attention dim = 128
        super(SelfAttentivePooling, self).__init__()
        self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1)  # equals W and b in the paper
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)  # equals V and k in the paper
    def forward(self, x):
        # DON'T use ReLU here! In experiments, I find ReLU hard to converge.
        alpha = torch.tanh(self.linear1(x))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        return mean
```