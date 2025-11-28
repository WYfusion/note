# 深度学习RNN追踪

## 1. 概述

**循环神经网络 (RNN)** 可以学习复杂的时序依赖关系，用于说话人追踪。

---

## 2. RNN架构

### 2.1 基本RNN

$$\mathbf{h}_t = \tanh(\mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{W}_x \mathbf{x}_t + \mathbf{b})$$

$$\mathbf{y}_t = \mathbf{W}_y \mathbf{h}_t + \mathbf{b}_y$$

### 2.2 LSTM

**长短期记忆网络**：

$$\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$$
$$\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$$
$$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$$
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$
$$\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$$
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

---

## 3. 追踪网络设计

### 3.1 输入特征

- 当前观测：DOA、TDOA
- 历史轨迹
- 音频特征

### 3.2 输出

- 位置预测
- 速度预测
- 不确定性估计

### 3.3 实现

```python
import torch
import torch.nn as nn

class LSTMTracker(nn.Module):
    """LSTM说话人追踪器"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=False
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, hidden=None):
        """
        参数:
            x: [batch, seq_len, input_dim]
            hidden: LSTM隐状态
        
        返回:
            output: [batch, seq_len, output_dim]
        """
        lstm_out, hidden = self.lstm(x, hidden)
        output = self.fc(lstm_out)
        return output, hidden


# 使用示例
model = LSTMTracker(input_dim=4, hidden_dim=64, output_dim=2)
x = torch.randn(32, 10, 4)  # [batch, seq, features]
output, _ = model(x)
print(f"输出形状: {output.shape}")  # [32, 10, 2]
```

---

## 4. 训练策略

### 4.1 损失函数

**MSE损失**：
$$\mathcal{L} = \frac{1}{T}\sum_{t=1}^{T} \|\mathbf{x}_t - \hat{\mathbf{x}}_t\|^2$$

**Huber损失**（鲁棒）：
$$\mathcal{L}_{\delta}(a) = \begin{cases} \frac{1}{2}a^2 & |a| \leq \delta \\ \delta(|a| - \frac{1}{2}\delta) & |a| > \delta \end{cases}$$

### 4.2 数据增强

- 轨迹扰动
- 观测噪声
- 缺失数据

---

## 5. 优缺点

**优点**：
- 学习复杂模式
- 端到端训练
- 自适应性强

**缺点**：
- 需要大量数据
- 计算量大
- 可解释性差

---

## 参考文献

1. Hochreiter, S., & Schmidhuber, J. (1997). "Long short-term memory." Neural Computation.

2. Graves, A. (2013). "Generating sequences with recurrent neural networks." arXiv.
