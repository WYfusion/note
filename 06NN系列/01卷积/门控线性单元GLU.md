以二维为例，2D-GLU 是在 time×freq 的二维卷积上加入门控（Gated Linear Unit）并常用于下采样（encoder），2D-DeGLU 则是对应的上采样（decoder）模块。门控让网络决定哪些通道信息被“通过”、哪些被“抑制”，从而提高表示能力并稳定训练。
# GLU 的公式
假设某个二维卷积层输出张量$Z\in\mathbb{R}^{B\times 2C\times F'\times T'}$，将通道轴分成两半$Z = [A,\,B]$每半维度为 $C$），则 GLU 的运算为：

$$\text{GLU}(Z) = A \odot \sigma(B)$$

其中 $\sigma$ 是 sigmoid，$\odot$ 表示逐元素乘（broadcast 在 shape 上对齐）。
```python
outputs, gate = x.chunk(2, dim=1)
return outputs * gate.sigmoid()
```
把标准线性变换（outputs）与一个门（gate）结合，门用 sigmoid 将通道信息映射到 (0,1)，对 outputs 做按位缩放。
### GLU 的优点：
- 类似于 LSTM 的门控思想，可以抑制不必要信息，提供容量更大且更易训练的非线性变换；
- 门信号可以学习选择不同频段/时间的通道特征，从而提高表达的稀疏性与鲁棒性。