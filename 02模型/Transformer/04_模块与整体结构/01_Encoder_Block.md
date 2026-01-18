## 结构 (Pre-LN 常用)

1) $H_1 = X + \text{MHA}(\text{LN}(X))$
2) $H_2 = H_1 + \text{FFN}(\text{LN}(H_1))$

- FFN：$\text{FFN}(x) = \sigma(xW_1 + b_1) W_2 + b_2$，常用 $\text{GELU}$，中间维度 $d_{ff} \approx 4d_{model}$。
- 残差保持梯度流，LayerNorm 稳定训练；Post-LN（Norm 放输出端）在浅层可行，但深层易不稳。

## 功能

- 将全局上下文编码为一组上下文相关表示，供下游任务或 Decoder 使用。
- 对称堆叠 $N$ 层，输出形状保持 $(L, d_{model})$。

## 实践提示

- Dropout：注意力权重、FFN 输出、残差旁路均可插入 dropout。
- 正则：可加入 Attention Dropout + LayerDrop；深层时可用 RMSNorm 代替 LayerNorm。
