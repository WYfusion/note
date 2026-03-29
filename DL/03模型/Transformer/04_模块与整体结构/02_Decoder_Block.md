## 结构 (Pre-LN)

1) $H_1 = X + \text{MHA}_{\text{masked}}(\text{LN}(X))$  （因果掩码）
2) $H_2 = H_1 + \text{CrossAttn}(\text{LN}(H_1),\ K^{enc}, V^{enc})$
3) $H_3 = H_2 + \text{FFN}(\text{LN}(H_2))$

- CrossAttn 的 $K,V$ 来自 Encoder，$Q$ 来自当前层输入。
- 仍保留残差与 LayerNorm；可用门控残差 (Gated Residual) 细化信息流。

## 解码缓存

- 自回归推理：缓存历史 $K,V$，增量更新 $Q$，避免重复计算；Cross Attention 的 $K,V$ 仅在编码器侧计算一次。

## 变体

- 共享或轻量 CrossAttn：降低解码开销，可结合多查询/GQA。
- 门控 FFN 或 Mixture-of-Experts：提升容量并减少单例计算。

