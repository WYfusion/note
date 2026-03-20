## CTC vs Attention (Seq2Seq)

| 特性 | CTC | Attention (Seq2Seq / Transducer) |
| :--- | :--- | :--- |
| **对齐方式** | **硬性单调**：时间只能向前，不能回头。 | **软对齐**：Attention 矩阵可以关注任意位置（虽然 ASR 中通常也是对角线）。 |
| **依赖性** | **条件独立**：$y_t \perp y_{t-1} | X$。 | **自回归**：$y_t$ 强依赖 $y_{<t}$。 |
| **输出长度** | 必须 $\le$ 输入长度 $T$。 | 可以任意长度（虽然通常也不会超）。 |
| **推理速度** | **极快**：非自回归，支持并行解码。 | **较慢**：通常需要一步步串行生成（除非用 NAT）。 |
| **语言能力** | 弱，严重依赖外部 LM。 | 强，Decoder 本身就是一个语言模型。 |
| **主要问题** | 无法建模输出内部的依赖（如语法）。 | 训练不稳定，对齐难收敛；推理延迟高。 |

## 联合训练 (Hybrid CTC/Attention)

在 Transformer/Conformer ASR 模型中，目前的标准范式是 **Hybrid CTC/Attention**。

![[Structure_Diagram_Placeholder]]
*(此处可想象一个共享 Encoder，顶部有两个头：一个 CTC Head，一个 Attention Decoder)*

1. **共享 Encoder**：利用 Transformer/Conformer 强大的特征提取能力。
2. **训练时**：
   $$L_{total} = \lambda L_{CTC} + (1-\lambda) L_{Attn}$$
   - **CTC 的作用**：强制模型学习单调对齐，加速收敛，防止 Attention“乱看”。
   - **Attention 的作用**：学习语言模型，提升最终识别精度。
3. **推理时**：
   - **方案 A (Rescoring)**：用 CTC 做粗解码（得到 Top-K），用 Attention Decoder 对这 K 个结果重打分。
   - **方案 B (One-pass)**：仅使用 CTC（速度最快）。

## 总结

CTC 是序列对齐领域的基石。尽管纯 CTC 模型因缺乏语言建模能力而在精度上不如 Transducer 或 Attention 模型，但它凭借**极快的推理速度**和**稳定的对齐特性**，依然是现代语音识别系统中不可或缺的组件（常作为辅助 Loss 或用于流式系统的初筛）。
