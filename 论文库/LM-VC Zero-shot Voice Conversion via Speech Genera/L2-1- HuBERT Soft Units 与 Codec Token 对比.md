## 前置知识

> [!important]
> 
> 阅读本页前建议先读：[[LM-VC- Zero-shot Voice Conversion via Speech Generation based on Language Models]]

---

## 0. 定位

> [!important]
> 
> 本页聚焦 LM-VC 的内容表征选择：HuBERT soft units vs. K-means hard tokens 的 trade-off，以及这一选择对后续工作（Vevo、R-VC）的启示。

---

## 1. 两种表征

|表征|形式|内容保持|音色残留|序列长度|
|---|---|---|---|---|
|**Hard Tokens (K-means)**|离散 ID|⚠️ 中（WER↑）|✅ 少|50 Hz|

---

## 2. LM-VC 的选择与理由

LM-VC 选择 **soft units**，优先保证内容质量，将去音色任务交给 LM 的 prompt conditioning 隐式完成。

> [!important]
> 
> **思辨：这个 trade-off 后来如何被解决？**
> 
> LM-VC 面临的两难：连续 → 保内容但泄音色；离散 → 去音色但损内容。后续工作从不同角度打破了这一 trade-off：
> 
> - **Vevo**：用 VQ-VAE 替代 K-means，端到端优化量化，同样 $K$ 下 WER 低 39% → 离散化损失大幅减少
> 
> - **R-VC**：数据扰动 + K-means + 去重三管齐下 → WER=3.51 且音色泄漏极低，**同时解决了内容和音色问题**
> 
> - **Seed-VC**：不在表征层解决，而是用 Timbre Shifter 从训练策略层消除 mismatch → 连续特征也能低泄漏
> 
> **结论**：LM-VC 的「不得不二选一」在 2024-2025 年已不再成立——新方法要么让离散化损失可忽略（Vevo），要么让连续特征不泄漏（Seed-VC），要么两者兼得（R-VC）。

---

## 延伸阅读

> [!important]
> 
> - 下一页推荐：L2-2 AR/NAR 双阶段 Codec 生成

## 参考文献

- [Wang et al., 2023] LM-VC 原论文 §3.1 Content Representation