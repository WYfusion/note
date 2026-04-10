## 前置知识

> [!important]
> 
> 阅读本页前建议先读：L2-2 U-DiT 架构与 Flow Matching

---

## 0. 定位

> [!important]
> 
> 本页聚焦 Seed-VC 如何从语音 VC 扩展到**歌声转换（Singing Voice Conversion, SVC）**，核心创新为 F0 指数量化条件注入。

---

## 1. SVC 的额外挑战

语音 VC 的方法直接迁移到歌声面临两个问题：

1. **音高（F0）是歌曲的核心信息**：语音中 F0 主要表达语调，可以「随便改」；歌声中 F0 是旋律，必须精确保留

1. **更宽的音域**：歌声 F0 范围远大于语音，需要更强的 F0 表征能力

---

## 2. F0 指数量化

将连续 F0 值离散化为 256 个 bin，使用指数量化适应宽音域：

$$\text{bin} = \text{round}\left(256 \cdot \frac{\log(f_0) - \log(f_{\min})}{\log(f_{\max}) - \log(f_{\min})}\right)$$

> [!important]
> 
> **思辨：为什么用指数量化而非线性量化？**
> 
> 人耳对音高的感知是对数关系（如每个八度频率翻倍）。线性量化在低频段分辨率过高（浪费 bin）、高频段分辨率不足。指数量化使每个 bin 对应的感知差异大致相等，更高效地利用有限的量化级别。

---

## 3. Singing 模型配置

|维度|Speech VC (base)|**Singing VC**|
|---|---|---|
|Mel bins|80|**128**|
|注意力头数|8|**12**|
|F0 条件|无|**指数量化 256 bins**|

---

## 4. 歌声转换实验

|Model|F0CORR ↑|CER ↓|SECS ↑|
|---|---|---|---|
|**Seed-VC (singing)**|0.9375|**19.70**|—|

F0 精度接近专用 SVC 模型 RVCv2，歌词准确度大幅领先。

---

## 延伸阅读

> [!important]
> 
> - 上一页：L2-2 U-DiT 架构
> 
> - 下一页推荐：L2-4 实验与消融分析

## 参考文献

- [Liu, 2024] Seed-VC 原论文 §3.3 Singing Voice Conversion Extension