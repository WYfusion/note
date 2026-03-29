# LoRA 系列详解

LoRA（Low-Rank Adaptation）是当前最流行的 PEFT 方法。本节覆盖 LoRA 家族的所有主要变体。

---

## LoRA 核心思想

预训练权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$ 在微调时不动，只学一个**低秩增量** $Delta W = BA$：

$$
W = W_0 + \Delta W = W_0 + BA
$$

其中 $B in mathbb{R}^{d times r}$，$A in mathbb{R}^{r times k}$，$r ll min(d, k)$。

**关键点**：

- 可训练参数量从 $d \times k$ 降到 $r \times (d + k)$
- 推理时可将 $BA$ 合并回 $W_0$，**零额外推理开销**
- 通常作用于注意力层的 $W_Q, W_K, W_V, W_O$ 和/或前馈层

---

## LoRA 家族速览

| 方法 | 在 LoRA 基础上的改进 | 核心优势 |
| --- | --- | --- |
| **LoRA** | 基准方法 | 简单高效，业界标准 |
| **QLoRA** | 4-bit NF4 量化底座 + LoRA | 显存减少 ~75%，单卡微调大模型 |
| **DoRA** | 将权重分解为方向+幅度，LoRA 只更新方向 | 更接近全参数微调的更新模式 |
| **rsLoRA** | 用 $1/\sqrt{r}$ 替代 $\alpha/r$ 做缩放 | 高秩时更稳定 |
| **AdaLoRA** | 自适应分配不同层的秩 | 重要层给高秩，不重要层给低秩 |
| **LoHa** | 用 Hadamard 积做低秩分解 | 更紧凑的参数化 |
| **LoKr** | 用 Kronecker 积做低秩分解 | 更灵活的秩结构 |

---

## 📂 子页面（叶子层，含代码与公式）

- [LoRA 详解与实战](LoRA%20详解与实战.md) — 完整原理推导 + PEFT 库代码 + 权重合并
- [QLoRA 详解与实战](QLoRA%20详解与实战.md) — NF4 量化 + BitsAndBytes + 完整训练代码
- [DoRA 与 rsLoRA 详解](DoRA%20与%20rsLoRA%20详解.md) — 权重分解 + rank-stabilized scaling + AdaLoRA

**相关页面**：[PEFT 参数高效微调方案族](PEFT%20%E5%8F%82%E6%95%B0%E9%AB%98%E6%95%88%E5%BE%AE%E8%B0%83%E6%96%B9%E6%A1%88%E6%97%8F%2007bcd7a7aa894f4984c232d57a0e7376.md) · [Prompt 与 Prefix Tuning 系列](Prompt%20与%20Prefix%20Tuning%20系列.md) · [Adapter 系列与其他 PEFT 方法](Adapter%20系列与其他%20PEFT%20方法.md) · [LLM 微调技术全景指南](LLM%20微调技术全景指南.md)

[DoRA 与 rsLoRA 详解](DoRA%20与%20rsLoRA%20详解.md)

[LoRA 详解与实战](LoRA%20详解与实战.md)

[QLoRA 详解与实战](QLoRA%20详解与实战.md)