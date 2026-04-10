## 前置知识

> [!important]
> 
> 阅读本页前建议先读：Conditional Flow Matching (OT-CFM) 基础、[[R-VC- Rhythm Controllable Zero-Shot Voice Conversion via Shortcut Flow Matching]] §3.3

---

## 0. 定位

> [!important]
> 
> 本页聚焦 R-VC 的**核心加速创新**：Shortcut Flow Matching 如何在不引入 teacher 模型或蒸馏 pipeline 的前提下，将 Flow Matching 的采样步数从 10 降至 2，实现 2.83× 加速且性能无损。

---

## 1. 标准 Flow Matching 的瓶颈

标准 OT-CFM 学习的速度场 $v_\theta(x_t, t)$ 在 无效的公式 内是**连续曲线**，但 Euler solver 用有限步离散逼近时会引入截断误差。步数越少，误差越大：

- NFE=32：高质量但慢

- NFE=10：质量可接受

- NFE=2：质量严重下降（标准 FM 下）

核心矛盾：**OT 线性路径虽然比 DDPM 更直，但在 2 步内仍不够直。**

---

## 2. Shortcut FM 核心思想

在标准 FM 的基础上，额外以**步长** $d$ 为条件输入模型。当 $d$ 小时，模型退化为标准 FM；当 $d$ 大时，模型学会**预判曲率并「跳跃」到正确位置**。

直觉类比：

- 标准 FM = 沿弯道开车，需要小步频繁转弯

- Shortcut FM = 会预判弯道的老司机，大步走也能精准到达

---

## 3. 数学公式

### 3.1 训练损失

KaTeX parse error: Undefined control sequence: \[ at position 37: …race{\mathbb{E}\̲[̲\|s_\theta(x_t,…

其中自洽目标 $s_{\text{target}}$ 由两小步的平均定义：

KaTeX parse error: Undefined control sequence: \[ at position 32: …} = \frac{1}{2}\̲[̲s_\theta(x_t, t…

$x'_{t+d} = x_t + d \cdot s_\theta(x_t, t, d)$ 是一小步后的中间状态。

|符号|含义|维度/范围|$x_t$|时刻 $t$ 的中间状态|与 Mel 维度相同|
|---|---|---|---|---|---|
|$t$|流匹配时间|无效的公式|$d$|步长条件|$\{0, 1/N, 2/N, ..., 1/2\}$|
|$s_\theta$|速度场预测|与 $x_t$ 同维|$\lambda$|自洽损失权重|0.3（论文设置）|

### 3.2 直觉解释

自洽损失的含义：**走 2d 一大步的预测 ≈ 走 d 两小步的平均**。这迫使模型在大步长时学会「看远路」——预判两步后的位置并直接跳过去。

---

## 4. 训练策略

- **混合训练**：每个 batch 中 70% 样本用标准 FM 损失（$d=0$），30% 用自洽损失（$d>0$）

- **步长采样**：$d$ 从 $\{0, 1/N, 2/N, ..., 1/2\}$ 均匀采样，$N$ 为目标步数

- **自洽目标需 2 次前向传播**：先算 $s_theta(x_t, t, d)$，再算 $s_\theta(x'_{t+d}, t+d, d)$

- **Stop-gradient**：自洽目标中的两步预测 detach 梯度（不通过 target 反传）

> [!important]
> 
> **思辨：Shortcut FM vs. Consistency Models vs. Progressive Distillation**
> 
> |方法|需要 Teacher？|额外训练成本|最少步数|质量|
> |---|---|---|---|---|
> |Consistency Models|✅ (EMA teacher)|teacher 训练 + 蒸馏|1-2|≈ 多步 Diffusion|
> |Rectified Flow|❌|需 reflow 迭代|1-2|需多次 reflow|
> 
> Shortcut FM 的核心优势：**训练 pipeline 最简单**——不需要预训练 teacher，不需要多阶段蒸馏，只需在标准 FM 训练中加入 30% 的自洽样本。代价是每个 batch 中 30% 样本需两次前向传播，但总成本远低于蒸馏方案。

---

## 5. 消融实验

|NFE|SECS ↑|WER ↓|UTMOS ↑|RTF ↓|
|---|---|---|---|---|
|4|0.930|3.52|3.91|0.18|
|1|0.921|3.89|3.76|0.07|

**关键结论**：NFE=2 与 NFE=10 在所有指标上几乎无差异，但速度快 2.83×。NFE=1 出现明显质量下降。

---

## 延伸阅读

> [!important]
> 
> - 上一页：L2-2 Mask Transformer Duration Model
> 
> - 下一页推荐：L2-4 DiT 解码器架构

## 参考文献

- [Frans et al., 2024] "One Step Diffusion via Shortcut Models" — Shortcut FM 原始论文

- [Lipman et al., 2022] "Flow Matching for Generative Modeling" — FM 理论基础

- [Song et al., 2023] "Consistency Models" — 对比方法

- [Salimans & Ho, 2022] "Progressive Distillation for Fast Sampling" — 对比方法