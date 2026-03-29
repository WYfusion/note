## 概述

Scaling Law 描述了模型质量（loss）与参数量 $N$、数据量 $D$、计算量 $C$ 之间的幂律关系，是 **"模型该多大、数据该多少、训练该多久"** 的理论指南。

> [!important]
> 
> 核心公式：$Loss approx L_infty + aN^{-alpha} + bD^{-beta} + cC^{-gamma}$
> 
> ——Loss 对参数、数据、计算量近似呈 power law 下降。

---

## 3.1 经典主线

### 泛化表达

$$Loss \approx L_\infty + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

- $L_infty$：不可约损失（数据本身的熵下界）

- $A/N^alpha$：参数不足带来的欠拟合项

- $B/D^beta$：数据不足带来的欠拟合项

给定总计算预算 $C approx 6ND$（Dense），问题转化为在 $C$ 约束下最小化 Loss。

---

## 3.2 两代核心思想

|维度|**Kaplan 系** (OpenAI 2020)|**Chinchilla 系** (Hoffmann et al. 2022)|
|---|---|---|
|核心结论|固定 $C$ 下偏向更大 $N$、较少 $D$、较早停|固定 $C$ 下 $N$ 与 $D$ 应更平衡地共同增长|
|最优比例|$N propto C^{0.73}$，数据相对"省"|$N propto C^{0.50}$，$D propto C^{0.50}$，即 $D \approx 20N$|
|代表模型|GPT-3 (175B, 300B tokens)|Chinchilla (70B, 1.4T tokens)|
|局限|过分偏向大模型，数据利用率低|仅优化训练 loss，不考虑部署推理成本|

---

## 3.3 2024+ 关键修正：推理感知 Scaling

> [!important]
> 
> **核心洞察**：当部署推理需求 $R_{infer}$ 远大于训练计算时，Chinchilla-optimal 不再是 TCO-optimal。

### 总成本模型

$$TCO = C_{train} + R_{infer} \cdot c_{infer}(N)$$

- $C_{train} = 6ND$：训练计算

- $c_{infer}(N) propto N$（每 token 推理成本正比于模型大小）

- $R_{infer}$：部署期总推理量

当 $R_{infer}$ 很大时，最优解偏向 **更小模型 + 更多 token 训练**（over-train），以降低推理期 $N$ 带来的持续成本。

### 实际案例

- **LLaMA-1**：7B 训练了 1T tokens（$D/N approx 143$），远超 Chinchilla 比例 $D/N \approx 20$

- **LLaMA-3**：8B 训练了 15T tokens（$D/N approx 1875$），极端 over-training

- **Mistral/Qwen 系列**：均采用小模型 + 大数据策略

---

## 3.4 实务结论

> [!important]
> 
> 选模型大小不能只问"能不能训"，要同时最优化：
> 
> - **训练损失**（质量下限）
> 
> - **服务成本**（推理期 TCO）
> 
> - **时延目标**（TTFT / TPOT）
> 
> - **总拥有成本**（训练 + 推理 + 运维）

---

## L3 子页面

- [[1 Kaplan vs Chinchilla 详细对比]] — 两代 Scaling Law 的实验设计、拟合参数与结论差异

- [[2 推理感知 Scaling 与 TCO 优化]] — inference-aware scaling、LLaMA 系列 over-training 策略分析

[[1 Kaplan vs Chinchilla 详细对比]]

[[2 推理感知 Scaling 与 TCO 优化]]