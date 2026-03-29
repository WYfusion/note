## 概述

两代 Scaling Law 的核心差异在于：**给定固定算力 $C$，最优的** $N$ **和** $D$ **分配比例是什么？**

---

## Kaplan 系（OpenAI, 2020）

### 核心论文

_Scaling Laws for Neural Language Models_ (Kaplan et al., 2020)

### 实验设置

- 模型范围：768 参数 ~ 1.5B

- 数据：WebText2（~40B tokens）

- 训练策略：**不训到收敛**，较早停止

### 主要发现

$$L(N) \propto N^{-0.076}, \quad L(D) \propto D^{-0.095}, \quad L(C) \propto C^{-0.057}$$

### 最优分配（固定 $C$）

$$N_{opt} \propto C^{0.73}, \quad D_{opt} \propto C^{0.27}$$

> [!important]
> 
> **含义**：算力增加 10x 时，应把大部分预算分给模型（$N$ 增长 $sim 5.4x$），数据只需增长 $sim 1.9x$。

### 问题

- 训练未到收敛 → 低估了数据的边际收益

- 导致 GPT-3（175B）仅用 300B tokens 训练（$D/N approx 1.7$）

---

## Chinchilla 系（DeepMind, 2022）

### 核心论文

_Training Compute-Optimal Large Language Models_ (Hoffmann et al., 2022)

### 实验设置

- 模型范围：70M ~ 16B

- **训练到接近收敛**

- 三种互相验证的方法论

### 主要发现

$$N_{opt} \propto C^{0.50}, \quad D_{opt} \propto C^{0.50}$$

即 $N$ 和 $D$ 应以 **相同速率增长**。

### 经验比例

$$\boxed{D_{opt} \approx 20 N_{opt}}$$

> [!important]
> 
> **含义**：10B 模型的计算最优训练量约为 200B tokens。GPT-3 的 175B/300B tokens 严重"欠训练"——按 Chinchilla 法则应训练 3.5T tokens。

### 验证

Chinchilla（70B, 1.4T tokens）在多数 benchmark 上超越了 Gopher（280B, 300B tokens），尽管参数量仅为其 1/4。

---

## 对比总结

|维度|Kaplan (2020)|Chinchilla (2022)|
|---|---|---|
|$N_{opt}$ 指数|$C^{0.73}$|$C^{0.50}$|
|$D_{opt}$ 指数|$C^{0.27}$|$C^{0.50}$|
|$D/N$ 比例|~1.7（GPT-3 实际）|~20|
|核心偏向|大模型 + 少数据|均衡增长|
|训练范式|提前停止|接近收敛|
|主要局限|低估数据价值|仅优化训练 loss，不考虑推理成本|

---

## Scaling Law 的数学本质

两代法则本质上都在解以下优化问题：

$$\min_{N, D} L(N, D) \quad \text{s.t.} \quad C = 6ND$$

差异在于 $L(N,D)$ 的拟合形式和实验条件不同，导致拉格朗日乘子法给出的最优比例不同：

```Python
import numpy as np
import matplotlib.pyplot as plt

def loss_chinchilla(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, L_inf=1.69):
    """Chinchilla 风格的 loss 模型"""
    return L_inf + A / N**alpha + B / D**beta

def optimal_allocation_chinchilla(C, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    """给定 C=6ND，求最优 N, D"""
    # 解析解 (近似): N ∝ C^(beta/(alpha+beta)), D ∝ C^(alpha/(alpha+beta))
    exp_N = beta / (alpha + beta)  # ≈ 0.45
    exp_D = alpha / (alpha + beta) # ≈ 0.55
    # 这里简化展示比例关系
    ratio = (A * alpha) / (B * beta)  # N/D 的比例因子
    N_opt = (C / 6 * ratio)**(1/2)
    D_opt = C / (6 * N_opt)
    return N_opt, D_opt

# 演示不同预算下的最优分配
for log_C in [21, 22, 23, 24]:
    C = 10**log_C
    N, D = optimal_allocation_chinchilla(C)
    print(f"C=1e{log_C}: N={N/1e9:.1f}B, D={D/1e9:.0f}B tokens, D/N={D/N:.0f}")
```