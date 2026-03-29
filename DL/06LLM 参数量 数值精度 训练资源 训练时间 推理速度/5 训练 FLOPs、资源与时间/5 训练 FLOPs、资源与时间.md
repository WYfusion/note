## 概述

训练时间的本质是 $T_{train} = C_{train} / (G cdot F_{peak} cdot eta)$。本页拆解每一项。

---

## 6.1 Dense 模型常用近似

### 训练 FLOPs / token

$$c_{train\_token} \approx 6N$$

- 来源：每个参数在 forward 贡献 $2N$ FLOPs（乘加），backward 贡献 $4N$ FLOPs（梯度对激活 + 梯度对权重各 $2N$）

- 总计：$2N + 4N = 6N$

### 总训练 FLOPs

$$C_{train} \approx 6ND$$

> [!important]
> 
> **速算**：7B 模型训练 1T tokens → $C = 6 \times 7 \times 10^9 \times 10^{12} = 4.2 \times 10^{22}$ FLOPs ≈ $4.2 \times 10^{10}$ TFLOPs。

> [!important]
> 
> 当序列很长时，attention 项 $c_{attn}(S) \approx 4 \cdot S \cdot h \cdot d_{head}$ 不可忽略，应写为 $c_{token} approx 6N + c_{attn}(S)$。

---

## 6.2 MoE 的修正

$$C_{train} \approx D \times (6N_{active} + c_{attn}(S) + c_{router})$$

- $N_{active}$：每 token 激活参数（共享部分 + top-k expert）

- $c_{router}$：路由计算开销（通常远小于 GEMM，可忽略）

> [!important]
> 
> **MoE 关键区分**：训练/推理 FLOPs 看 $N_{active}$，但显存/存储/通信仍受 $N_{total}$ 影响。

---

## 6.3 训练时间公式

$$T_{train} = \frac{C_{train}}{G \cdot F_{peak} \cdot \eta}$$

### 全局训练 token 吞吐

$$\text{tok/s} = \frac{G \cdot F_{peak} \cdot \eta}{6N_{active} + c_{attn}(S)}$$

### 实例估算

|模型|$N_{active}$|$D$|$C_{train}$|集群 (GPU × 峰值)|$\eta$|估算时间|
|---|---|---|---|---|---|---|
|LLaMA-2-7B|7B|2T|$8.4 \times 10^{22}$|2048×A100 (312TF)|~0.40|~13 天|
|LLaMA-2-70B|70B|2T|$8.4 \times 10^{23}$|2048×A100 (312TF)|~0.38|~34 天|
|DeepSeek-V3|37B (MoE)|14.8T|$\sim 3.3 \times 10^{24}$|2048×H800 (990TF FP8)|~0.30|~2 个月|

---

## 6.4 训练资源不只看 GPU 数

> [!important]
> 
> 训练资源 = **算力** + **显存** + **互联** + **存储** + **系统效率**

|资源维度|典型指标|瓶颈场景|
|---|---|---|
|**算力**|$G \times F_{peak}$|模型/数据不够大时 Tensor Core 吃不满|
|**显存**|HBM 容量（80GB/GPU A100）|参数+优化器+激活放不下|
|**互联**|NVLink 900GB/s, IB 400Gb/s|TP/EP all-reduce / all-to-all 通信|
|**存储**|checkpoint I/O, dataset 带宽|大模型 checkpoint 写入慢|
|**系统效率**|kernel 融合, 通信重叠, 容错|MoE 路由不均衡, 异构集群|

---

## 6.5 实际训练变慢的主要来源（$eta$ 杀手）

1. 小 batch / 小矩阵 → Tensor Core 利用率低

1. TP/PP/EP/CP 通信过重 → GPU idle 等待

1. Activation checkpointing → 重算引入额外 FLOPs

1. Dataloader / checkpoint I/O → CPU/存储瓶颈

1. Kernel 不融合 → 多次 HBM 读写

1. 长上下文 attention 访存过重

1. MoE 路由不均衡 → expert 负载偏斜

---

## L3 子页面

- [[1 分布式并行策略：TP / PP / FSDP / ZeRO / CP / EP]] — 各并行策略原理、适用场景、组合方式

- [[2 MFU 与 HFU：系统效率度量]] — MFU/HFU 定义、测量方法、典型值

- [[3 训练效率瓶颈诊断]] — profiling 方法、通信/计算/I/O 瓶颈识别

[[2 MFU 与 HFU：系统效率度量]]

[[1 分布式并行策略：TP - PP - FSDP - ZeRO - CP - EP]]

[[3 训练效率瓶颈诊断]]