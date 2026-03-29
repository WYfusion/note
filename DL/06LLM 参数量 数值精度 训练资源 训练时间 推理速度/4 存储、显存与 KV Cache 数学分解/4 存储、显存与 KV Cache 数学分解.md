## 概述

模型的"大小"不只是参数量——训练时显存包含 7 项主要成分，推理时 KV cache 往往比权重更先炸显存。本页给出各项的数学公式。

---

## 5.1 纯权重体积

$$M_{weight} \approx N \times \frac{b_w}{8} \text{ (bytes)}$$

|参数量|FP32 (4B/param)|BF16/FP16 (2B)|FP8/INT8 (1B)|4-bit (0.5B)|
|---|---|---|---|---|
|1B|~4 GB|~2 GB|~1 GB|~0.5 GB|
|7B|~28 GB|~14 GB|~7 GB|~3.5 GB|
|70B|~280 GB|~140 GB|~70 GB|~35 GB|
|671B (MoE)|~2.7 TB|~1.3 TB|~671 GB|~335 GB|

> [!important]
> 
> 不含 scale / metadata / 内存对齐开销。实际量化模型还需加 group scale 存储（通常 +5%~15%）。

---

## 5.2 训练显存总式

$$M_{train} \approx M_{weight} + M_{master} + M_{grad} + M_{opt} + M_{act} + M_{temp} + M_{frag}$$

|项目|含义|典型大小 (BF16 + AdamW)|
|---|---|---|
|$M_{weight}$|模型权重 (BF16)|$2N$ bytes|
|$M_{master}$|FP32 主权重副本|$4N$ bytes|
|$M_{grad}$|梯度 (BF16)|$2N$ bytes|
|$M_{opt}$|Adam $m,v$ (FP32)|$8N$ bytes|
|$M_{act}$|中间激活|$\propto B \times S \times d \times L$|
|$M_{temp}$|kernel workspace / all-gather buffer|变化大|
|$M_{frag}$|碎片与框架开销|~10-20% overhead|

> [!important]
> 
> **常用速算**：BF16 混合精度 + AdamW，不含激活时，参数相关显存 ≈ $16N$ bytes（即 $2+4+2+8=16$ bytes/param）。对 7B 模型约 112 GB，仅此一项就超出单张 80GB GPU。

---

## 5.3 训练显存主结论

- **小 batch / 短序列** → 参数 + 优化器状态主导

- **大 batch / 长序列** → 激活显存 $M_{act}$ 迅速成为主项

- **长上下文训练** → attention 显存 $propto B times S^2$（无 FlashAttention 时），需 FlashAttention / CP / checkpointing 配合

---

## 5.4 推理 KV Cache

$$M_{KV} \approx B \times T \times L \times 2 \times h_{kv} \times d_{head} \times \frac{b_{kv}}{8}$$

- $B$：并发请求数

- $T$：当前上下文长度

- $L$：层数

- $2$：K 和 V 各一份

- $h_{kv}$：KV head 数（GQA/MQA 直接降低此项）

- $d_{head}$：每头维度

- $b_{kv}$：KV 精度 bit 数

### 示例：LLaMA-2-70B, BF16, T=4096, B=32

$M_{KV} = 32 \times 4096 \times 80 \times 2 \times 8 \times 128 \times \frac{16}{8} \approx 42.9\ \text{GB}$

> [!important]
> 
> **长上下文 / 大并发时，KV cache 往往比权重更先炸显存。** 这就是 GQA/MQA、KV 量化、PagedAttention 的核心驱动力。

---

## L3 子页面

- [[1 训练显存七项分解详解]] — 每项的精确计算 + ZeRO 各 Stage 如何分片

- [[2 KV Cache 工程与优化]] — GQA/MQA 降 KV、KV 量化、PagedAttention 虚拟内存

[[1 训练显存七项分解详解]]

[[2 KV Cache 工程与优化]]