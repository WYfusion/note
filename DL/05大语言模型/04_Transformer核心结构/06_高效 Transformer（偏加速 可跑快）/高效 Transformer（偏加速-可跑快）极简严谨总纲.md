## 0. 范围

- 关注：**稀疏 / 局部 / 线性 / 低秩 / 分块注意力 / FlashAttention / 长上下文工程 / 推理加速**

- 不展开：KV 设计、头设计、MoE、量化细节、纯替代架构（SSM/RNN 等）

---

## 1. 问题本质

### 1.1 标准自注意力瓶颈

- 设序列长 `n`，头维 `d`

- 朴素 self-attention：计算 $O(n^2 d)$，注意力矩阵显存 $O(n^2)$

- 真正工程瓶颈常不只是 FLOPs，而是 **HBM/显存读写、kernel launch、低占用、跨卡通信**

### 1.2 高效 Transformer 的 4 条主线

1. **不改语义，只改实现** — 保持 exact attention，降低 IO / 提高并行度（FlashAttention / Ring Attention）

1. **改连接模式（Sparse / Local / Block）** — 少算一部分注意力

1. **改近似形式（Linear / Low-rank / Kernel）** — 把 $n^2$ 降为近线性

1. **改有效上下文与服务系统** — 少送 token、提高吞吐

---

## 2. 六大技术分类（详见子页面）

> 以下每类均有独立子页面展开，叶子层级包含数学推导、工程实现与 Python 示例。

### 第一类：Exact Attention 的工程优化（现代主线）

- FlashAttention / FlashAttention-2 / FlashAttention-3

- 核心：tiling + online softmax + kernel 融合，不改数学，只降 IO

- **默认优先级最高**

### 第二类：局部 / 滑窗 / 分块 / 稀疏注意力

- Sliding Window（Longformer）、Block Sparse（Sparse Transformer）、Local+Global+Random（BigBird）、Hash/Routing（Reformer）

- 核心：减少注意力连接数，复杂度近线性

### 第三类：线性 / 低秩 / 核方法近似注意力

- Linformer（低秩投影）、Nyströmformer（Nyström 近似）、Performer（FAVOR+ 核映射）

- 核心：避免显式构造 $n \times n$ 矩阵

### 第四类：分布式长上下文 Exact Attention

- Ring Attention、Striped Attention、Context/Sequence Parallelism

- 核心：跨设备块级 exact attention + 通信计算重叠

### 第五类：上下文工程

- Context Selection、Prompt Compression（LLMLingua）、Hierarchical Context、Chunk Packing

- 核心：不改模型，减少有效输入长度 $n to n'$，ROI 极高

### 第六类：推理加速

- Speculative Decoding（EAGLE/Medusa）、Continuous Batching、Chunked Prefill

- 核心：减少大模型串行步数 / 提高 GPU 利用率

---

## 3. 严谨分类法：从"改动最小"到"改动最大"

|Level|策略|代表|特点|
|---|---|---|---|
|**L1：只换 kernel/系统**|FlashAttention、continuous batching、prompt compression、speculative decoding|不改模型|风险最低，工程最先做|
|**L2：改连接模式**|sliding window、block sparse、local+global、ring/blockwise exact|改注意力拓扑|需重训/微调|
|**L3：改注意力数学**|Linformer、Nyströmformer、Performer|近似 attention|质量需单独验证|

---

## 4. 训练与推理最优优先级

### 训练侧默认顺序

1. **FlashAttention**

1. Activation checkpointing

1. Sequence/context parallelism

1. 合理 micro-batch / global batch

1. 若仍需超长序列 → sliding window / block sparse / ring attention

1. 若从头设计极长序列模型 → linear / low-rank attention

### 推理侧默认顺序

1. **减少无效上下文**

1. **FlashAttention / 高效 prefill kernel**

1. Continuous batching + 调度

1. Speculative decoding

1. 若业务天然局部 → sliding window serving

1. 若必须极长上下文且模型可重训 → linear attention 主干

---

## 5. 最常见误区

- ❌ "复杂度更低就一定更快" — GPU 实测依赖 kernel 成熟度、memory access pattern、Tensor Core 利用率

- ❌ "linear attention 一定优于 FlashAttention" — 很多现实区间里 exact + IO-aware kernel 更稳更快

- ❌ "长上下文问题一定靠改注意力解决" — 先做 context selection / compression / packing

- ❌ "稀疏越稀越好" — 不规则稀疏可能更慢，**结构化稀疏 > 任意稀疏**

- ❌ "训练优化和推理优化是一回事" — 训练看激活显存/通信/MFU；推理看 prefill 吞吐/decode 串行性/调度

---

## 6. 推荐综合使用流程

### 工程落地

1. 建立 dense baseline → 上 FlashAttention

1. Profile：prefill vs decode、compute-bound vs memory-bound、单卡 vs 跨卡

1. 上下文工程：retrieval / truncation / compression / packing

1. 服务优化：batching / scheduling / speculative decoding

1. 长上下文仍是矛盾 → sliding window / block sparse / ring attention

1. 从头构建极长序列主干 → linear / low-rank attention

### 研究设计

1. 明确依赖类型：局部主导 / 稀疏全局 / 密集全局

1. 局部 → sliding window；局部+全局 → BigBird/sparse；exact 必须 → Flash+Ring；超长+可近似 → Performer/Linformer

---

## 7. 一页式结论

> [!important]
> 
> **高效 Transformer = 数学降复杂度 + IO 优化 + 分布式并行 + 上下文缩减 + 服务调度**
> 
> 真正决定 wall-clock 的常常不是公式阶数，而是 IO、kernel、通信、有效上下文长度、调度。
> 
> **最稳主线**：FlashAttention + 上下文工程 + 服务调度 → 长上下文加 ring/sequence parallelism → 结构改造加 sparse → 最后才考虑 linear/low-rank 主干替换。
> 
> **总原则**：能不改模型先不改；能减 token 先减 token；能 exact 先 exact；能结构化稀疏不做任意稀疏；先看 profile 再选方案。

[[0. FlashAttention 系详解：Exact Attention 的工程优化]]

[[0. 局部 - 滑窗 - 分块 - 稀疏注意力详解]]

[[0. 线性 - 低秩 - 核方法近似注意力详解]]

[[0. 分布式长上下文 Exact Attention 详解]]

[[0. 上下文工程详解：最常被低估但 ROI 最高]]

[[0. 推理加速详解：不改或少改主模型]]