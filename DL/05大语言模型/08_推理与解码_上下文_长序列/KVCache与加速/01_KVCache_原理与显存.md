# KV Cache：原理与显存分析

## 1. 什么是 KV Cache？
在 Transformer 的自回归解码（Decode）阶段，每一步生成新的 Token 时，都需要计算它与之前所有 Token 的 Attention。

$$ \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

- **问题**：之前 Token 的 Key ($K$) 和 Value ($V$) 向量在每一步计算中都是不变的。如果每次都重新计算，会造成巨大的算力浪费。
- **解决**：将之前所有 Token 的 $K$ 和 $V$ 向量缓存下来，称为 **KV Cache**。每一步只计算当前新 Token 的 $q, k, v$，并将新的 $k, v$ 追加（Append）到 Cache 中。

## 2. 显存占用计算
KV Cache 是推理阶段显存占用的主要来源之一，特别是对于长序列。

**公式**：
$$ \text{Size}_{KV} = 2 \times L \times N_{head} \times d_{head} \times S \times B \times P_{bytes} $$
- $2$：Key 和 Value 两个矩阵。
- $L$：层数 (Layers)。
- $N_{head}$：注意力头数。
- $d_{head}$：每个头的维度。
- $S$：序列长度 (Sequence Length)。
- $B$：Batch Size。
- $P_{bytes}$：精度字节数 (FP16 = 2 bytes)。

## 3. 语音大模型的 KV Cache 挑战

### 3.1 序列长度对比
假设我们要处理一段 30 秒的音频对话：
- **文本模型**：
  - 语速 3词/秒 $\rightarrow$ 90 词 $\approx$ 120 Tokens。
  - KV Cache 极小，几乎可以忽略。
- **语音模型 (Audio LLM)**：
  - 使用 EnCodec (24kHz, downsample 320) $\rightarrow$ 75 Hz frame rate。
  - 30秒 $\rightarrow$ $30 \times 75 = 2250$ Tokens。
  - 序列长度是文本的 **~20倍**。

### 3.2 显存压力实例
以 LLaMA-7B 为例 ($L=32, N_{head}=32, d_{head}=128$)，FP16，Batch Size = 1。
- **文本 (120 Tokens)**：
  $2 \times 32 \times 32 \times 128 \times 120 \times 1 \times 2 \approx 60 \text{ MB}$。
- **语音 (2250 Tokens)**：
  $2 \times 32 \times 32 \times 128 \times 2250 \times 1 \times 2 \approx 1.1 \text{ GB}$。

**结论**：对于 Audio LLM，KV Cache 不再是"配角"，而是显存消耗的"主角"。如果 Batch Size 增加到 32，仅 KV Cache 就需要 35GB 显存，单张 A100 (40GB) 甚至无法放下。

### 3.3 优化策略
针对 Audio LLM 的 KV Cache 优化至关重要：
1.  **[[05_多查询注意力MQA|MQA]] / [[06_分组注意力GQA|GQA]] (Multi-Query / [[06_分组注意力GQA|Grouped-Query Attention]])**：减少 Key/Value 的头数，直接成倍减少 KV Cache 大小。
2.  **Window Attention**：只缓存最近 N 个 Token 的 KV（滑动窗口），适合流式语音生成。
3.  **KV Cache Quantization**：将 KV Cache 量化为 INT8 或 FP8，减少一半显存。
