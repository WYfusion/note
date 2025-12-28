# vLLM 核心概念：PagedAttention 与连续批处理

vLLM 是目前最先进的大模型推理引擎之一，其核心创新在于解决了显存碎片化和批处理效率低下的问题。对于 Qwen-Audio 等基于 Transformer Decoder 的语音大模型，vLLM 同样能带来巨大的吞吐量提升。

## 1. PagedAttention

### 1.1 显存碎片化问题
在传统的推理框架（如 HuggingFace Transformers）中，KV Cache 必须存储在连续的显存空间中。
*   **预分配浪费**: 由于不知道生成的序列有多长，通常需要预分配最大长度（如 2048）的显存，导致大量闲置。
*   **碎片化**: 随着请求的生成和结束，显存中会出现大量不可用的碎片。

### 1.2 操作系统分页思想的引入
PagedAttention 借鉴了操作系统的虚拟内存管理机制：
*   **Block**: 将 KV Cache 切分为固定大小的块（Block），例如每块包含 16 个 Token 的 KV。
*   **Block Table**: 维护一个映射表，记录逻辑上的 Token 序列对应物理显存中的哪些 Block。
*   **非连续存储**: 物理显存中的 Block 可以是不连续的。

### 1.3 对 Audio LLM 的意义
语音生成的长度差异极大（短指令 vs 长演讲）。PagedAttention 允许显存按需分配，极大地提高了显存利用率，从而允许更大的 Batch Size。

---

## 2. 连续批处理 (Continuous Batching)

### 2.1 静态批处理的缺陷
传统 Batching 要求同一个 Batch 内的所有请求必须同时结束。
*   **短板效应**: 如果 Batch 中有一个长任务（生成 500 Tokens）和一个短任务（生成 10 Tokens），短任务结束后，显卡必须空转等待长任务结束。

### 2.2 迭代级调度 (Iteration-level Scheduling)
vLLM 实现了连续批处理（也称 In-flight Batching）：
*   **随时插入**: 当一个请求生成结束（遇到 EOS），vLLM 会立即释放其显存，并从等待队列中插入一个新的请求进入当前 Batch。
*   **利用率拉满**: GPU 始终处于满载状态，不会因为等待长任务而闲置。

---

## 3. 多模态支持 (Multimodal Support)

vLLM 正在积极扩展对多模态模型的支持。
*   **原理**: 对于 Qwen-Audio 这类模型，音频特征被视为一种特殊的 Embedding 输入。vLLM 的 PagedAttention 机制天然支持处理这些 Embedding 对应的 KV Cache。
*   **现状**: 目前 vLLM 已支持 LLaVA (Vision), Qwen-VL 等。对 Qwen-Audio 的支持通常通过自定义 Model Class 或官方更新实现。

$$
\text{Throughput} = \frac{\text{Total Tokens Generated}}{\text{Total Time}}
$$
在实测中，vLLM 的吞吐量通常是 HuggingFace Transformers 的 2-4 倍。
