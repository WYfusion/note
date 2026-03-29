# 模型并行：Tensor Parallel 与 Pipeline Parallel

当单张显卡连`模型参数`都放不下时（如 70B 模型），必须`切分模型`。

## 1. Tensor Parallelism (TP)

### 1.1 原理 (Megatron-LM)
将 Transformer 层内部的矩阵乘法切分到多张卡上。
*   **列切分 (Column Parallel)**: 将权重矩阵 $W$ 按列切分。
    $$ Y = X [W_1, W_2] = [XW_1, XW_2] $$
*   **行切分 (Row Parallel)**: 将权重矩阵 $W$ 按行切分。
*   **通信**: 每一层都需要 `All-Reduce` 通信，带宽要求极高（通常需要 NVLink）。

### 1.2 语音中的序列并行 (Sequence Parallelism)
语音序列通常极长（如 1小时音频）。TP 也可以结合序列并行。
*   **原理**: 在 Layer Norm 和 Dropout 层，将序列在时间维度切分到不同 GPU。
*   **Ring Attention**: 专门针对超长序列的注意力计算并行化，允许处理百万级 Token 的上下文（如整本有声书）。

## 2. Pipeline Parallelism (PP)

### 2.1 原理 (GPipe)
将模型的不同层（Layer）切分到不同 GPU 上。
*   GPU-0: Layer 1-10
*   GPU-1: Layer 11-20
*   ...
数据像流水线一样流过各个 GPU。

### 2.2 气泡问题 (Bubble)
*   当 GPU-0 计算 Batch 1 时，GPU-1 是空闲的。
*   **1F1B (One-Forward-One-Backward)**: 优化的调度策略，减少空闲时间。

## 3. 3D 并行

训练超大模型（如 GPT-4, Speech-LLaMA-Large）通常同时使用三种并行：
*   **Data Parallel**: 扩展 Batch Size。
*   **Tensor Parallel**: 扩展模型宽度/层内计算。
*   **Pipeline Parallel**: 扩展模型深度。

对于语音大模型，由于参数量通常小于文本 LLM（Whisper Large 仅 1.5B），通常 **DDP + ZeRO** 就足够了。只有在做超长上下文（Long Context）训练时，才需要 TP 或 Ring Attention。
