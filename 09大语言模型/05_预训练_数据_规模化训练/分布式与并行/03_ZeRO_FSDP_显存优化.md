# 显存优化：ZeRO 与 FSDP

显存是训练大模型的最大瓶颈。除了模型参数，优化器状态（Optimizer States）和梯度（Gradients）也占用大量显存。

## 1. ZeRO (Zero Redundancy Optimizer)

DeepSpeed 提出的`零冗余优化器`。
https://arxiv.org/pdf/1910.02054.pdf
### 1.1 三个阶段
*   **ZeRO-1**: 切分优化器状态 (Optimizer States)。显存节省 4 倍（对于 Adam）。
*   **ZeRO-2**: 切分梯度 (Gradients)。显存进一步节省 2 倍。
*   **ZeRO-3**: 切分模型参数 (Parameters)。显存占用与 GPU 数量成反比。

### 1.2 通信换显存
https://arxiv.org/pdf/2304.11277.pdf
ZeRO-3 在前向传播时，需要临时从其他 GPU 拉取参数，计算完立即释放。这增加了通信开销，但允许在有限显存下训练超大模型。
## 2. FSDP (Fully Sharded Data Parallel，完全分片数据并行)

PyTorch 原生实现的 ZeRO-3 算法，旨在解决大模型训练中的显存瓶颈。它将模型参数、梯度和优化器状态切分到所有数据并行的 Worker 上。

### 2.1 核心工作逻辑
FSDP 的核心思想是“**用通信换显存**”。它结合了数据并行（Data Parallelism）和模型分片。

*   **逐层计算 (Layer-wise / Unit-wise)**:
    *   FSDP 将模型划分为多个 **FSDP Unit**（通常对应 Transformer 的一个 Layer 或 Block）。
    *   **计算时机**: 只有在真正计算某一层（Unit）时，才临时重组该层的完整参数。
    *   **参数可见性**: 在任意时刻，GPU 显存中**只存在当前正在计算的那一层**的完整参数，而不是整个模型的完整参数。其他层的参数仍然保持分片状态。

*   **数据并行与 AllGather**:
    *   每个 GPU 拥有不同的**数据切片**（Batch 的一部分）。
    *   **AllGather 后**: 所有 GPU **同时**拥有了当前 Unit 的**完整参数**。
    *   **并行计算**: 所有 GPU 使用**相同的完整参数**，但输入是**各自不同的数据切片**，并行进行前向/反向传播。

*   **分片 (Sharding)**:
    *   **参数 (Parameters)**: 每个 GPU 只存储 $1/N$ 的模型参数。
    *   **梯度 (Gradients)**: 每个 GPU 只存储 $1/N$ 的梯度。
    *   **优化器状态 (Optimizer States)**: 每个 GPU 只负责更新 $1/N$ 的参数。

*   **通信原语**:
    *   **AllGather**: 在正向和反向传播前，收集完整参数。
    *   **ReduceScatter**: 在反向传播后，同步并切分梯度。

### 2.2 训练流程与注意事项

#### 正向传播 (Forward Pass)
1.  **AllGather**: 对于当前计算层（FSDP Unit），从所有 GPU 收集完整的参数分片。
2.  **Compute**: 执行前向计算。
3.  **Free**: 计算完成后，立即释放收集到的完整参数，仅保留属于本 GPU 的分片。

**注意事项**:
*   **通信开销**: 频繁的 AllGather 会带来大量通信。FSDP 通过**参数预取 (Prefetching)** 来掩盖这部分延迟（在计算当前层时，提前拉取下一层的参数）。
*   **显存峰值**: 虽然静态显存低，但前向传播时会瞬间持有完整层的参数，产生显存峰值。

#### 反向传播 (Backward Pass)
1.  **AllGather**: 再次收集当前层的完整参数（因为前向传播后已经释放了）。
2.  **Compute Gradients**: 计算当前层的梯度。
3.  **Free**: 释放完整参数。
4.  **ReduceScatter**: 将计算出的梯度在所有 GPU 间进行归约（Sum/Avg），然后每个 GPU 只保留自己负责的那部分梯度分片。

**注意事项**:
*   **两次 AllGather**: 相比 DDP，FSDP 在反向传播时多了一次 AllGather 参数的过程（因为参数被释放了）。
*   **梯度同步**: 梯度计算完后立即 ReduceScatter，释放未分片的梯度显存。

### 2.3 FSDP1 vs FSDP2 (PyTorch 版本演进)

随着 PyTorch 的发展，FSDP 经历了架构上的升级。

#### FSDP1 (FlatParameter)
*   **机制**: 将一个模块（Module）内的所有参数展平（Flatten）为一个巨大的 1D Tensor (`FlatParameter`)，然后对这个大 Tensor 进行切分。
*   **优点**: 内存布局连续，通信时可以合并小 Tensor，提高带宽利用率。
*   **缺点**:
    *   **Padding**: 如果参数量不能被 GPU 数整除，需要填充 (Padding)，浪费显存和带宽。
    *   **复杂性**: 展平操作破坏了原始参数的结构，使得与某些插件或自定义操作的兼容性变差。
    *   **锁步执行**: 通信和计算的重叠（Overlap）受到一定限制。

#### FSDP2 (Per-Parameter Sharding / DTensor)
*   **机制**: 基于 `DTensor` (Distributed Tensor)，不再展平参数。直接在每个原始参数张量上进行切分（通常在维度 0）。
*   **拆分逻辑详解**:
    *   FSDP2 将每个 Parameter 视为一个独立的实体进行管理。
    *   利用 `DTensor` 将逻辑上的全局 Tensor 映射到物理设备网格（Device Mesh）上。
*   **处理不均匀切分 (Uneven Sharding)**:
    *   **问题**: 当参数的某个维度大小不能被 GPU 数量（World Size）整除时（例如维度为 10，GPU 数为 4）。
    *   **FSDP1 做法**: 必须进行 **Padding**（填充 0），使其能被整除（变成 12），这浪费了显存和通信带宽。
    *   **FSDP2 做法**: 支持 **Uneven Sharding**（不均匀切分）。它允许不同 GPU 持有不同大小的分片。
        *   例如：GPU 0 和 1 各持有 3 个元素，GPU 2 和 3 各持有 2 个元素。
        *   这种方式完全消除了 Padding 开销。
*   **优点**:
    *   **无 Padding**: 节省显存和带宽。
    *   **更灵活**: 保留了原始参数结构，对用户更透明。
    *   **更好的流水线**: 通信和计算的重叠做得更好，支持更细粒度的异步操作。
    *   **Composable**: 更容易与其他并行策略（如 TP 张量并行、SP 序列并行）组合使用。

## 3. 激活重算 (Activation Checkpointing / Gradient Checkpointing)

### 3.1 原理
*   **标准反向传播**: 需要保存前向传播的所有中间激活值（Activations），用于计算梯度。显存占用 $O(L)$。
*   **重算**: 不保存中间激活值。在反向传播时，重新计算前向传播。
*   **代价**: 计算量增加约 33%，但显存占用显著降低（$O(\sqrt{L})$ 或 $O(1)$）。

### 3.2 语音模型中的应用
语音 Encoder（如 Conformer）通常很深且序列很长，激活值占用的显存非常惊人。
*   **例子**: 训练 24 层 Conformer，序列长度 3000。如果不开启 Checkpointing，A100 (80G) 可能只能跑 Batch Size = 1。
*   **开启后**: Batch Size 可以提升到 8 或 16，显著提升训练吞吐量。

## 4. 混合精度与显存

*   **FP32**: 4 bytes/param。
*   **FP16/BF16**: 2 bytes/param。
*   **KV Cache**: 在推理时，KV Cache 占用大量显存。FlashAttention 可以减少这部分开销。
