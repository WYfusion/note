# 显存优化：ZeRO 与 FSDP

显存是训练大模型的最大瓶颈。除了模型参数，优化器状态（Optimizer States）和梯度（Gradients）也占用大量显存。

## 1. ZeRO (Zero Redundancy Optimizer)

DeepSpeed 提出的`零冗余优化器`。

### 1.1 三个阶段
*   **ZeRO-1**: 切分优化器状态 (Optimizer States)。显存节省 4 倍（对于 Adam）。
*   **ZeRO-2**: 切分梯度 (Gradients)。显存进一步节省 2 倍。
*   **ZeRO-3**: 切分模型参数 (Parameters)。显存占用与 GPU 数量成反比。

### 1.2 通信换显存
ZeRO-3 在前向传播时，需要临时从其他 GPU 拉取参数，计算完立即释放。这增加了通信开销，但允许在有限显存下训练超大模型。

## 2. FSDP (Fully Sharded Data Parallel)

PyTorch 原生实现的 ZeRO-3。
*   **Unit**: 将模型层包装为 FSDP Unit。
*   **Prefetch**: 预取下一层的参数，掩盖通信延迟。

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
