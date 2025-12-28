# FlashAttention 与 GEMM 优化

在底层硬件层面，Transformer 的效率主要受限于**计算 (Compute-bound)** 和 **显存带宽 (Memory-bound)**。

## 1. GEMM (General Matrix Multiply)

Transformer 的大部分计算量都在矩阵乘法上（Linear 层, Attention 中的 QK, PV）。
$$ C = \alpha AB + \beta C $$

### 1.1 Tensor Cores
现代 NVIDIA GPU (Volta, Ampere, Hopper) 都有专门的 Tensor Cores 用于加速矩阵乘法。
*   **混合精度**: FP16 / BF16 计算，FP32 累加。
*   **形状要求**: 矩阵维度通常需要是 8 或 16 的倍数才能最大化利用 Tensor Cores。

## 2. FlashAttention 的内核优化

FlashAttention 不仅是算法创新，更是系统级优化。

### 2.1 IO-Awareness (IO 感知)
*   **HBM vs SRAM**: GPU 显存 (HBM) 容量大但速度慢，片上缓存 (SRAM) 速度快但容量小。
*   **核心策略**: 尽可能减少 HBM 的读写次数。将 $Q, K, V$ 分块加载到 SRAM，在 SRAM 内完成 Attention Score 计算和 Softmax，只将最终结果写回 HBM。

### 2.2 Kernel Fusion (算子融合)
将多个操作（MatMul, Mask, Softmax, Dropout, MatMul）融合为一个 CUDA Kernel。
*   **减少 Kernel Launch 开销**: CPU 发射 Kernel 也是有成本的。
*   **减少中间结果读写**: 融合后，中间结果直接在寄存器或 SRAM 中传递，无需写回 HBM。

## 3. 对长音频训练的意义

### 3.1 显存节省
标准 Attention 的显存占用是 $O(L^2)$。对于 1 小时的音频 ($L \approx 180k$)，这是不可接受的。
FlashAttention 将显存占用降低到 $O(L)$，使得在单卡上训练长音频模型成为可能。

### 3.2 训练速度
由于减少了 HBM 访问，FlashAttention 通常能带来 2-4 倍的训练加速。这意味着训练一个 Whisper 或 Speech-LLaMA 模型的时间大幅缩短。
