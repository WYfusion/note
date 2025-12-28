# FlashAttention: IO 感知的精确注意力加速

标准的 Self-Attention 计算不仅时间复杂度是 $O(L^2)$，显存占用也是 $O(L^2)$。对于长序列（如长音频），显存往往是瓶颈。FlashAttention 通过**IO 感知（IO-Aware）**的设计，在不改变计算结果的前提下，实现了显著的加速和显存节省。

## 1. 标准 Attention 的痛点

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

在 GPU 上执行时：
1.  从 HBM (显存) 读取 $Q, K$ 到 SRAM (片上缓存)。
2.  计算 $S = QK^T$，写回 HBM。
3.  从 HBM 读取 $S$，计算 $P = \text{softmax}(S)$，写回 HBM。
4.  从 HBM 读取 $P, V$，计算 $O = PV$，写回 HBM。

**问题**: 中间矩阵 $S$ 和 $P$ 的大小是 $[B, H, L, L]$。当 $L$ 很大时，HBM 的读写带宽成为瓶颈，且显存容易 OOM。

## 2. FlashAttention V1 核心原理

### 2.1 分块计算 (Tiling)
将 $Q, K, V$ 切分成小块（Block），使得每个 Block 的计算都能在 SRAM 中完成，不需要频繁读写 HBM。

### 2.2 重计算 (Recomputation)
为了节省显存，**不存储**中间巨大的注意力矩阵 $S$ 和 $P$。
在反向传播时，利用存储的 $Q, K, V$ 和输出 $O$，重新计算一遍 Attention Score。虽然多了一些 FLOPs（计算量），但减少了 HBM 读写（IO），总体速度反而更快。

### 2.3 算法流程
1.  加载 $Q, K, V$ 的小块到 SRAM。
2.  在 SRAM 中计算局部 Score。
3.  利用 Online Softmax 技术，动态更新局部的 Softmax 归一化因子。
4.  直接输出结果到 HBM。

## 3. FlashAttention V2 的改进

*   **减少非矩阵乘法操作**: 优化了 Softmax 的计算逻辑。
*   **更好的并行化**: V1 主要在 Batch 和 Head 维度并行，V2 增加了在 Sequence Length 维度的并行。

## 4. 对长音频的意义

音频序列通常极长。
*   10 秒语音 $\approx$ 500 帧 (20ms/帧)。
*   1 小时录音 $\approx$ 180,000 帧。

如果没有 FlashAttention，Transformer 很难处理超过 1 分钟的音频（$L \approx 3000$）。
有了 FlashAttention，我们可以轻松训练 $L=32k$ 甚至更长的序列，使得**长音频理解（Long-form Audio Understanding）**成为可能。
