# 数据并行 (Data Parallelism) 与 DDP

当单张显卡能放得下模型，但`放不下所有数据`时，使用`数据并行`。

## 1. 传统 DataParallel (DP)

### 1.1 原理
*   **单进程多线程**: 仅启动一个进程，利用多线程管理多个 GPU。
*   **Parameter Server 模式**: 主 GPU (通常是 GPU-0) 负责分发数据、汇总梯度和更新参数。
*   **流程**:
    1.  **Scatter**: 将 Batch 数据分发到各 GPU。
    2.  **Replicate**: 每次前向传播前，将模型从 GPU-0 复制到各 GPU（通信开销大）。
    3.  **Parallel Apply**: 各 GPU 并行计算前向传播。
    4.  **Gather**: 将各 GPU 的输出汇总到 GPU-0 计算 Loss。
    5.  **Backward**: 梯度反向传播，汇总到 GPU-0 更新参数。

### 1.2 缺陷
*   **GIL 瓶颈**: Python 全局解释器锁 (GIL) 限制了多线程的并行效率。
*   **负载不均**: GPU-0 承担了过多的通信和计算（汇总 Loss、更新参数），显存占用远高于其他卡，容易 OOM。
*   **现状**: 在大模型训练中基本被淘汰，仅用于简单的单机多卡调试。

## 2. DDP (Distributed Data Parallel) 原理

### 1.1 核心流程
1.  **模型复制**: 将模型参数复制到 $N$ 张 GPU 上。
2.  **数据切分**: 将一个 Batch 的数据切分为 $N$ 份，每张卡处理一份。
3.  **前向/反向**: 每张卡独立计算梯度。
4.  **梯度同步 (All-Reduce)**: 在更新参数前，所有卡通信，计算梯度的平均值。
5.  **参数更新**: 每张卡使用平均梯度更新参数，保证所有卡模型一致。

### 1.2 通信开销
*   通信量与模型参数量成正比。
*   **Ring All-Reduce**: 高效的环状通信算法，带宽利用率高。

## 2. 语音训练中的 DDP 挑战：变长序列

语音数据的长度差异极大（1秒 vs 30秒），这给 DDP 带来了负载不均衡问题。

### 2.1 问题描述
假设 GPU-0 分到的全是短音频，GPU-1 分到的全是长音频。
*   GPU-0 很快算完，然后空闲等待（Idle）。
*   GPU-1 还在计算。
*   **木桶效应**: 整体速度取决于最慢的 GPU-1。

### 2.2 解决方案：Bucket Batching (分桶)
*   **原理**: 将训练数据按长度排序。
*   **操作**:
    1.  构建多个 Bucket（如 0-5s, 5-10s, 10-20s）。
    2.  每次生成 Batch 时，只从同一个 Bucket 中取样。
    3.  **DDP 适配**: 确保分给不同 GPU 的数据长度尽可能一致。
*   **注意**: 即使使用了 Bucket，不同 Batch 之间的计算量也不同，可能导致 GPU 利用率波动。

### 2.3 动态 Batch Size
*   不固定 Batch 中的样本数量（Batch Size），而是固定总 Token 数（Max Tokens）。
*   长音频 Batch 包含较少样本，短音频 Batch 包含较多样本。
*   这能最大化显存利用率，减少 Padding 浪费。

## 3. 进阶方案：DeepSpeed

DeepSpeed 是微软推出的深度学习优化库，它本质上是一种**增强版的数据并行**。

### 3.1 核心优势
*   **ZeRO (Zero Redundancy Optimizer)**: 解决了 DDP 中每个 GPU 都存储完整模型参数和优化器状态的冗余问题。通过将这些状态切分到不同 GPU 上，DeepSpeed 可以在有限的显存中训练参数量大得多的模型。详见[[03_ZeRO_FSDP_显存优化]]
*   **高效通信**: 优化了 All-Reduce 通信策略，支持 1-bit Adam 等压缩通信技术。
*   **易用性**: 只需要简单的配置 (`deepspeed_config.json`) 即可替换 PyTorch 原生的 DDP，且无缝支持 HuggingFace Trainer。

### 3.2 与 DDP 的关系
*   DeepSpeed 在底层默认使用 DDP 的通信模式（Ring All-Reduce）。
*   当开启 ZeRO-1/2/3 时，它改变了参数和梯度的存储/同步方式，从而打破了显存墙。
*   对于语音大模型（如 Whisper Fine-tuning），DeepSpeed 是节省显存、提升 Batch Size 的首选工具。
