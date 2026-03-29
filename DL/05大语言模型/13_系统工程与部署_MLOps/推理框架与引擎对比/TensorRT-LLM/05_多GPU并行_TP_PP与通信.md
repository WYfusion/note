# 多 GPU 并行：TP, PP 与通信

当模型参数量超过单张显卡的显存限制，或者为了进一步降低延迟，我们需要使用多 GPU 并行。TensorRT-LLM 原生支持 Tensor Parallelism (TP) 和 Pipeline Parallelism (PP)。

## 1. 并行策略

### 1.1 Tensor Parallelism (TP)
*   **原理**: 将每一层的权重矩阵（如 $W_Q, W_K, W_V$）切分到多个 GPU 上。
*   **计算**: 每个 GPU 计算一部分结果，然后通过 `AllReduce` 操作合并。
*   **通信**: 通信频繁，发生在每一层。
*   **适用**: 单机多卡，需要高带宽互联（NVLink）。
*   **Audio LLM**: 对于 7B-70B 的模型，TP 是首选方案。

### 1.2 Pipeline Parallelism (PP)
*   **原理**: 将模型的不同层（Layers）分配到不同 GPU 上（如 GPU0 跑前 10 层，GPU1 跑后 10 层）。
*   **通信**: 仅在切分点进行点对点通信。
*   **适用**: 跨机多卡，或者超大模型（>100B）。
*   **缺点**: 存在“气泡”（Bubble），GPU 利用率不如 TP。

## 2. 构建支持 TP 的引擎

在转换 Checkpoint 时就需要指定 TP 的大小。

```bash
# 1. 转换权重 (指定 TP=2)
python3 convert_checkpoint.py \
    --model_dir ./Qwen-Audio-Chat \
    --output_dir ./tllm_checkpoint_tp2 \
    --tp_size 2

# 2. 构建引擎
# 注意：这会生成两个文件 rank0.engine 和 rank1.engine
trtllm-build \
    --checkpoint_dir ./tllm_checkpoint_tp2 \
    --output_dir ./tllm_engines_tp2
```

## 3. 运行多卡推理

TensorRT-LLM 使用 MPI (Message Passing Interface) 来管理多进程。

```bash
# 使用 mpirun 启动 2 个进程
mpirun -n 2 \
    --allow-run-as-root \
    python3 ../run.py \
    --engine_dir ./tllm_engines_tp2 \
    --input_text "测试音频"
```

### 3.1 常见错误
*   **World Size Mismatch**: 启动的 MPI 进程数必须等于构建时的 `tp_size`。
*   **NCCL Timeout**: 通常是因为防火墙阻挡了 GPU 之间的通信，或者 P2P 通信未启用。

## 4. 语音模型的特殊性

### 4.1 Encoder 的并行
*   Whisper 的 Encoder 相对较小，通常**不进行 TP 切分**，而是复制到每个 GPU 上（Replication），或者仅在 Rank 0 上运行。
*   如果 Encoder 不切分，而 Decoder 切分，需要在 Encoder 输出传给 Decoder 之前进行广播（Broadcast）。

### 4.2 性能权衡
*   对于较小的模型（如 Whisper Large v3, ~1.5B），开启 TP=2 可能会**变慢**。
*   原因：计算量本身很小，跨卡通信的开销超过了并行计算带来的收益。
*   **建议**: 10B 以下模型优先单卡跑，多卡用于增加 Batch Size（Data Parallelism）而非 TP。
