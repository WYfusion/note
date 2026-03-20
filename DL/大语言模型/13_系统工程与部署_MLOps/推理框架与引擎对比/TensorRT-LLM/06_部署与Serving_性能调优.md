# 部署与 Serving：Triton 集成与性能调优

构建好 Engine 后，生产环境通常使用 **Triton Inference Server** 进行部署。NVIDIA 提供了专门的 `tensorrt_llm` backend。

## 1. Triton 架构

在 Triton 中部署 TensorRT-LLM 通常涉及三个模型实例的协作（Ensemble 模式）：

1.  **Preprocessing**: 将文本/音频转为 Token ID 或 Feature（Python Backend）。
2.  **TensorRT-LLM**: 执行核心推理（C++ Backend）。
3.  **Postprocessing**: 将 Token ID 解码为文本（Python Backend）。

### 1.1 目录结构
```
model_repository/
  preprocessing/
  tensorrt_llm/
  postprocessing/
  ensemble_model/
```

## 2. In-flight Batching (IFB)

这是 TensorRT-LLM 相比标准 FasterTransformer 的最大改进。

### 2.1 原理
*   **传统 Batching**: 必须等 Batch 中最长的一个请求生成完，整个 Batch 才能结束。短请求会被阻塞。
*   **In-flight Batching**: 允许在某个请求生成结束时，立即插入新的请求进入当前 Batch。

### 2.2 配置
在 `tensorrt_llm` 的 `config.pbtxt` 中配置：

```protobuf
parameters [
  {
    key: "gpt_model_type"
    value: { string_value: "inflight_fused_batching" }
  },
  {
    key: "max_num_sequences"
    value: { string_value: "64" }
  }
]
```

## 3. 性能调优指南

### 3.1 Scheduler Policy
*   **Guaranteed No Evict**: 保证一旦请求开始处理，就不会被踢出。适合低延迟场景。
*   **Max Utilization**: 允许暂停某些请求以腾出显存给其他请求。适合高吞吐场景。

### 3.2 KV Cache 显存分配
*   `kv_cache_free_gpu_mem_fraction`: 类似于 vLLM 的 `gpu_memory_utilization`。
*   对于长音频任务，建议预留足够显存，防止 OOM。

### 3.3 语音模型特有的调优
*   **Encoder-Decoder 平衡**:
    *   Whisper 的 Encoder 计算量固定，Decoder 计算量随输出长度变化。
    *   如果 Encoder 是瓶颈（短语音），可以考虑增加 Encoder 的实例数。
    *   如果 Decoder 是瓶颈（长语音转录），应增大 `max_batch_size` 并开启 IFB。

## 4. 监控指标

Triton 提供了丰富的 Prometheus 指标：
*   `nv_inference_queue_duration_us`: 排队时间。
*   `nv_inference_compute_input_duration_us`: Prefill (Encoder) 阶段耗时。
*   `nv_inference_compute_infer_duration_us`: Decode 阶段耗时。

通过监控这些指标，可以判断是卡在计算上（需量化/TP）还是卡在调度上（需调整 Batch Size）。
