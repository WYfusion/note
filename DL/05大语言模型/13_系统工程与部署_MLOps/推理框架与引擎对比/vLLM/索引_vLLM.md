# vLLM 推理引擎

本章节介绍 vLLM 这一高性能推理引擎，重点探讨其在多模态（Audio LLM）场景下的应用与调优。

## 目录

### [01_vLLM核心概念_PagedAttention_连续批处理.md](01_vLLM核心概念_PagedAttention_连续批处理.md)
- **PagedAttention**：解决显存碎片化，提升 Audio LLM 长序列推理的 Batch Size。
- **Continuous Batching**：消除长短音频混合推理时的等待时间。

### [02_安装与GPU环境_CUDA驱动.md](02_安装与GPU环境_CUDA驱动.md)
- **环境要求**：CUDA 12.1+, PyTorch 2.x。
- **Docker**：推荐使用官方镜像以避免环境冲突。

### [03_OpenAI兼容服务_启动与参数.md](03_OpenAI兼容服务_启动与参数.md)
- **API Server**：完全兼容 OpenAI Chat Completions 协议。
- **Audio 调用**：通过多模态 Prompt 格式传入音频 URL。

### [04_吞吐与延迟_并发与批处理调参.md](04_吞吐与延迟_并发与批处理调参.md)
- **参数调优**：`max_num_seqs` (并发度) vs `max_num_batched_tokens`。
- **Audio 策略**：音频 Token 消耗大，需适当降低并发限制。

### [05_KVCache管理_显存估算.md](05_KVCache管理_显存估算.md)
- **显存分配**：权重占用 + KV Cache (Paged)。
- **Swap Space**：CPU 卸载机制防止 OOM。

### [06_量化支持_AWQ_GPTQ_bitsandbytes.md](06_量化支持_AWQ_GPTQ_bitsandbytes.md)
- **AWQ**：vLLM 首选量化格式，速度快精度高。
- **FP8**：H100/4090 专用加速。

### [07_LoRA在线挂载与多Adapter.md](07_LoRA在线挂载与多Adapter.md)
- **Multi-LoRA**：单实例同时服务多个微调模型（如不同领域的 ASR）。
- **S-LoRA**：零开销切换。

### [08_常见问题_爆显存_卡死_输出异常.md](08_常见问题_爆显存_卡死_输出异常.md)
- **OOM**：调整 `gpu_memory_utilization` 和 `max_model_len`。
- **NCCL Timeout**：Docker 需开启 `--ipc=host`。

