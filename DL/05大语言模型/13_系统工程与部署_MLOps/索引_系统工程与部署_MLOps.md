# 系统工程与部署 (MLOps)

本章节关注如何将实验室中的模型转化为高可用、高性能的生产级服务。

## 目录

### [01_从训练到上线_Release流程.md](01_从训练到上线_Release流程.md)
- **MLOps 闭环**：Data -> Train -> Deploy -> Monitor。
- **语音数据工程**：Resampling, VAD Cleaning, Augmentation。
- **训练优化**：Gradient Checkpointing, Flash Attention。
- **监控指标**：RTF (Real Time Factor), TTFT (Time to First Token)。

## 子模块

### [性能优化](索引_性能优化_量化_蒸馏.md)
- **量化**：GPTQ, AWQ。Encoder 慎用 INT4。
- **蒸馏**：Distil-Whisper。
- **流式优化**：KV Cache, Streaming Encoder。

### [推理框架](索引_推理框架与引擎对比.md)
- **Faster-Whisper (CTranslate2)**：Whisper 部署首选。
- **vLLM**：多模态 LLM 的高吞吐选择。
- **ONNX Runtime**：端侧部署。

### [服务化与网关](索引_服务化与网关.md)
- **API 设计**：OpenAI Compatible Audio API。
- **协议**：WebSocket (Real-time), gRPC。
- **Batching**：Bucketing strategy for variable-length audio。

