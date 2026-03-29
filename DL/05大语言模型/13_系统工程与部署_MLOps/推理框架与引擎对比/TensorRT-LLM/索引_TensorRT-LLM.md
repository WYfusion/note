# TensorRT-LLM 学习索引

TensorRT-LLM 是 NVIDIA 推出的高性能大语言模型推理库，专为极致性能设计。

## 核心模块

### 1. 基础概念
*   [01_TensorRT-LLM定位_与ONNX_TensorRT区别.md](01_TensorRT-LLM定位_与ONNX_TensorRT区别.md): 了解它为什么快，以及与标准 TRT 的区别。
*   [02_模型导出与构建_权重转换流程.md](02_模型导出与构建_权重转换流程.md): 从 HF Checkpoint 到 TensorRT Engine 的完整流水线。

### 2. 性能优化
*   [03_插件与算子_GEMM_Attention.md](03_插件与算子_GEMM_Attention.md): 深入理解 FlashAttention 和 GEMM Plugin。
*   [04_量化_INT8_INT4_FP8流程.md](04_量化_INT8_INT4_FP8流程.md): 使用 ModelOpt (AMMO) 进行量化，降低显存占用。
*   [05_多GPU并行_TP_PP与通信.md](05_多GPU并行_TP_PP与通信.md): Tensor Parallelism 在大模型中的应用。

### 3. 生产部署
*   [06_部署与Serving_性能调优.md](06_部署与Serving_性能调优.md): 集成 Triton Inference Server，配置 In-flight Batching。
*   [07_常见坑_版本兼容_CUDA驱动.md](07_常见坑_版本兼容_CUDA驱动.md): 避坑指南，解决构建和运行时的常见错误。

## 学习路径建议

1.  **入门**: 先阅读 `01` 和 `02`，跑通一个简单的 Whisper 或 Qwen 示例。
2.  **进阶**: 尝试 `04` 进行 INT4 量化，观察显存和速度变化。
3.  **生产**: 结合 `06` 使用 Triton 部署服务，并参考 `07` 排查问题。
