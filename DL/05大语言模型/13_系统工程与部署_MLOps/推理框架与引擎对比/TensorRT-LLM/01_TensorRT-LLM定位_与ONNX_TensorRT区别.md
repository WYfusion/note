# TensorRT-LLM 定位：与 ONNX、TensorRT 的区别

TensorRT-LLM 是 NVIDIA 专门为大语言模型（LLM）推理推出的高性能库。对于追求极致性能的语音大模型（如 Whisper, Qwen-Audio）部署，它是目前的“天花板”方案。

## 1. 核心定位

### 1.1 什么是 TensorRT-LLM？
它不是一个简单的推理引擎，而是一个**工具箱**，包含：
*   **Model Definition API**: 类似于 PyTorch 的 API，用于定义模型结构。
*   **Optimized Kernels**: 针对 LLM 优化的 CUDA 核函数（如 FlashAttention, PagedAttention）。
*   **In-flight Batching**: 支持连续批处理。
*   **Tensor Parallelism**: 自动处理多卡通信。

### 1.2 与标准 TensorRT 的区别
*   **TensorRT**: 通用的深度学习编译器，擅长 CNN 和静态 Shape 的模型。对于 Transformer 的动态 Shape 和 KV Cache 管理，原生支持较弱，通常需要手写 Plugin。
*   **TensorRT-LLM**: 基于 TensorRT，但内置了所有 LLM 需要的高级 Plugin（如 GPT Attention Plugin），并提供了 Python 层的封装，极大降低了使用门槛。

### 1.3 与 ONNX Runtime 的区别
*   **ONNX Runtime**: 跨平台、兼容性好，支持 CPU/GPU/NPU。适合端侧部署或非 NVIDIA 硬件。
*   **TensorRT-LLM**: 绑定 NVIDIA GPU，性能远超 ONNX Runtime（通常快 2-4 倍），但部署流程更复杂。

---

## 2. 语音模型支持现状

### 2.1 Whisper
TensorRT-LLM 官方提供了 Whisper 的完整支持。
*   **Encoder**: 编译为 TensorRT Engine。
*   **Decoder**: 使用 TensorRT-LLM 的 GPT Manager 进行管理，支持 Beam Search 和 KV Cache。
*   **性能**: 相比 PyTorch 实现，吞吐量提升可达 5-10 倍。

### 2.2 Qwen-Audio
Qwen-Audio 本质上是 Qwen LLM + Audio Encoder。
*   **LLM 部分**: 完全支持（Qwen 结构已内置）。
*   **Audio Encoder**: 通常需要单独导出为 ONNX 或 TensorRT Engine，然后在推理流程中串联。

---

## 3. 选型建议

| 场景 | 推荐方案 | 理由 |
| :--- | :--- | :--- |
| **极致吞吐 (Server)** | **TensorRT-LLM** | 榨干 GPU 性能，支持 FP8/INT4 量化 |
| **快速验证 / 研究** | **HuggingFace** | 代码简单，生态丰富 |
| **端侧 / 跨平台** | **ONNX Runtime** | 不依赖 CUDA，兼容性强 |
| **多模态复杂流水线** | **vLLM** | 灵活性与性能的平衡，易于集成 |
