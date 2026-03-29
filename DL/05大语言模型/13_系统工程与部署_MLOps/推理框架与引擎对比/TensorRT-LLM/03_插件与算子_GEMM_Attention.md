# 插件与算子：GEMM 与 Attention 优化

TensorRT-LLM 的高性能主要源于其高度优化的**插件 (Plugins)**。这些插件封装了 NVIDIA 最新的 CUDA Kernel，专门解决 LLM 推理中的瓶颈。

## 1. 核心插件概览

在构建 Engine 时（`trtllm-build`），我们可以选择开启特定的插件。

| 插件名称 | 功能 | 适用场景 |
| :--- | :--- | :--- |
| **GEMM Plugin** | 矩阵乘法优化 | 所有全连接层 (Linear Layers) |
| **GPT Attention Plugin** | 自注意力机制优化 | Transformer Decoder (FlashAttention 2) |
| **Lookup Plugin** | Embedding 查找 | 输入层 |
| **Nccl Plugin** | 多卡通信 | Tensor Parallelism (TP) |

## 2. GPT Attention Plugin

这是对 Audio LLM 最重要的插件。

### 2.1 功能
它实现了 **FlashAttention-2** 和 **PagedAttention** 的融合算子。
*   **Memory Efficient**: 显存占用从 $O(N^2)$ 降低到 $O(N)$。
*   **Speed**: 极大提升了长序列（Long Context）的推理速度。

### 2.2 对语音模型的意义
语音识别（ASR）任务通常涉及长序列。
*   例如：一段 30 秒的音频在 Whisper 中对应 1500 个特征帧。
*   如果不开启 Attention Plugin，传统的 Softmax 计算会成为瓶颈。
*   **In-flight Batching**: 该插件支持在解码过程中动态插入新的请求，这对于实时语音流服务至关重要。

## 3. GEMM Plugin (General Matrix Multiply)

### 3.1 功能
利用 `cuBLASLt` 库进行矩阵乘法，支持 FP16, BF16, INT8, FP8 等多种精度。

### 3.2 自动调优
在构建 Engine 时，TensorRT 会进行 **Auto-tuning**：
*   它会尝试多种 GEMM 算法。
*   选择在当前 GPU 上速度最快的一种并“固化”到 Engine 中。
*   **注意**: 这就是为什么构建过程比较慢，且构建出的 Engine 不能跨显卡型号使用的原因。

## 4. 启用方式

在 `trtllm-build` 命令中显式开启：

```bash
trtllm-build \
    ... \
    --gemm_plugin float16 \
    --gpt_attention_plugin float16 \
    --context_fmha enable \
    --paged_kv_cache enable
```

*   `--context_fmha enable`: 开启 Context 阶段的 Fused Multi-Head Attention（处理 Prompt/Audio Encoder 输出）。
*   `--paged_kv_cache enable`: 开启分页 KV Cache（处理 Generation 阶段）。

## 5. 自定义插件 (Advanced)

对于特殊的 Audio Encoder 结构（如某些卷积下采样层），如果 TensorRT 原生不支持，可能需要编写自定义 Plugin。
*   **C++ API**: 继承 `nvinfer1::IPluginV2DynamicExt`。
*   **Python Binding**: 通过 Pybind11 暴露给 Python 构建脚本。
*   *大多数主流模型（Whisper, Qwen-Audio）目前已无需手写插件。*
