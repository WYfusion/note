# 常见坑：版本兼容、CUDA 驱动与构建错误

TensorRT-LLM 以高性能著称，但其代价是**极高的环境敏感度**。

## 1. 版本地狱 (Dependency Hell)

TensorRT-LLM 与 TensorRT、CUDA、cuDNN 甚至 PyTorch 的版本通过“强绑定”关系连接。

### 1.1 严格对应关系
*   **TensorRT-LLM v0.8.0** 必须配合 **TensorRT 9.3.0**。
*   **TensorRT-LLM v0.5.0** 必须配合 **TensorRT 9.1.0**。
*   一旦版本不匹配，通常会报 `Symbol not found` 或 `Segmentation fault`。

### 1.2 解决方案
*   **使用官方 Docker 镜像**: 强烈建议不要在裸机上安装，直接使用 NVIDIA 提供的镜像。
    ```bash
    docker pull nvcr.io/nvidia/tensorrt-llm:0.8.0
    ```
*   **不要随意升级 pip 包**: 容器内的环境是经过验证的，`pip install --upgrade` 可能会破坏依赖。

## 2. 构建阶段 (Build) 常见错误

### 2.1 OOM (Out of Memory)
*   **现象**: `trtllm-build` 过程中进程被 Kill。
*   **原因**: 编译 Engine 需要大量内存（RAM）和显存。
*   **解决**:
    *   增加 Swap 空间。
    *   减少 `--max_batch_size` 或 `--max_input_len`。
    *   使用 `--workers 1` 限制并行编译线程数。

### 2.2 Unsupported Operator
*   **现象**: 报错 `IPluginV2DynamicExt not found` 或类似算子不支持。
*   **原因**: 模型结构太新，当前版本的 TRT-LLM 尚未支持。
*   **解决**:
    *   检查 TRT-LLM 的 `examples` 目录，确认模型是否在支持列表中。
    *   回退 HuggingFace 模型的版本（某些模型更新架构后会破坏兼容性）。

## 3. 运行时 (Runtime) 常见错误

### 3.1 CUDA Error: invalid argument
*   **原因**: 输入的 Tensor Shape 超过了构建时指定的 `max_input_len` 或 `max_batch_size`。
*   **语音模型特例**: 音频特征长度超过限制。例如构建时设为 3000 帧，但输入了 3001 帧。
*   **解决**: 重新构建 Engine，增大限制；或者在预处理阶段截断音频。

### 3.2 Output is garbage (乱码)
*   **原因**:
    *   EOS Token ID 设置错误。
    *   量化校准数据集分布与真实数据差异过大。
    *   **RoPE Scaling** 参数未正确传递。

## 4. 调试技巧

*   **Verbose Log**: 开启详细日志以定位 Crash 位置。
    ```bash
    export TLLM_LOG_LEVEL=VERBOSE
    ```
*   **Polygraphy**: 使用 NVIDIA 的 Polygraphy 工具对比 PyTorch 和 TensorRT 的逐层输出，定位精度损失层。
