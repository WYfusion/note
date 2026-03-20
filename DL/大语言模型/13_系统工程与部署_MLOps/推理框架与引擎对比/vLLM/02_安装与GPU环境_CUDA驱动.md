# 安装与 GPU 环境配置

vLLM 对 CUDA 版本和 PyTorch 版本有严格要求，因为它包含大量编译好的 CUDA Kernel。

## 1. 快速安装

### 1.1 Pip 安装
vLLM 提供了预编译的 Wheel 包。

```bash
# 务必确保 CUDA 版本匹配 (通常支持 CUDA 12.1)
pip install vllm
```

### 1.2 验证安装
```bash
python -c "import vllm; print(vllm.__version__)"
```

---

## 2. 环境依赖

### 2.1 CUDA 驱动
*   **推荐**: CUDA 12.1 或更高。
*   **检查**: 运行 `nvidia-smi` 查看 Driver Version 和 CUDA Version。
*   **兼容性**: 如果宿主机的 Driver 太旧（如不支持 CUDA 12），vLLM 将无法启动。

### 2.2 PyTorch
vLLM 通常依赖最新稳定版的 PyTorch（如 2.1.2 或 2.3.0）。安装 vLLM 时会自动拉取对应的 PyTorch 版本。

---

## 3. Docker 部署 (推荐)

为了避免环境冲突（特别是 CUDA 库版本不一致），强烈推荐使用官方 Docker 镜像。

```bash
# 拉取镜像
docker pull vllm/vllm-openai:latest

# 启动容器
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen-Audio-Chat
```

### 3.1 参数解释
*   `--runtime nvidia --gpus all`: 启用 GPU 支持。
*   `--ipc=host`: 必须开启！vLLM 使用共享内存进行多进程通信（特别是多卡推理时），默认的 Docker SHM 大小（64MB）不够用。
*   `-v ...`: 挂载 HuggingFace 缓存目录，避免重复下载模型。

---

## 4. 常见报错

### 4.1 `CUDA error: no kernel image is available for execution on the device`
*   **原因**: 安装的 vLLM 编译版本与当前显卡的计算能力（Compute Capability）不匹配。例如在 V100 (sm_70) 上运行了为 A100 (sm_80) 编译的包。
*   **解决**: 重新安装对应版本的 vLLM 或从源码编译。

### 4.2 `ImportError: libcudart.so.12: cannot open shared object file`
*   **原因**: 系统缺少 CUDA 运行时库。
*   **解决**: 安装 CUDA Toolkit 或使用 Docker。
