# NVIDIA 容器运行原理

## 适用场景
- 理解为什么容器里不需要装显卡驱动。
- 解决 "CUDA version mismatch" 或 "Driver too old" 问题。
- 理解 `--gpus` 参数背后的黑魔法。

## 1. 核心架构：驱动在宿主机，CUDA 在容器
这是新手最容易混淆的概念。

| 组件 | 位置 | 描述 |
| :--- | :--- | :--- |
| **NVIDIA Driver** (`.ko`) | **宿主机 (Host)** | 内核态驱动。**容器无法包含驱动**，必须复用宿主机的。 |
| **CUDA Toolkit** (`libcudart.so`) | **容器 (Container)** | 用户态库。包含在 Docker 镜像里 (如 `nvidia/cuda`)。 |
| **NVIDIA Container Toolkit** | **宿主机 (Host)** | 桥梁。负责在启动容器时，把宿主机的驱动文件挂载进容器。 |

> [!IMPORTANT] 兼容性原则
> **宿主机驱动版本 >= 容器内 CUDA 版本要求**。
> 例如：宿主机驱动是 470 (支持 CUDA 11.4)，你想跑 CUDA 12.1 的容器 -> **不行**，必须升级宿主机驱动。
> 反之：宿主机驱动 535 (支持 CUDA 12.2)，跑 CUDA 11.8 的容器 -> **可以** (向下兼容)。

## 2. `--gpus` 到底做了什么？
当你执行 `docker run --gpus all ...` 时，Docker 调用了 `nvidia-container-runtime`，它执行了以下操作：

1.  **设备挂载**: 将 `/dev/nvidia0`, `/dev/nvidiactl` 等设备文件挂载到容器内。
2.  **驱动注入**: 将宿主机的驱动库 (如 `libcuda.so`, `libnvidia-ml.so`) 挂载到容器的 `/usr/lib/x86_64-linux-gnu/`。
3.  **工具注入**: 挂载 `nvidia-smi` 等工具（视配置而定）。

这就是为什么你在容器里能看到 `nvidia-smi`，但找不到驱动安装包的原因。

## 3. 常见报错与排查

### 报错 1: `docker: Error response from daemon: could not select device driver ""`
**原因**: 未安装 `nvidia-container-toolkit` 或者 Docker 未重启。
**解决**:
```bash
# Ubuntu 安装
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 报错 2: `CUDA driver version is insufficient for CUDA runtime version`
**原因**: 宿主机驱动太老，容器内 CUDA 太新。
**解决**: 升级宿主机驱动，或者换一个旧版 CUDA 的镜像。

### 报错 3: `Found no NVIDIA driver on your system`
**原因**: 忘记加 `--gpus all` 参数，或者宿主机驱动挂了。
**验证**:
```bash
# 在宿主机跑
nvidia-smi
# 在容器里跑
docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```

## 4. 环境变量控制
你可以通过环境变量控制可见性（虽然现在推荐用 `--gpus`）：

- `NVIDIA_VISIBLE_DEVICES=0,1`: 仅暴露前两张卡。
- `NVIDIA_DRIVER_CAPABILITIES=compute,utility`: 仅暴露计算和管理功能（不暴露图形显示）。


## TODO
- [ ] 记录驱动/CUDA 兼容基本规则
