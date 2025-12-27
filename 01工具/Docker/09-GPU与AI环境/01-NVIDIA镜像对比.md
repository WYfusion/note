# NVIDIA 镜像类型对比 (Base vs Runtime vs Devel)

## 适用场景
- 纠结 `FROM nvidia/cuda:xxx` 到底该选哪个 Tag。
- 镜像体积优化（从 10GB 减到 4GB）。
- 解决 "nvcc not found" 或 "missing cuda.h" 报错。

## 1. 三大核心 Tag 对比
NVIDIA 官方镜像 (`nvidia/cuda`) 通常有三种后缀：`base`, `runtime`, `devel`。

| 类型 | 包含内容 | 体积 (估算) | 适用场景 | 典型报错 |
| :--- | :--- | :--- | :--- | :--- |
| **base** | 仅包含 CUDA 运行时依赖 (`libcudart.so`)。**不含 CuDNN**。 | 最小 (~100MB) | 部署已编译好的二进制程序 (如 TensorRT 引擎)。 | `ImportError: libcudnn.so.8: cannot open shared object file` |
| **runtime** | 包含 CUDA 运行时 + **CuDNN** + Python (部分)。 | 中等 (~1-2GB) | **深度学习推理/训练** (只要不涉及编译自定义算子)。 | `nvcc: command not found` |
| **devel** | 包含 **runtime** + **编译器 (nvcc)** + 头文件 (`cuda.h`) + 调试工具。 | 最大 (~4-6GB) | **开发环境**、编译自定义 CUDA 算子 (如 FlashAttention, Deformable Detr)。 | 无 (最全) |

## 2. 如何选择？

### 场景 A：纯跑 PyTorch/TensorFlow (推荐 Runtime)
如果你只是 `pip install torch` 然后跑代码，通常 **Runtime** 就够了。
因为 PyTorch 的 Wheel 包里已经自带了 CUDA 运行时和 CuDNN (这也是为什么 torch 包那么大)。
*甚至，你可以直接用 `python:3.9-slim` 做基础镜像，完全依赖 pip 安装的 torch 里的 cuda 库（前提是宿主机驱动兼容）。*

### 场景 B：需要编译扩展 (推荐 Devel)
如果你安装的库需要现场编译 CUDA 代码，例如：
- `pip install flash-attn`
- `pip install mamba-ssm`
- `pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"`
这些库在安装时会寻找 `nvcc`。如果你用 Runtime 镜像，会报错。**必须用 Devel**。

### 场景 C：生产环境部署 (推荐 Base/Runtime)
为了减小镜像体积和攻击面，生产环境应使用 Multi-stage 构建：
1. 在 `devel` 镜像里编译好 wheel 包。
2. 复制 wheel 包到 `runtime` 或 `base` 镜像里安装。

## 3. 常用镜像 Tag 示例
```dockerfile
# 1. 开发环境 (最稳妥，但体积大)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 2. 推理环境 (体积小)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 3. 极致精简 (仅限已编译好的二进制)
FROM nvidia/cuda:11.8.0-base-ubuntu22.04
```

> [!TIP] 坑点提示
> `pytorch/pytorch` 官方镜像通常基于 `nvidia/cuda:runtime` 或 `devel`。
> 如果你用 `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime`，里面是没有 `nvcc` 的。


#### 2. runtime（运行时版本）
- 包含内容：在base基础上增加了CUDA数学库（如cuBLAS、cuSOLVER等）
- 大小：中等
- 适用场景：运行已编译好的CUDA应用程序
- 特点：无需编译环境，只提供运行支持

#### 3. devel（开发版本）
- 包含内容：在runtime基础上增加了CUDA头文件、开发工具、编译器（如nvcc）
- 大小：较大
- 适用场景：开发和编译CUDA应用程序
- 特点：完整的CUDA开发环境，可以在容器内编译代码

#### 4. cudnn-runtime（cuDNN运行时版本）
- 包含内容：在runtime基础上增加了cuDNN运行库
- 大小：中等到较大
- 适用场景：运行使用cuDNN的深度学习应用程序
- 特点：针对深度学习优化，无需编译环境

#### 5. cudnn-devel（cuDNN开发版本）
- 包含内容：在devel基础上增加了cuDNN开发库和头文件
- 大小：最大
- 适用场景：开发和编译使用cuDNN的深度学习应用程序
- 特点：最完整的深度学习开发环境

#### 容器大小比较
一般来说：base < runtime < cudnn-runtime < devel < cudnn-devel

#### 选择建议
1. 仅运行预编译应用：选择runtime或cudnn-runtime
2. 开发CUDA应用但不使用深度学习：选择devel
3. 开发深度学习应用：选择cudnn-devel
4. 资源有限且只需基本CUDA功能：选择base
5. 生产环境部署：通常选择runtime或cudnn-runtime以减小容器大小

这些镜像都基于Ubuntu或Centos等基础操作系统，NVIDIA还提供不同CUDA版本和操作系统组合的镜像，可以根据具体需求选择适合的版本组合。



## UBI8操作系统介绍
UBI8（Universal Base Image 8）是Red Hat提供的一种通用基础容器镜像，它具有以下特点：
#### UBI8基本信息
- 全称：Red Hat Universal Base Image 8
- 基于：Red Hat Enterp
- 提供方：Red Hat公司
- 许可证：免费使用和再分发（不需要Red Hat订阅）
#### UBI8的主要特点
- 企业级品质：继承了RHEL的稳定性和安全性
- 免费使用：不像完整版RHEL那样需要付费订阅
- 定期安全更新：由Red Hat提供安全更新和补丁
- 容器优化：专为容器环境设计和优化
- 开发友好：包含常用开发工具和库
- 生产就绪：适合用于生产环境
#### UBI8版本变体
UBI8提供了几种不同的变体以满足不同需求：
1. 标准版(Standard)：包含完整的软件包集
2. 最小版(Minimal)：极小的镜像大小，适合轻量级应用
3. 初始版(Init)：包含systemd，支持运行系统服务
4. 运行时(Runtime)：针对特定语言优化的运行时环境