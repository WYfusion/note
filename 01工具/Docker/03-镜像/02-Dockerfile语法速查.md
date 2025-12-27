# Dockerfile 语法速查

## 适用场景
- 写 Dockerfile 时快速查指令含义与常用写法。
- 搞不清 CMD 和 ENTRYPOINT 的区别。
- 需要标准的语言构建模板。

## 核心指令详解

### 1. FROM（指定基础镜像）
必须是第一条指令（ARG 除外）。
```dockerfile
FROM ubuntu:22.04
FROM python:3.9-slim
FROM node:18-alpine
```

### 2. WORKDIR（工作目录）
相当于 `cd`，但比 `cd` 安全（如果目录不存在会自动创建）。
**强烈建议使用绝对路径**。
```dockerfile
WORKDIR /app
# 后面的 RUN, CMD, COPY 都会在这个目录下执行
```

### 3. COPY vs ADD（复制文件）
> **结论**：99% 的情况用 `COPY`。

- **COPY**：单纯复制本地文件到镜像。
  ```dockerfile
  COPY requirements.txt .
  COPY src/ ./src/
  ```
- **ADD**：功能更强但有副作用。
  - 如果源文件是 `tar.gz`，ADD 会自动解压（有时你并不想解压）。
  - 支持 URL 下载（不推荐，建议用 curl/wget）。

### 4. RUN（构建时执行）
在 `docker build` 阶段执行，每条 RUN 都会产生一个新的镜像层。
**最佳实践**：合并命令，清理缓存。
```dockerfile
RUN apt-get update && apt-get install -y \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*
```

### 5. CMD vs ENTRYPOINT（启动命令）
> **结论**：推荐 `ENTRYPOINT` 设为固定程序，`CMD` 设为默认参数。

- **CMD**：容器启动时的默认命令，**容易被 `docker run` 后面的参数覆盖**。
- **ENTRYPOINT**：容器启动时的入口程序，**不会被覆盖**（除非用 `--entrypoint`），`docker run` 后面的参数会拼接到它后面。

**最佳搭配示例**：
```dockerfile
ENTRYPOINT ["python", "app.py"]
CMD ["--help"]
```
- 跑 `docker run myimg` -> 执行 `python app.py --help`
- 跑 `docker run myimg --port 80` -> 执行 `python app.py --port 80`

### 6. ENV vs ARG（变量）
- **ENV**：环境变量。**构建时有效，运行时也有效**（容器里能读到）。
  ```dockerfile
  ENV APP_ENV=production
  ```
- **ARG**：构建参数。**只在构建时有效**，容器运行时消失。
  ```dockerfile
  ARG VERSION=1.0
  RUN echo "Building version $VERSION"
  ```

### 7. EXPOSE（声明端口）
**仅声明**，不会自动端口映射。
```dockerfile
EXPOSE 8080
```
作用是给看 Dockerfile 的人，或者 `docker run -P`（随机端口）时用。真正映射还得靠 `docker run -p 8080:8080`。

### 8. USER（切换用户）
为了安全，不要一直用 root。
```dockerfile
RUN useradd -m myuser
USER myuser
```

### 9. HEALTHCHECK（健康检查）
告诉 Docker 怎么判断容器是不是“活”的。
```dockerfile
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost/ || exit 1
```

## 推荐模板

### 1. Python (Flask/Django)
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 先拷依赖文件（利用缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再拷源码
COPY . .

ENV PORT=8000
EXPOSE 8000

CMD ["python", "app.py"]
```

### 2. Node.js (Express)
```dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .

EXPOSE 3000
USER node

CMD ["node", "index.js"]
```

### 3. 深度学习开发环境 (CUDA + Conda)
```dockerfile
# 1. 基础镜像：选择 NVIDIA 官方 CUDA 镜像
FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 2. 环境变量：防止交互式提示，设置时区
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai

# 3. 系统依赖：换源并安装基础工具
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends \
       wget git vim libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. 安装 Miniconda
ENV PATH=/opt/conda/bin:$PATH
RUN wget --quiet https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh \
    && conda init bash

# 5. 创建环境并安装 PyTorch
# 使用 SHELL 指令激活 conda 环境
SHELL ["/bin/bash", "--login", "-c"]
RUN conda create -n myenv python=3.9 -y \
    && conda activate myenv \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    && conda clean -ya

# 6. 设置入口
CMD ["/bin/bash"]
```

