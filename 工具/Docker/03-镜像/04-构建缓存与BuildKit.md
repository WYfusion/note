# 构建缓存与 BuildKit (buildx)

## 适用场景
- 构建速度太慢，每次都要重新下载依赖（pip/npm/maven）。
- 需要构建多架构镜像（同时支持 x86 和 ARM/M1）。
- 构建时需要 SSH 密钥拉取私有代码，但不能把密钥泄露在镜像里。

## 什么是 BuildKit
BuildKit 是 Docker 的下一代构建引擎，比旧版构建器更快、更高效。
- **并发构建**：自动分析依赖，并行执行无关的步骤。
- **更强的缓存**：支持挂载缓存目录。
- **安全性**：支持 Secret 挂载。

**启用方式**：
Docker Desktop 默认已启用。Linux 上如果没启用，可以加环境变量：
```bash
DOCKER_BUILDKIT=1 docker build .
```

## 杀手级特性 1：挂载缓存 (`--mount=type=cache`)
这是 BuildKit 最实用的功能。它允许你在构建过程中挂载一个**持久化**的缓存目录，即使镜像层重建了，这个目录的内容还在。

### 场景：加速 pip 安装 (深度学习必备)
深度学习的包（torch, tensorflow）动辄几百 MB。如果 `requirements.txt` 变了，pip 默认会重新下载所有包。使用缓存挂载后，已下载的包可以直接复用。

```dockerfile
# Dockerfile 写法
# target 指向容器内 pip 的缓存目录
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt
```

### 场景：加速 Conda 安装
Conda 的包索引和 tarballs 也可以缓存。

```dockerfile
RUN --mount=type=cache,target=/opt/conda/pkgs \
    conda install -y numpy pandas
```

### 场景：加速 apt 安装
```dockerfile
# 缓存 /var/cache/apt
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && apt-get install -y gcc
```

## 杀手级特性 2：安全挂载密钥 (`--mount=type=secret`)
以前为了拉私有仓库代码，你可能把 SSH Key `COPY` 进镜像，这非常危险（Key 会留在历史层里）。
BuildKit 允许你临时挂载密钥，**用完即焚，不会写入镜像**。

### 1. Dockerfile 写法
```dockerfile
# 挂载 id 为 my_ssh_key 的密钥到指定路径
RUN --mount=type=secret,id=my_ssh_key,target=/root/.ssh/id_rsa \
    git clone git@github.com:myorg/private-model-repo.git
```

### 2. 构建命令
```bash
docker build --secret id=my_ssh_key,src=$HOME/.ssh/id_rsa .
```

## `docker buildx`：多平台构建
如果你用 M1/M2 Mac 开发，但服务器是 Linux x86，你需要构建多架构镜像。

### 1. 创建 builder 实例
```bash
docker buildx create --use
```

### 2. 构建并推送多架构镜像
注意：多架构镜像构建后**必须 push 到仓库**（或者 load 到本地，但 load 不支持多架构），不能直接保存在本地 images 列表里（除非指定单架构）。

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t myapp:latest \
  --push .
```

## 总结
- 只要能用 BuildKit 就一定用（现在基本是默认）。
- 善用 `--mount=type=cache`，构建速度能提升 10 倍。
- 涉及私有代码拉取，必须用 `--mount=type=secret`。

