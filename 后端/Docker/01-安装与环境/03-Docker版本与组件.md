# Docker 版本与组件（Engine / CLI / Compose / Buildx）

> 目标：让你能一句话回答“我现在用的 Docker 到底是哪一套”，并且出了问题知道该看哪里。

## 适用场景
- Windows/Mac 装了 Docker Desktop，但不知道它和 Linux 上的 Docker Engine 有啥区别。
- `docker`、`docker compose`、`docker buildx` 到底分别是什么？
- 排障时要看哪些版本信息、哪些字段最关键。

## 一张脑图（文字版）
### 最常见的两类环境
1) **Docker Desktop（Windows/Mac 常见）**
- 你安装的是一个“整合套件”：UI + Docker Engine（跑在 WSL2/虚拟机里）+ CLI 插件（compose/buildx）。

2) **Docker Engine（Linux 常见）**
- 你安装的是 `dockerd`（daemon）+ `docker`（CLI）等组件。

## 关键组件逐个说明
### 1) Docker CLI（你敲的 `docker` 命令）
- 作用：作为客户端，向 Docker daemon（`dockerd`）发送请求。
- 表现：你在终端里执行 `docker ps`、`docker run` 其实是在调用 API。

### 2) Docker Engine / dockerd（真正干活的服务端）
- 作用：管理镜像、创建容器、网络、volume。
- 常见排障：
	- `Cannot connect to the Docker daemon` 通常就是 daemon 不可用或 socket 权限问题。

### 3) docker compose（Compose v2 插件）
- 现在新的 Compose 通常是：
	- 命令形态：`docker compose ...`（注意中间是空格，不是破折号）
	- 它是 docker CLI 的一个插件（v2）。

> 旧版 Compose（v1）曾经是 `docker-compose`（单独二进制）。现在多数环境推荐 v2。

### 4) BuildKit（新一代构建后端）
- 作用：更快的构建、更好的缓存、更安全的 secret/ssh、支持高级 `RUN --mount=...` 等。
- 常见开关：`DOCKER_BUILDKIT=1`（有些环境默认已启用）。

### 5) buildx（BuildKit 的前端/多平台构建工具）
- 命令：`docker buildx ...`
- 典型用途：
	- 多平台镜像（amd64/arm64）
	- 远端 builder
	- 缓存导入导出

### 6) containerd / runc（运行时栈：把“容器”真正跑起来）
- 粗略理解：
	- `dockerd` 管理生命周期
	- 底层通过 `containerd` 管理运行时细节
	- 最终由 `runc` 依据 OCI 规范启动容器进程

> 你不需要每天都碰它们，但当你进入 K8s/云原生时会频繁看到 containerd。

## 我应该用哪些命令“确认我在用什么”
### 1) `docker version`
用途：一眼看出 Client/Server 是否都正常。

关注点：
- 是否能看到 **Server** 版本信息
- API version 是否匹配（一般无需手动干预）

### 2) `docker info`
用途：环境“体检报告”。

重点字段（建议你记住）：
- `Server Version`
- `Operating System`（Desktop 下通常会看到 Linuxkit/WSL2 相关信息）
- `Cgroup Version`（Linux 上很关键）
- `Docker Root Dir`（磁盘占用排查）
- `Registry Mirrors`（镜像源是否生效）
- `HTTP Proxy/HTTPS Proxy/No Proxy`（代理是否生效）

### 3) `docker compose version`
用途：确认 compose v2 是否存在、版本是多少。

### 4) `docker buildx version` / `docker buildx ls`
用途：确认 buildx 是否可用、当前 builder 是谁。

## 常见现象 → 该怀疑哪个组件
- `docker` 命令存在，但 `Server` 显示不出来：
	- Desktop 没启动 / Engine 不可用 / socket 权限
- `docker compose` 不能用：
	- compose 插件缺失或版本过旧
- 构建很慢、缓存不生效：
	- BuildKit/buildx 配置未启用或缓存策略不对（见 `../03-镜像/04-构建缓存与BuildKit.md`）

## 例子：最小“环境检查”流程
> 你可以把它当作装完 Docker 后的验收顺序。

1. `docker version`
2. `docker info`
3. `docker run --rm hello-world`
4. `docker compose version`
5. `docker buildx version`

## 进阶：和 Docker Desktop 的关系（Windows/WSL2）
- 你在 Windows PowerShell 运行的 `docker` 依然是 CLI。
- 真实的 daemon 多半运行在 WSL2 的 Linux 环境里。
- 因此：
	- Desktop 的代理/镜像源设置会影响 `docker pull`
	- WSL2 自己的网络与证书也可能影响某些场景

