# OCI 与运行时（runc / containerd）

> 目标：你不需要手写 OCI 配置文件，但你需要理解“Docker 为什么能在不同平台跑起来”，以及排障时该看哪一层。

## 适用场景
- 你想理解 Docker 背后的“容器运行时栈”。
- 你遇到容器启动/构建问题，希望知道是 CLI、daemon、运行时还是镜像的问题。
- 你准备从 Docker 迁移到 K8s/containerd，想建立正确的概念映射。

## 一句话概括
- **OCI**：一组“容器怎么打包（镜像）”和“容器怎么启动（运行时）”的标准。
- **runc**：一个实现 OCI Runtime 的工具（真正 fork/exec 出容器进程）。
- **containerd**：更上层的容器运行时管理者（生命周期、镜像、快照等）。
- **dockerd**：Docker 的 daemon，提供 Docker API，并管理网络/volume/构建等。

## 先区分两个“规范”
### 1) OCI Image Spec（镜像规范）
规定：
- 镜像里有哪些层（layers）
- 配置里怎么描述入口命令、环境变量、工作目录等
- 如何通过 digest 校验内容

你平时看到的：
- `nginx:1.27`（tag）
- `sha256:...`（digest）

### 2) OCI Runtime Spec（运行时规范）
规定：
- 启动容器时需要哪些配置（namespace/cgroup/mount/capabilities 等）
- 最终如何把容器进程启动起来

## 运行时链路：docker -> containerd -> runc（5 句话版）
1. 你执行 `docker run ...`，这是 **docker CLI** 发请求。
2. 请求发给 **dockerd**（Docker daemon），它负责创建容器需要的元数据、网络、挂载等。
3. dockerd 通常把“实际运行容器”的任务交给 **containerd**。
4. containerd 再调用符合 OCI 的低层运行时（常见是 **runc**）去启动进程。
5. runc 根据 OCI runtime spec 配置 namespace/cgroup/mount，然后启动你的进程（比如 nginx）。

> 你看到“容器里 PID=1 的进程”就是最后这一步启动出来的。

## containerd 在做什么（你需要知道的程度）
你可以把 containerd 理解为：
- 更通用的“容器生命周期管理组件”
- 管理镜像内容与快照（snapshot）
- 负责启动/停止容器任务（task）

K8s 里你经常会看到它，是因为：
- K8s 最常用的运行时栈就是 containerd（不依赖 Docker Desktop UI）。

## runc 在做什么（最核心）
你可以把 runc 理解为：
- 最终把容器进程跑起来的人
- 它会做：namespace 隔离、cgroup 限制、mount 挂载、capabilities 等

## shim / snapshotter（概念级，知道名字足够）
- **shim**：运行时的“中间层/守护”组件，保证容器进程与上层管理解耦。
- **snapshotter**：镜像层到真正可读写文件系统的实现（overlayfs 等）。

## 怎么用 docker 命令观察“我现在的运行时大概是什么”
### 1) `docker info`
你经常能看到一些线索，例如：
- `Cgroup Version`（Linux 上很重要）
- 存储驱动（overlay2 等）
- Desktop/WSL2 场景的 Operating System 描述

### 2) `docker version`
确认 Client/Server 是否正常。

### 3) Desktop/WSL2 场景特别注意
Windows 上运行 Docker Desktop 时：
- 你的 CLI 在 Windows
- daemon/运行时通常在 WSL2 的 Linux 环境

于是很多问题的排查会分成两段：
- Windows：CLI 能否连上 daemon
- WSL2：daemon 内部的网络/证书/文件系统表现

## 常见问题：应该怀疑哪一层
### 1) `docker` 命令本身不存在
怀疑：CLI 没装/路径问题。

### 2) 只能看到 Client，看不到 Server
怀疑：dockerd 不可用 / Desktop 没启动 / 权限问题。

### 3) 容器启动报权限/资源/隔离相关错误
怀疑：更靠近运行时/内核能力（namespace/cgroup/capabilities）。

### 4) K8s 环境里说 “用的是 containerd”
这不意味着你不能用镜像：
- OCI 镜像是通用格式
- 只是管理入口从 Docker CLI 变成了 containerd 的工具链

## 下一步建议
- 想理解隔离与资源限制：看 `03-命名空间与Cgroups速记.md`
- 想理解镜像规范与 digest：看 `../03-镜像/05-镜像标签与版本策略.md`
- 想理解与 K8s 的概念映射：看 `../11-与K8s云原生衔接（可选）/01-Docker到Kubernetes概念映射.md`

