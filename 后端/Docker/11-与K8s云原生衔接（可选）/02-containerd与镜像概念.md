# containerd 与镜像概念

## 适用场景
- 现代 Kubernetes (v1.24+) 已经弃用了 Docker Shim，直接使用 containerd 作为运行时。
- 调试 K8s 节点上的容器问题时，可能无法使用 `docker` 命令。

## 1. 什么是 containerd？
Docker 实际上是一个庞大的工具集（CLI + API + BuildKit + Network + Volume）。
而 **containerd** 是 Docker 剥离出来的核心运行时，只负责：
- 拉取镜像 (Pull)
- 管理容器生命周期 (Start/Stop)
- 管理存储 (Snapshotter)

K8s 为了轻量化，现在直接调用 containerd，绕过了 Docker Daemon。

## 2. 常用命令对照 (ctr vs crictl vs nerdctl)

在 K8s 节点上，你可能找不到 `docker` 命令，取而代之的是：

| 场景 | Docker 命令 | containerd (ctr) | Kubernetes (crictl) | nerdctl (推荐) |
| :--- | :--- | :--- | :--- | :--- |
| **定位** | 开发者友好 | 底层调试工具，难用 | K8s 专用调试工具 | Docker 兼容 CLI |
| **列出容器** | `docker ps` | `ctr c ls` | `crictl ps` | `nerdctl ps` |
| **列出镜像** | `docker images` | `ctr i ls` | `crictl images` | `nerdctl images` |
| **拉取镜像** | `docker pull` | `ctr i pull` | `crictl pull` | `nerdctl pull` |
| **查看日志** | `docker logs` | (不支持) | `crictl logs` | `nerdctl logs` |
| **进入容器** | `docker exec -it` | (复杂) | `crictl exec -it` | `nerdctl exec -it` |

## 3. 命名空间 (Namespaces)
Docker 默认使用 `moby` 命名空间（你看不到这个概念，因为它是隐藏的）。
而 K8s 使用 `k8s.io` 命名空间。

如果你在 K8s 节点上运行 `ctr images ls` 发现是空的，可能是没指定命名空间：
```bash
# 查看 K8s 的镜像
ctr -n k8s.io images ls
```

## 4. 镜像存储 (Snapshotter)
containerd 支持多种存储驱动。
- **Overlayfs**: 类似 Docker 的 overlay2，最常用。
- **Stargz**: 谷歌推出的“懒加载”镜像格式，允许容器在镜像没下载完时就启动（按需拉取文件），极大加速大规模扩容。


## TODO
- [ ] 用对照表说明 Docker vs containerd
