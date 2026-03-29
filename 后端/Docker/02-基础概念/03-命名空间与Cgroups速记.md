# 命名空间与 Cgroups 速记

## 适用场景
- 为什么容器能“隔离”？为什么要做资源限制？

## 目录
- Namespaces：pid/net/mnt/uts/ipc/user
- Cgroups：CPU/Memory/IO/PIDs
- cgroups v1 vs v2（概念）

## 先记住两句话
- **Namespace** 负责“看起来像隔离”（你在容器里看到的进程/网络/主机名像是独立的）。
- **Cgroups** 负责“用起来被限制”（CPU/内存/进程数/IO 不会无限制抢占宿主机）。

## Windows（Docker Desktop/WSL2） vs Linux 原生：先消除一个常见误解
- **Linux 原生（Docker Engine）**：容器直接跑在这台 Linux 的内核上，namespace/cgroups 就是这台机器内核提供的隔离与限制能力。
- **Windows + Docker Desktop（默认 Linux containers）**：容器实际跑在一个 Linux 环境里（通常是 WSL2 发行版或轻量 VM）。
	- 这意味着：你在容器里看到的 namespace/cgroups，是“WSL2/VM 里的 Linux 内核”的结果。
	- 一些资源上限会受到 Desktop/WSL 的额外约束（例如 Desktop 设置里的 CPU/Memory/Swap 上限）。

你可以把它理解成：
$$\text{Windows 宿主机} \rightarrow \text{WSL2/VM(运行 dockerd)} \rightarrow \text{容器(Namespaces/Cgroups)}$$

实用结论：
- 在 Windows 上做资源限制实验时，先确认 **Docker Desktop 的资源设置**，否则 `--memory/--cpus` 的现象可能会被“外层限制”干扰。
- `-v`/`--mount` 在 Windows 上的行为，除了 mnt namespace 之外，还会受 **路径转换、权限、文件系统类型（NTFS/\$\{wsl\}）** 影响（详见 `../06-存储/03-Windows路径挂载注意事项.md`）。

## Namespaces（隔离的来源）
> 直觉：namespace 让同一台机器上的不同进程组看到“不同的世界”。

### pid namespace（进程隔离）
现象：
- 你在容器里看到的 PID 1（通常是你的主进程），不等于宿主机的 PID。

平台提示：
- **Linux 原生**：宿主机上用 `ps -ef`/`top` 能更直接地把“某个容器对应哪些进程”串起来看。
- **Windows + Desktop**：你在 Windows 进程列表里看不到容器 PID 的直接对应；要观察更“宿主机视角”的进程，需要进入 WSL2/VM 那一层。

排查价值：
- “容器为什么退出？”通常看 PID 1 进程是否退出。

### net namespace（网络隔离）
现象：
- 容器有自己的网卡、路由、iptables 视图（取决于网络模式）。

注意：
- 端口映射 `-p` 是把宿主机端口转发到容器网络命名空间里。

平台提示：
- **Linux 原生**：`-p` 常见影响点是宿主机防火墙（iptables/nftables）与监听地址（0.0.0.0/127.0.0.1）。
- **Windows + Desktop**：除了容器自身网络，还叠加了 Windows 防火墙与 Desktop 的端口转发链路；“容器里能访问、宿主机能访问、局域网不能访问”更常见。

### mnt namespace（挂载/文件系统视图隔离）
现象：
- 容器里看到的挂载点（mount）与宿主机不同。

注意：
- `-v`/`--mount` 其实就是在容器的 mnt namespace 里追加挂载。

平台提示：
- **Linux 原生**：挂载失败更多是权限/SELinux/AppArmor/路径不存在等问题。
- **Windows + Desktop**：常见问题集中在路径写法、盘符共享、大小写、CRLF、以及“Windows 文件系统与 Linux 权限语义差异”。优先看 `../06-存储/03-Windows路径挂载注意事项.md`。

### uts namespace（主机名隔离）
现象：
- 容器有自己的 hostname。

常用参数：
- `--hostname mybox`

### ipc namespace（进程间通信隔离）
现象：
- 影响共享内存、信号量等 IPC 资源的隔离。

### user namespace（用户映射隔离）
现象：
- 容器内 root 不一定等价于宿主机 root（取决于是否启用 userns remap/rootless）。

平台提示：
- **Linux 原生**：userns remap / rootless 是常见安全增强手段。
- **Windows + Desktop**：你通常接触到的是“容器内 root vs WSL2/VM 内核视角”的差异；挂载到 Windows 目录时还会叠加文件权限映射行为。

## Cgroups（资源限制的来源）
> 直觉：cgroups 让你能对一组进程“分配配额”。

### 1) CPU 限制
#### `--cpus`
限制容器最多使用多少个 CPU 核的算力（不是绑定具体核）：

```bash
docker run --rm --cpus=1.5 ubuntu bash
```

平台提示：
- **Windows + Desktop**：最终可用 CPU 上限受 Desktop/WSL2 分配的 CPU 数影响。

#### `--cpuset-cpus`
绑定可用 CPU 核（用于做隔离/性能实验）：

```bash
docker run --rm --cpuset-cpus="0,1" ubuntu bash
```

注意：
- `--cpus` 是“配额”，`--cpuset-cpus` 是“可用核集合”，可以叠加。

### 2) 内存限制
#### `--memory`

```bash
docker run --rm --memory=512m ubuntu bash
```

常见坑：
- 容器超限会触发 OOM（Out Of Memory），进程可能被 kill。
- 排障建议结合 `docker logs` / `docker inspect`（看退出码、OOMKilled 等）。

平台提示：
- **Windows + Desktop**：如果 Desktop 给 WSL2/VM 的内存上限较小，你可能会看到“还没到 `--memory` 就整体变慢/触顶”的现象；先从 Desktop 的 Resources 设置排查。

### 3) 进程数限制（防 fork 炸机）
#### `--pids-limit`

```bash
docker run --rm --pids-limit=256 ubuntu bash
```

适用：
- 多租户机器、CI runner、避免 fork bomb。

### 4) IO 限制（概念）
Docker 也有一些块设备 IO 的限制参数，但不同存储驱动/内核下表现差异较大。
如果你主要是开发/常规部署，优先掌握 CPU/内存/PIDs 就够用了。

## 一个深度学习/多进程高频坑：共享内存（不是 cgroup memory）
很多数据加载（例如 PyTorch DataLoader 多进程）会用到 `/dev/shm`。

### `--shm-size`

```bash
docker run --rm --shm-size=8g -it ubuntu bash
```

注意：
- Docker 默认的 shm 很小（常见 64MB），会导致“看似内存够但仍然报错”。
- 这和 `--memory` 不完全是一回事（一个是 shm 挂载大小，一个是容器内存上限）。

平台提示：
- **Windows + Desktop**：`/dev/shm` 仍然发生在“WSL2/VM 的 Linux 内核”里；如果你在 Windows 宿主机感觉资源很富余，但容器仍频繁 shm 报错/卡住，优先检查 `--shm-size` 以及 Desktop 资源上限。

## 与常用 docker 参数的对应关系（速查）
- CPU：`--cpus`、`--cpuset-cpus`
- 内存：`--memory`
- 进程数：`--pids-limit`
- 共享内存：`--shm-size`

## cgroups v1 vs v2（概念）
- v2 是更统一的新接口，许多新系统默认使用 v2。
- 你通常只需要知道它影响“某些参数行为/统计口径”，真正排障时再深入。

## 下一步建议
- 实战资源限制：见 `../04-容器/03-资源限制与隔离.md`
- 网络基础：见 `../05-网络/01-Docker网络模型.md`

