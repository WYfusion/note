## 前置知识

> [!important]
> 
> 阅读本页前建议先读：[[0 总体定位：建立正确心智模型]]——建立 Image / Container / Volume / Network 的整体概念框架。

---

## 0. 定位

> 深入剖析 Image / Container / Docker daemon 三个基础对象的本质、关键属性、生命周期与常见误区，为 Dockerfile（§2）和 Compose（§3）的学习打下坚实的对象模型基础。

---

## 1. 镜像（Image）

### 1.1 本质

> [!important]
> 
> **镜像（Image）** 是一个**只读的、分层的文件系统快照**，附带元数据（环境变量、启动命令、暴露端口等）。它是 `docker build` 的产物，也是创建容器的唯一模板。

直觉类比：镜像就像一份「已编译好的安装光盘」——内容固定，你可以用它在任意机器上创建运行实例，但光盘本身不会被修改。

### 1.2 分层存储机制（Layered Storage）

镜像由多个**只读层（Read-only Layer）**叠加而成，每一条 Dockerfile 指令生成一层：
![[2026-04-18 20.36.05docker分层存储.excalidraw|200]]

**层的核心特性**：

- **内容寻址（Content Addressable Storage）**：每层通过 SHA256 哈希标识；内容相同的层在磁盘上只存一份

- **共享复用**：如果 10 个镜像都基于 `debian:bookworm-slim`，这个基础层只占一份磁盘空间

- **写时复制（Copy-on-Write, CoW）**：容器运行时在只读层之上叠加可写层，只有被修改的文件才会被复制到可写层

### 1.3 关键属性

|属性|作用|关键认知|
|---|---|---|
|`ENTRYPOINT`|容器启动时**固定执行**的命令前缀|通常设为主进程入口，如 `python`、`node`、`/app/server`|
|`CMD`|容器启动时的**默认参数**（可被 `docker run` 覆盖）|与 ENTRYPOINT 配合：`ENTRYPOINT` 定义「做什么」，`CMD` 定义「默认怎么做」|
|`ENV`|镜像内置的默认环境变量|会进入运行时；运行时可通过 `-e` 或 Compose `environment` 覆盖|
|`EXPOSE`|声明容器监听的端口（**仅元数据**）|**不等于真的开放端口**；必须通过 `ports` / `-p` 才能发布到宿主机|
|`USER`|指定容器内进程的运行用户|安全基线：生产镜像应设为非 root 用户|
|`WORKDIR`|设置后续指令和容器启动时的工作目录|始终显式设置，避免依赖默认 `/`|

### 1.4 ENTRYPOINT 与 CMD 的交互

这是 Docker 初学者最常混淆的概念之一。两者的关系可以用一个公式概括：

> [!important]
> 
> **容器启动实际执行的命令 = ENTRYPOINT + CMD**
> 
> - `ENTRYPOINT` 是不可被 `docker run <args>` 直接覆盖的前缀（除非用 `--entrypoint`）
> 
> - `CMD` 是默认追加参数，会被 `docker run <args>` 替换

|Dockerfile 写法|`docker run myimg`|`docker run myimg serve --port 8080`|
|---|---|---|
|`ENTRYPOINT ["python"]`  <br>`CMD ["app.py"]`|`python app.py`|`python serve --port 8080`|
|`CMD ["python", "app.py"]`  <br>（无 ENTRYPOINT）|`python app.py`|`serve --port 8080`（整个 CMD 被替换）|
|`ENTRYPOINT ["python", "app.py"]`  <br>（无 CMD）|`python app.py`|`python app.py serve --port 8080`（参数追加）|

> [!important]
> 
> **工程判断**：
> 
> - **应用型镜像**（API 服务、Worker）：`ENTRYPOINT` 设为主进程，`CMD` 设默认参数
> 
> - **工具型镜像**（CLI 工具、编译器）：`ENTRYPOINT` 设为工具命令，`CMD` 设 `--help`
> 
> - **通用基础镜像**：只设 `CMD`，不设 `ENTRYPOINT`，给下游最大灵活性

### 1.5 镜像关键认知三条

> [!important]
> 
> **必须刻入脑中的三条规则：**
> 
> 1. 镜像应尽量**小、确定、可复现**——使用固定 tag（如 `python:3.12-slim-bookworm`）或 digest（`sha256:...`），**不要用** `**latest**`
> 
> 1. **不应把运行态数据写进镜像**——数据库文件、用户上传、日志等运行时产生的数据属于 Volume
> 
> 1. **不应把机密写进镜像层**——密钥、token、证书不能出现在 `COPY`、`ENV`、`ARG` 中，因为镜像层是可被 `docker history` 查看的

---

## 2. 容器（Container）

### 2.1 本质

> [!important]
> 
> **容器（Container）** 是镜像的**运行实例**——在镜像的只读层之上叠加一个**可写层（Writable Layer）**，加上命名空间（Namespace）和控制组（Cgroup）隔离，形成一个独立的进程运行环境。

![[1 Docker 基础对象：必须讲清的边界 - 2.1 本质 - 图 01.excalidraw|800]]

### 2.2 生命周期

![[2026-04-18 20.39.32容器生命周期.excalidraw|600]]

**关键细节**：

- `docker stop` 会先发送 **SIGTERM**，等待宽限期（默认 10 秒），再发 **SIGKILL**

- 应用应正确处理 SIGTERM 实现**优雅退出（Graceful Shutdown）**——关闭连接、刷新缓冲、保存状态

- `docker rm` 只能移除已停止的容器；`docker rm -f` 会先强制停止再移除

### 2.3 可写层的代价

容器的可写层使用**写时复制（CoW）**机制：

1. 读取文件时，直接从镜像只读层读取，零开销

1. **修改**文件时，先将文件从只读层**复制**到可写层，再修改——这带来 IO 开销

1. 可写层的数据在容器被 `docker rm` 后**永久丢失**

> [!important]
> 
> **常见误区：「在容器里 apt install / pip install 了一堆东西」**
> 
> 这些修改只存在于可写层。容器一删就没了。如果需要持久化安装结果，应该写进 Dockerfile 的 `RUN` 指令中（成为镜像层）。如果是运行态数据，应该挂 Volume。

### 2.4 常见误区与正确心智模型

|错误心智模型|正确心智模型|为什么|
|---|---|---|
|容器是轻量虚拟机|容器是**隔离的进程组**|容器共享宿主机内核，不模拟硬件|
|容器应该长期运行、手工维护|容器应**可删除、可重建**（Cattle, not Pets）|任何手工修改在 `rm` 后丢失；自动化重建才可靠|
|配置和数据可以放容器里|配置通过环境变量/Secret 注入，数据通过 Volume 持久化|容器可写层是临时的，且不应被手工依赖|
|`docker exec -it bash` 是正常运维方式|`exec` 只用于**临时调试**，不用于常规操作|需要改配置就改 Compose/Dockerfile 并重建|

---

## 3. Docker daemon / CLI / context

### 3.1 架构：客户端-服务器模型

> [!important]
> 
> Docker 采用**客户端-服务器架构（Client-Server Architecture）**：`docker` CLI 是客户端，`dockerd`（Docker daemon，Docker 守护进程）是服务端。两者通过 Unix socket 或 TCP 通信。

![[1 Docker 基础对象：必须讲清的边界 - 3.1 架构：客户端-服务器模型 - 图 02.excalidraw|800]]

**关键理解**：

- `docker` 命令本身**不直接管理容器**——它只是向 daemon 发送 API 请求

- daemon 负责镜像管理、容器创建、网络配置、存储管理等所有实际工作

- daemon 下层依赖 **containerd**（容器运行时管理器）和 **runc**（OCI 标准容器运行时，OCI 即 Open Container Initiative，开放容器倡议）

### 3.2 docker context：多环境管理

**docker context（Docker 上下文）** 用于管理多个 daemon 连接——本地 CLI 可以切换连接到不同的 daemon（本地、远程开发机、云主机等）。

```Bash
# 列出所有 context
docker context ls

# 创建一个连接远程机器的 context（通过 SSH）
docker context create remote-dev --docker "host=ssh://user@remote-host"

# 切换到远程 context
docker context use remote-dev

# 此后所有 docker 命令都发送到远程 daemon
docker ps  # 列出的是远程机器上的容器
```

### 3.3 远程 daemon 安全

> [!important]
> 
> **严重安全警告：裸 TCP 暴露 Docker daemon 等同于给 root 权限**
> 
> Docker daemon 默认通过 Unix socket 通信，仅本机可访问。如果为了「方便远程开发」将 daemon 通过 TCP 端口（如 `2375`）暴露到网络，任何能访问该端口的人都可以：
> 
> - 启动特权容器
> 
> - 挂载宿主机根文件系统
> 
> - 读取任意文件、安装后门
> 
> **必须通过 SSH 隧道或 TLS 客户端证书访问远程 daemon，不能裸开 TCP。**

**安全访问远程 daemon 的三种方式**：

|方式|安全性|适用场景|配置复杂度|
|---|---|---|---|
|**SSH**（推荐）|高（复用 SSH 认证）|开发 / 运维|低|
|**TLS 客户端证书**|高（双向 TLS）|自动化 / CI|中|
|**受控代理**|取决于代理实现|特殊网络环境|高|

---

## 4. 思辨：容器 vs 虚拟机——何时用哪个？

> [!important]
> 
> **「容器能完全替代虚拟机吗？」**
> 
> 不能。两者的隔离层级不同，适用场景也不同：
> 
> - **容器**：共享宿主机内核，启动秒级，适合**应用级隔离**——微服务、API、Worker、开发环境
> 
> - **虚拟机**：独立内核，启动分钟级，适合**系统级隔离**——多租户强隔离、不同 OS 需求、安全敏感工作负载
> 
> - **结论**：生产中两者常共存——虚拟机提供基础设施级隔离，容器在虚拟机内提供应用级隔离。不要把「容器轻量」误解为「容器更安全」——容器的隔离边界天然弱于 VM。

|维度|容器（Container）|虚拟机（VM）|
|---|---|---|
|**隔离机制**|命名空间 + Cgroup（进程级）|Hypervisor（硬件级）|
|**内核**|共享宿主机内核|独立内核|
|**启动速度**|毫秒 ~ 秒|秒 ~ 分钟|
|**资源开销**|极低（MB 级）|较高（GB 级）|
|**隔离强度**|中（内核漏洞可逃逸）|高（Hypervisor 边界）|
|**典型密度**|单机数百容器|单机数十 VM|

---

## 延伸阅读

> [!important]
> 
> - §2 Dockerfile — 指令体系、多阶段构建、BuildKit 高阶能力
> 
> - §3 Docker Compose — 多服务编排与 Compose Specification
> 
> - §4 挂载 — Volume / Bind / tmpfs 的深入选型
> 
> - §6 安全 — 运行时安全基线、rootless、机密管理

## 参考文献

- [1] Docker Overview — [https://docs.docker.com/get-started/overview/](https://docs.docker.com/get-started/overview/)

- [2] Docker Storage Drivers — [https://docs.docker.com/storage/storagedriver/](https://docs.docker.com/storage/storagedriver/)

- [3] Docker Context — [https://docs.docker.com/engine/context/working-with-contexts/](https://docs.docker.com/engine/context/working-with-contexts/)

- [4] Protect the Docker daemon socket — [https://docs.docker.com/engine/security/protect-access/](https://docs.docker.com/engine/security/protect-access/)

- [5] Dockerfile ENTRYPOINT reference — [https://docs.docker.com/reference/dockerfile/#entrypoint](https://docs.docker.com/reference/dockerfile/#entrypoint)