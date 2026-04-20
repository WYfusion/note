## 前置知识

> [!important]
> 
> 阅读本页前建议先读：
> 
> - [[1 Docker 基础对象：必须讲清的边界]]——理解 Network 作为四大核心对象之一的定位
> 
> - [[3 Docker Compose：当前推荐的项目主入口]]——理解 Compose 中 `networks` 顶层项与服务级网络配置

---

## 0. 定位

> Docker 六种网络模式（bridge / host / none / overlay / macvlan / ipvlan）的本质、选型准则、端口暴露与发布机制、网络隔离设计。本页覆盖**容器网络通信的完整知识**。

---

## 1. 网络模式总览

> [!important]
> 
> Docker 网络模式（Network Driver）决定了容器如何与其他容器、宿主机、外部网络通信。不同模式提供不同级别的隔离和连通性。**选择网络模式 = 选择隔离与连通的平衡点**。

![[2026-04-18 20.46.20Docker网络.excalidraw|1000]]

|网络模式|本质|典型场景|推荐度|
|---|---|---|---|
|**bridge**|虚拟网桥 + NAT|单机多容器互通（绝大多数场景）|⭐⭐⭐⭐⭐ 默认首选|
|**host**|共享宿主机网络栈|高性能网络工具、特殊代理|⭐⭐ 特殊需求|
|**none**|仅 loopback|完全离线计算、安全沙箱|⭐⭐ 特殊需求|
|**overlay**|跨主机虚拟网络|多机服务互联、Swarm 集群|⭐⭐⭐ 多机刚需|
|**macvlan**|容器获得独立 MAC 地址|需要 LAN 级可见性|⭐ 高级场景|
|**ipvlan**|容器获得独立 IP（共享 MAC）|类似 macvlan 但兼容性更好|⭐ 高级场景|

---

## 2. bridge（最常用）

### 2.1 工作原理

> [!important]
> 
> **bridge 网络（桥接网络）** 在宿主机上创建一个**虚拟网桥（Virtual Bridge）**，每个加入该网络的容器获得一个虚拟网卡（veth pair），通过网桥互相通信。容器与外部网络的通信通过 **NAT（网络地址转换）** 和**端口发布**实现。

![[5 网络：Docker 网络模式与选择准则 - 2.1 工作原理 - 图 01 .excalidraw|800]]

### 2.2 默认 bridge vs 用户自定义 bridge

|维度|默认 bridge（`docker0`）|用户自定义 bridge|
|---|---|---|
|**服务发现**|只能通过 IP 互访，无 DNS|✅ 自动 DNS：容器名即主机名|
|**隔离性**|所有容器共享一个网络|✅ 按项目/组件划分网络|
|**可管理性**|全局共享，难以管理|✅ 可按需创建/删除|
|**Compose 行为**|不使用|✅ Compose 自动为项目创建|

> [!important]
> 
> **工程判断**：**始终使用用户自定义 bridge，不要使用默认 bridge**。Compose 会自动为每个项目创建专用的 bridge 网络，这是开箱即用的最佳实践。

### 2.3 Compose 中的 bridge 网络

```YAML
# Compose 自动创建项目级 bridge 网络
# 所有服务默认加入同一个网络，通过服务名互访
services:
  api:
    build: ./services/api
    environment:
      - DB_HOST=db           # 直接用服务名访问
      - REDIS_HOST=redis
  db:
    image: postgres:16-alpine
  redis:
    image: redis:7-alpine

# 自定义多网络实现隔离
networks:
  frontend:
  backend:
  data:

services:
  gateway:
    networks: [frontend, backend]   # 网关连接前后端
  api:
    networks: [backend, data]       # API 连接后端和数据层
  db:
    networks: [data]                # DB 只在数据层
```

---

## 3. host

> [!important]
> 
> **host 网络模式** 让容器直接共享宿主机的网络命名空间（Network Namespace）。容器不再拥有独立的 IP 地址和端口空间，而是直接使用宿主机的网络栈。

**优点**：

- 少一层 NAT / 端口映射，网络性能略高

- 某些需要广播发现、组播、网络探测的程序更直接

**缺点**：

- 网络隔离**显著降低**——容器进程直接绑定宿主机端口

- 端口冲突直接与宿主机冲突

- 无法使用 Docker 内建的服务发现

> [!important]
> 
> **常见误区：「用 host 网络掩盖网络配置问题」**
> 
> 当容器之间网络不通时，有些开发者直接切到 host 模式「解决」问题。这实际上是掩盖了网络拓扑设计的缺陷。**host 模式不是一般 Web 项目的默认方案**，只用于确有需求的特殊服务：高性能网络工具、特定代理/抓包程序、广播发现类程序。

```YAML
services:
  network-monitor:
    image: my-monitor:latest
    network_mode: host        # 使用 host 网络
```

---

## 4. none

**none 网络模式**只保留容器内的 loopback（`127.0.0.1`），完全没有外部网络连接。

**适用场景**：

- 完全离线的计算任务（如离线数据处理）

- 高隔离安全沙箱（如执行不可信代码）

- 不需要网络的纯 CPU 计算

```YAML
services:
  sandbox:
    image: my-sandbox:latest
    network_mode: none        # 完全断网
```

---

## 5. overlay

> [!important]
> 
> **overlay 网络** 是 Docker 的**跨主机网络方案**，在多个 Docker daemon 主机之上创建一个虚拟的分布式网络层。不同主机上的容器通过 overlay 网络可以像在同一台机器上一样互相通信。

**适用场景**：

- 多机服务互联

- Docker Swarm 集群

- 跨数据中心部署

**代价**：

- 网络与运维复杂度明显上升

- 需要额外的服务发现和状态存储（如 etcd）

- 性能开销：封装/解封装数据包

> [!important]
> 
> **「什么时候需要 overlay？」**
> 
> 只有当你确实需要**多台物理/虚拟主机上的容器互相通信**时才需要 overlay。如果所有容器都在同一台机器上，user-defined bridge 完全够用。**不要为了「看起来高级」而上 overlay**。

---

## 6. macvlan / ipvlan

### 6.1 macvlan

**macvlan** 让容器拥有**独立的 MAC 地址**，使其在物理网络中看起来像一台独立设备。容器直接获得 LAN 中的 IP 地址。

### 6.2 ipvlan

**ipvlan** 与 macvlan 类似，但所有容器**共享父接口的 MAC 地址**，只分配独立的 IP。在某些网络环境中（如限制 MAC 地址数量的交换机），ipvlan 兼容性更好。

**适用场景**：

- 需要容器直接获得 LAN 地址

- 需要网络设备级可见性

- 某些传统系统集成

**风险与代价**：

- 网络设计复杂

- 交换机/VLAN/物理网络约束明显

- Docker 自带的 bridge 防火墙规则**不再替你兜底**——你需要自己管理防火墙

> [!important]
> 
> **macvlan / ipvlan 不是普通业务的默认网络**。除非你有明确的物理网络集成需求，否则不要使用。切换到更底层的网络模式意味着你**自己承担更多的安全与网络控制责任**。

---

## 7. 端口暴露与发布

### 7.1 EXPOSE vs ports

|机制|作用|实际效果|
|---|---|---|
|`EXPOSE`（Dockerfile）|镜像元数据声明|**不等于真的开放端口**，仅文档作用|
|`expose`（Compose）|内部端口声明|同网络容器可访问，不发布到宿主机|
|`ports` / `-p`|真正发布到宿主机|**外部可访问**——攻击面放大器|

### 7.2 端口发布安全原则

> [!important]
> 
> **端口发布是攻击面的直接放大器**。遵循以下原则：
> 
> 1. **内部服务不 publish**——DB、Redis、消息队列等只在内部网络通信
> 
> 1. **只把入口层暴露**——如 gateway / API / Web
> 
> 1. **本机开发绑定** `**127.0.0.1**`——不要默认 `0.0.0.0`
> 
> 1. **不要** `**0.0.0.0:所有服务**`——逐个评估哪些需要对外

```YAML
services:
  # ✅ 正确：只暴露入口层，绑定本机
  gateway:
    ports:
      - "127.0.0.1:443:443"   # 仅本机可访问
      - "127.0.0.1:80:80"
  
  api:
    expose:
      - "8000"                 # 仅内部网络可访问
  
  db:
    # ❌ 不要这样：数据库不应暴露到宿主机
    # ports:
    #   - "0.0.0.0:5432:5432"
    expose:
      - "5432"                 # 仅内部网络
```

---

## 8. 网络隔离设计

### 8.1 分层网络架构

![[5 网络：Docker 网络模式与选择准则 - 8.1 分层网络架构 - 图 02 .excalidraw|800]]

```YAML
networks:
  frontend:
  backend:
  data:
  observability:

services:
  gateway:
    networks: [frontend, backend]
    ports:
      - "127.0.0.1:443:443"
  
  api:
    networks: [backend, data]
  
  worker:
    networks: [backend, data]
  
  db:
    networks: [data]           # DB 只在数据层
  
  redis:
    networks: [data]
  
  prometheus:
    networks: [observability, backend]
    profiles: [observability]
```

### 8.2 设计原则

- ✅ 让服务只加入**必要的**网络

- ✅ DB / 缓存 / 消息队列**不应在公网暴露**

- ✅ 跨网络通信通过**显式入口或网关**

- ✅ 入口层（gateway）是连接外部与内部的唯一桥梁

- ❌ 不要让所有服务都在同一个扁平网络中

---

## 延伸阅读

> [!important]
> 
> - [[1 Docker 基础对象：必须讲清的边界]] — Network 基础概念
> 
> - [[3 Docker Compose：当前推荐的项目主入口]] — Compose 中的网络声明
> 
> - §6 安全 — 防火墙规则与网络安全
> 
> - §10 AI/Agent/LLM 项目 Docker — 网络分层的完整方案

## 参考文献

- [1] Docker networking overview — [https://docs.docker.com/network/](https://docs.docker.com/network/)

- [2] Bridge networks — [https://docs.docker.com/network/bridge/](https://docs.docker.com/network/bridge/)

- [3] Host networking — [https://docs.docker.com/network/host/](https://docs.docker.com/network/host/)

- [4] Overlay networks — [https://docs.docker.com/network/overlay/](https://docs.docker.com/network/overlay/)

- [5] Macvlan networks — [https://docs.docker.com/network/macvlan/](https://docs.docker.com/network/macvlan/)

- [6] IPvlan networks — [https://docs.docker.com/network/ipvlan/](https://docs.docker.com/network/ipvlan/)