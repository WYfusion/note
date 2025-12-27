# Docker 网络模型

## 适用场景
- **深度学习分布式训练**：多机多卡训练时，容器间需要高速互通。
- **高性能服务**：部署 Triton Inference Server 时，减少 NAT 转发损耗。
- **网络隔离**：防止测试环境的容器误连生产数据库。

## 1. 核心网络模式
Docker 提供了多种网络驱动，最常用的是 `bridge` 和 `host`。

| 模式 | 描述 | 适用场景 | 性能 |
| :--- | :--- | :--- | :--- |
| **bridge** (默认) | 容器有独立 IP，通过 NAT 访问外网。 | 单机开发、端口隔离。 | 中 (NAT 损耗) |
| **host** | 容器**共享宿主机网络栈**，没有独立 IP。 | **分布式训练**、高性能推理、网络调试。 | **高 (原生)** |
| **none** | 无网络，只有 loopback。 | 极高安全要求的离线任务。 | - |
| **container** | 共享另一个容器的网络栈 (k8s Pod 原理)。 | 边车模式 (Sidecar)。 | 高 |

### 1.1 Bridge 模式 (默认)
这是最常见的模式。容器像是在一个私有局域网里。
- **特点**：需要 `-p` 映射端口才能被外部访问。
- **缺点**：NAT 转发会消耗少量 CPU，且增加延迟。

```bash
# 默认就是 bridge
docker run -d -p 80:80 nginx
```

### 1.2 Host 模式 (性能之选)
容器直接使用宿主机的 IP 和端口。
- **特点**：**不需要 `-p`**。容器里监听 80 端口，宿主机的 80 就被占用了。
- **优势**：消除 NAT 损耗，吞吐量最大。**PyTorch 分布式训练 (DDP) 推荐使用此模式**，避免复杂的端口映射配置。
- **缺点**：端口冲突风险高；Windows/Mac Docker Desktop 对此模式支持不完整（因为中间隔了一层虚拟机）。

```bash
# 容器内的服务直接暴露在宿主机网络上
docker run -d --network host my-inference-server
```

## 2. 自定义 Bridge 网络 (推荐)
默认的 `bridge` 网络有个缺陷：**容器间无法通过“容器名”互相访问**（只能用 IP，但 IP 会变）。
**最佳实践**是创建一个自定义网络。

### 步骤 1：创建网络
```bash
docker network create my-dl-net
```

### 步骤 2：加入网络
```bash
# 启动容器 A (数据库)
docker run -d --name redis --network my-dl-net redis

# 启动容器 B (训练脚本)
docker run -it --network my-dl-net pytorch/pytorch bash
```

### 步骤 3：互通验证
在容器 B 里，可以直接用 `redis` 这个名字访问容器 A：
```bash
# 在容器 B 内
ping redis
# 输出: 64 bytes from 172.18.0.2 ... (自动解析到了容器 A 的 IP)
```

## 3. 常用网络命令
```bash
# 列出所有网络
docker network ls

# 查看网络详情 (看哪些容器连在这个网络上)
docker network inspect my-dl-net

# 将运行中的容器加入网络
docker network connect my-dl-net my-container

# 删除网络
docker network rm my-dl-net
```

## 总结
- **单机简单用**：默认 Bridge + `-p` 映射。
- **多容器互联**：`docker network create` 自定义网络，用名字互连。
- **追求高性能/分布式**：`--network host` (Linux only)。

