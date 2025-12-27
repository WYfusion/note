# Docker 到 Kubernetes 概念映射

## 适用场景
- 熟悉 Docker 的开发者快速理解 Kubernetes (K8s) 的核心概念。
- 准备将单机 Docker Compose 应用迁移到 K8s 集群。

## 1. 核心对象对照表

| Docker 概念 | Kubernetes 概念 | 说明 |
| :--- | :--- | :--- |
| **Image** | **Image** | 镜像概念完全一致，都存储在 Registry 中。 |
| **Container** | **Container** | 容器运行时（如 containerd）负责拉起，但 K8s 不直接管理容器。 |
| **(无对应)** | **Pod** | K8s 的最小调度单元。一个 Pod 可以包含多个共享网络/存储的容器（Sidecar 模式）。 |
| **docker run** | **Pod** (一次性) | 手动启动一个 Pod。 |
| **Compose Service** | **Deployment** | 负责无状态应用（Web/API）的副本管理、滚动更新。 |
| **Compose Service** | **StatefulSet** | 负责有状态应用（DB/Redis），保证网络 ID 和存储的持久性。 |
| **Volume** | **PersistentVolume (PV) / PVC** | 存储卷。K8s 将存储的“提供” (PV) 和“使用” (PVC) 分离了。 |
| **-p 80:80** | **Service (NodePort/LoadBalancer)** | 暴露服务端口。 |
| **Network** | **CNI / Service / Ingress** | K8s 网络更复杂，Service 负责服务发现，Ingress 负责 HTTP 路由。 |
| **compose.yaml** | **Manifests (.yaml)** | K8s 的资源定义文件，通常一个文件包含多个 Kind。 |

## 2. 深度学习场景迁移示例

### Docker Compose
```yaml
services:
  jupyter:
    image: pytorch/pytorch
    ports: ["8888:8888"]
    volumes: ["./data:/workspace"]
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

### Kubernetes (简化版)
在 K8s 中，这通常需要定义一个 `Pod` 或 `Deployment`。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: jupyter-pod
spec:
  containers:
  - name: jupyter
    image: pytorch/pytorch
    ports:
    - containerPort: 8888
    volumeMounts:
    - mountPath: /workspace
      name: data-volume
    resources:
      limits:
        nvidia.com/gpu: 1  # 请求 1 块 GPU
  volumes:
  - name: data-volume
    hostPath:
      path: /data/user1  # 简单场景用 hostPath，生产用 PVC
```

- volume -> PV/PVC

## TODO
- [ ] 画一张对照表
