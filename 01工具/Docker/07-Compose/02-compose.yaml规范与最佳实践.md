# compose.yaml 规范与最佳实践

## 适用场景
- 编写清晰、可维护、适用于生产或协作的 Compose 文件。
- **深度学习场景**：如何正确配置 GPU 资源。

## 1. 文件结构概览
一个标准的 `compose.yaml` 通常包含三个顶级键：

```yaml
name: my-project  # 项目名称 (可选，默认是文件夹名)

services:    # 定义容器
  web: ...
  db: ...

networks:    # 定义网络 (可选)
  my-net: ...

volumes:     # 定义数据卷 (可选)
  db-data: ...
```

## 2. 关键配置详解

### 2.1 GPU 支持 (重点)
在 Docker Compose V2 中，配置 GPU 不再需要 `runtime: nvidia`，而是使用 `deploy` 字段。

```yaml
services:
  training:
    image: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all        # 使用所有 GPU
              # count: 1        # 使用 1 个 GPU
              # device_ids: ['0', '2']  # 指定 GPU ID
              capabilities: [gpu]
```

### 2.2 网络 (Networks)
建议显式定义网络，而不是依赖默认的 `default` 网络，以便更好地控制服务发现。

```yaml
services:
  app:
    networks:
      - backend
  db:
    networks:
      - backend

networks:
  backend:
    driver: bridge
```

### 2.3 数据卷 (Volumes)
- **Bind Mount (开发用)**: 挂载本地代码。
- **Named Volume (数据库用)**: 持久化数据。

```yaml
services:
  lab:
    volumes:
      - ./code:/workspace  # Bind Mount
      - tensorboard-logs:/logs # Named Volume

volumes:
  tensorboard-logs: # 必须在顶级声明
```

### 2.4 环境变量
推荐使用 `env_file` 而不是 `environment` 列表，保持 yaml 简洁。

```yaml
services:
  app:
    env_file:
      - .env
      - .env.local
```

## 3. 最佳实践清单
1.  **使用相对路径**: `build: ./backend` 或 `volumes: - ./data:/data`，方便团队成员 clone 代码后直接运行。
2.  **显式指定镜像 Tag**: 别用 `latest`，用 `postgres:15-alpine`。
3.  **利用 `depends_on`**: 虽然它只等待容器启动（不等待服务就绪），但至少能保证启动顺序。
    ```yaml
    depends_on:
      db:
        condition: service_started
    ```
4.  **端口最小化暴露**: 数据库端口尽量不要映射到宿主机 (`ports`)，除非你需要用本地 GUI 工具连接。容器间通信直接用服务名即可。

- profiles（多环境）
- restart 策略
- healthcheck

## TODO
- [ ] 给出推荐 compose 模板（开发环境）
- [ ] 记录一份字段速查表
