## 前置知识

> [!important]
> 
> 阅读本页前建议先读：
> 
> - [[3 Docker Compose：当前推荐的项目主入口]]——理解 profiles、GPU 声明、多文件合并
> 
> - [[4 挂载：Mount - Volume - Bind - tmpfs 一定要分清]]——理解各挂载类型的选型
> 
> - [[5 网络：Docker 网络模式与选择准则]]——理解网络隔离设计
> 
> - [[6 安全：当前项目设置中最关键的一层]]——理解安全基线

---

## 0. 定位

> AI / Agent / LLM 项目的 Docker 服务拆分、网络设计、数据挂载、GPU 管理、安全隔离。本页覆盖 **AI 场景下 Docker 工程化的完整推荐方案**。

---

## 1. 服务拆分建议

> [!important]
> 
> AI/LLM 项目的服务拆分核心原则：**按职责隔离、按资源需求分离**。GPU 密集的模型推理服务与 CPU 密集的业务逻辑服务不应混在同一个容器中——它们的基础镜像、资源需求、扩缩容策略完全不同。

![[10 AI - Agent - LLM 项目中的 Docker 推荐做法 - 1. 服务拆分建议 - 图 01 .excalidraw|800]]

### 1.1 典型服务清单

| 服务              | 职责                       | 资源需求    | 典型镜像                        |
| --------------- | ------------------------ | ------- | --------------------------- |
| `gateway`       | 反向代理、TLS 终止、限流           | CPU 低   | nginx / caddy               |
| `api` / `agent` | 业务逻辑、API 路由、Agent 编排     | CPU 中   | 自定义 Python/Node             |
| `worker`        | 异步任务（文档解析、embedding、批处理） | CPU 中~高 | 自定义 Python                  |
| `model-serving` | GPU 模型推理（LLM / TTS / VC） | **GPU** | vLLM / TGI / Triton / 自定义   |
| `vector-db`     | 向量索引与检索                  | CPU/内存  | qdrant / milvus / weaviate  |
| `postgres`      | 关系型数据存储                  | CPU/IO  | postgres:16-alpine          |
| `redis`         | 缓存、消息队列、session          | 内存      | redis:7-alpine              |
| `observability` | 监控、日志、追踪                 | CPU 低~中 | prometheus / grafana / loki |

---

## 2. 网络设计建议

> [!important]
> 
> **工程判断**：AI 项目的网络隔离应遵循**最小暴露原则**。数据库和向量库绝不应直接暴露到外部；模型推理服务只需要被 API/Worker 访问。

```YAML
# compose.yaml 网络定义
networks:
  public:       # 仅 gateway
  app:          # api / worker / model-serving
  data:         # db / cache / vector
  observability: # 监控组件

services:
  gateway:
    networks: [public, app]
  api:
    networks: [app, data]
  worker:
    networks: [app, data]
  model-serving:
    networks: [app]
  postgres:
    networks: [data]          # ❌ 不加入 public
  redis:
    networks: [data]
  vector-db:
    networks: [data]
```

|网络|包含服务|原则|
|---|---|---|
|`public`|仅 gateway|唯一对外入口|
|`app`|api / worker / model-serving|业务互通|
|`data`|db / cache / vector|仅允许 app 层访问|
|`observability`|prometheus / grafana|独立监控网络|

---

## 3. 数据挂载建议

|数据类型|挂载方式|说明|
|---|---|---|
|**模型缓存**|volume|避免每次启动重新下载模型（如 HuggingFace cache）|
|**向量索引**|volume|Qdrant / Milvus 持久化数据|
|**代码**|bind mount（仅开发）|生产态镜像内包含|
|**临时推理文件**|tmpfs|音频/图像中间结果，用完即丢|
|**机密**|Compose secrets|API key、模型访问 token|
|**DB 数据**|volume|PostgreSQL / Redis 持久化|

```YAML
volumes:
  model-cache:           # HuggingFace / 模型权重缓存
  vector-data:           # 向量数据库索引
  pg-data:               # PostgreSQL
  redis-data:            # Redis

services:
  model-serving:
    volumes:
      - model-cache:/root/.cache/huggingface    # 模型缓存持久化
    tmpfs:
      - /tmp                                      # 临时推理文件
```

---

## 4. GPU 建议

> [!important]
> 
> Compose 支持通过 `deploy.resources.reservations.devices` 声明 GPU 需求。GPU 服务应**与 CPU 服务分离**——不同的基础镜像（CUDA vs Alpine）、不同的资源需求、不同的扩缩容策略。

### 4.1 GPU 服务独立管理

```YAML
# compose.gpu.yaml
services:
  model-serving:
    image: vllm/vllm-openai:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1              # 或 "all"
              capabilities: [gpu]
    volumes:
      - model-cache:/root/.cache/huggingface
    environment:
      - MODEL_NAME=meta-llama/Llama-3-8B
      - MAX_MODEL_LEN=4096
```

### 4.2 GPU 管理原则

- ✅ GPU 服务单独 profile 或独立 compose 文件

- ✅ 明确模型 cache 挂载——避免每次重新下载数 GB 模型

- ✅ CPU 服务与 GPU 服务使用不同的基础镜像

- ❌ 不要把所有服务都塞到 GPU 镜像中——CUDA 基础镜像通常 >3GB

- ❌ 不要让 CPU 服务依赖 GPU 环境

### 4.3 CPU vs GPU 镜像分离

![[10 AI - Agent - LLM 项目中的 Docker 推荐做法 - 4.3 CPU vs GPU 镜像分离 - 图 02 .excalidraw|800]]

---

## 5. 安全建议

> [!important]
> 
> **AI/Agent 项目的安全风险更高**
> 
> Agent 可能执行用户提供的代码、下载第三方资源、调用外部 API。这些行为如果在高权限容器中执行，风险极大。核心原则：**沙箱隔离 + 最小权限 + 出站控制**。

### 5.1 关键安全规则

- ❌ **绝不把** `**docker.sock**` **随便挂给 agent**——这等于给 agent 宿主机的完整控制权

- ✅ 沙箱工具容器与主业务容器分离

- ✅ 下载/执行第三方代码的容器使用更强隔离

- ✅ 出站网络控制——限制容器可以访问的外部地址

- ✅ 挂载范围最小化——agent 容器不应能访问主业务数据

- ✅ capabilities 更收紧——`cap_drop: [ALL]` 是底线

### 5.2 Agent 沙箱架构

![[10 AI - Agent - LLM 项目中的 Docker 推荐做法 - 5.2 Agent 沙箱架构 - 图 03 .excalidraw|800]]

---

## 完整 AI 项目 Compose 示例骨架

```YAML
# compose.yaml — AI/LLM 项目基础拓扑
name: ai-project

networks:
  public:
  app:
  data:

volumes:
  model-cache:
  vector-data:
  pg-data:
  redis-data:

secrets:
  openai-key:
    file: ./secrets/openai-key.txt

x-security: &security
  read_only: true
  cap_drop: [ALL]
  security_opt: ["no-new-privileges:true"]
  tmpfs: [/tmp]

services:
  gateway:
    <<: *security
    image: caddy:2-alpine
    ports:
      - "127.0.0.1:443:443"
    networks: [public, app]

  api:
    <<: *security
    build: ./services/api
    user: "1000:1000"
    networks: [app, data]
    secrets: [openai-key]
    depends_on:
      postgres: { condition: service_healthy }
      redis: { condition: service_healthy }

  worker:
    <<: *security
    build: ./services/worker
    user: "1000:1000"
    networks: [app, data]

  postgres:
    <<: *security
    image: postgres:16-alpine
    volumes: [pg-data:/var/lib/postgresql/data]
    networks: [data]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      retries: 5

  redis:
    <<: *security
    image: redis:7-alpine
    volumes: [redis-data:/data]
    networks: [data]
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      retries: 5

  vector-db:
    <<: *security
    image: qdrant/qdrant:latest
    volumes: [vector-data:/qdrant/storage]
    networks: [data]
```

---

## 延伸阅读

> [!important]
> 
> - [[3 Docker Compose：当前推荐的项目主入口]] — GPU 声明、profiles、多文件合并
> 
> - [[4 挂载：Mount - Volume - Bind - tmpfs 一定要分清]] — 挂载选型详解
> 
> - [[5 网络：Docker 网络模式与选择准则]] — 网络隔离设计
> 
> - [[6 安全：当前项目设置中最关键的一层]] — 安全基线与机密管理
> 
> - [[8 项目设置推荐：当前最稳的工程化组织方式]] — 目录结构与 Compose 文件组织

## 参考文献

- [1] Docker GPU support — [https://docs.docker.com/compose/gpu-support/](https://docs.docker.com/compose/gpu-support/)

- [2] vLLM Docker deployment — [https://docs.vllm.ai/en/latest/serving/deploying_with_docker.html](https://docs.vllm.ai/en/latest/serving/deploying_with_docker.html)

- [3] Docker security best practices — [https://docs.docker.com/develop/security-best-practices/](https://docs.docker.com/develop/security-best-practices/)