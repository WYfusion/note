# Docker Compose 快速上手

## 适用场景
- **单机多容器编排**：比如一个深度学习项目，需要同时启动 JupyterLab、TensorBoard 和一个 MySQL 数据库。
- **一键启动**：不再需要写一堆 `docker run -p ... -v ... --network ...` 命令，全部写在一个文件里。

## 1. 核心概念
Docker Compose 是一个用于定义和运行多容器 Docker 应用程序的工具。
- **配置文件**: `compose.yaml` (旧版本叫 `docker-compose.yml`)。
- **命令**: `docker compose` (新版 V2) 或 `docker-compose` (旧版 V1)。

## 2. 常用命令速查

| 命令 | 作用 | 常用参数 |
| :--- | :--- | :--- |
| `docker compose up` | **启动**所有服务 | `-d` (后台运行), `--build` (强制重构) |
| `docker compose down` | **停止**并删除容器、网络 | `-v` (同时删除数据卷 - **慎用**) |
| `docker compose ps` | 查看当前项目的容器状态 | `-a` (查看所有) |
| `docker compose logs` | 查看日志 | `-f` (实时跟随), `service_name` (只看某服务) |
| `docker compose exec` | 进入容器 | `service_name bash` |
| `docker compose restart` | 重启服务 | `service_name` |

## 3. 最小上手示例
假设我们需要一个 Python 环境和一个 Redis 数据库。

### 步骤 1: 编写 `compose.yaml`
```yaml
services:
  # 服务 1: Python 脚本
  app:
    image: python:3.9-slim
    command: python -m http.server 8000
    ports:
      - "8000:8000"
    volumes:
      - .:/app
    working_dir: /app
    depends_on:
      - redis

  # 服务 2: Redis 数据库
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

### 步骤 2: 启动
```bash
# 前台启动 (能看到所有日志)
docker compose up

# 后台启动 (推荐)
docker compose up -d
```

### 步骤 3: 验证
```bash
docker compose ps
```

### 步骤 4: 停止
```bash
docker compose down
```

## 4. 环境变量 (`.env`)
Compose 会自动读取同目录下的 `.env` 文件。
**场景**: 数据库密码、API Key 不想写死在 yaml 里。

`.env` 文件:
```bash
DB_PASSWORD=secret123
```

`compose.yaml`:
```yaml
services:
  db:
    image: mysql
    environment:
      MYSQL_ROOT_PASSWORD: ${DB_PASSWORD}
```


## TODO
- [ ] 一个最小 web + redis 例子（只给结构也行）
