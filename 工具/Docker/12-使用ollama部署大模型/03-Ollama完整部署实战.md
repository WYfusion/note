# Ollama + Docker 完整部署实战

本文将介绍如何使用 Docker 部署 Ollama，并结合 Open WebUI 搭建一套完整的本地大模型对话系统。

## 1. 为什么选择 Docker 部署？
- **环境隔离**：无需在宿主机安装复杂的依赖（如 CUDA 版本冲突问题）。
- **一键编排**：可以轻松结合 Web 界面（如 Open WebUI）或其他应用。
- **易于升级**：只需拉取新镜像即可更新 Ollama 版本。

## 2. 前置准备
1. **安装 Docker**：确保 Docker Engine 已安装并运行。
2. **GPU 支持（强烈推荐）**：
   - 确保显卡驱动已安装。
   - **关键步骤**：安装 **NVIDIA Container Toolkit**。如果未安装，Docker 无法调用显卡。
   - *排错指南*：参见 [01-NVIDIA Container Toolkit的缺失](./01-NVIDIA Container Toolkit的缺失.md)。

## 3. 基础玩法：快速启动服务

### 3.1 启动 Ollama 容器
根据你的硬件情况选择一种启动方式。

**方案 A：GPU 模式（推荐，速度快）**
```bash
docker run -d \
  --gpus=all \
  -v ollama:/root/.ollama \
  -p 11434:11434 \
  --name ollama \
  --restart always \
  ollama/ollama
```

**方案 B：CPU 模式（速度较慢）**
```bash
docker run -d \
  -v ollama:/root/.ollama \
  -p 11434:11434 \
  --name ollama \
  --restart always \
  ollama/ollama
```

### 3.2 下载并运行模型
容器启动后，我们需要进入容器内部（或通过 API）来下载模型。**你可以自由选择任何 Ollama 支持的模型。**

1. **挑选模型**：访问 [Ollama 模型库](https://ollama.com/library) 查找你感兴趣的模型（如 `llama3`, `deepseek-r1`, `mistral` 等）。
2. **运行模型**：

```bash
# 语法：docker exec -it <容器名> ollama run <模型名>
# 示例：运行阿里通义千问 2.5 (7B版本)
docker exec -it ollama ollama run qwen2.5:7b
```
- 上述命令中的 `qwen2.5:7b` 仅为示例，请替换为你实际想用的模型名称。
- 首次运行会自动下载模型文件。
- 下载完成后，你就可以在终端里直接和 AI 对话了。

---

## 4. 进阶玩法：搭建 ChatGPT 风格的 Web 界面 (Open WebUI)

单纯在终端对话体验一般，我们可以使用 **Docker Compose** 一键部署 **Ollama + Open WebUI**，获得类似 ChatGPT 的网页体验。

### 4.1 创建编排文件
新建一个文件夹（例如 `my-llm`），在其中创建 `docker-compose.yaml` 文件：

```yaml
version: '3.8'

services:
  # Ollama 服务
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    # 如果需要 GPU 支持，取消下面注释
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - ollama_data:/root/.ollama
    ports:
      - "11434:11434"
    restart: always

  # Open WebUI 服务 (原 Ollama WebUI)
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: open-webui
    volumes:
      - open-webui_data:/app/backend/data
    ports:
      - "3000:8080"
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434 # 连接到上面的 ollama 服务
    depends_on:
      - ollama
    restart: always

volumes:
  ollama_data:
  open-webui_data:
```

> **💡 技巧：如何复用之前下载的模型？**
> 如果你之前（如章节 3.1）运行容器时使用了 `-v ollama:/root/.ollama`，那么你的模型保存在名为 `ollama` 的数据卷中。
> 而上面的 Compose 文件默认会创建一个新的卷 `ollama_data`，这会导致**重新下载模型**。
> 
> **不想重新下载？请这样修改 `docker-compose.yaml`：**
> 1. 找到 `services: -> ollama: -> volumes:`，把 `- ollama_data:/root/.ollama` 改为 `- ollama:/root/.ollama`。
> 2. 修改文件最底部的 `volumes` 区域：
>    ```yaml
>    volumes:
>      ollama:
>        external: true  # 告诉 Docker 使用现有的名为 "ollama" 的卷
>      open-webui_data:
>    ```

### 4.2 启动服务
在 `docker-compose.yaml` 所在目录下运行：
```bash
docker compose up -d
```

### 4.3 访问与使用
1. 打开浏览器访问 `http://localhost:3000`。
2. 注册管理员账号（第一个注册的用户自动成为管理员，数据保存在本地）。
3. **下载模型**：在设置 -> 模型 -> 拉取模型中，输入你想用的模型名称（例如 `llama3`, `deepseek-r1`, `qwen2.5:7b` 等）并点击下载。
4. 下载完成后，在主界面顶部选择该模型，即可开始对话！

### 4.4 局域网访问配置（让同事也能用）
默认情况下，Docker 映射的端口（如 `3000`）是允许局域网访问的。要实现其他设备访问，只需两步：

1. **获取宿主机 IP**：
   - **Windows**: 打开终端输入 `ipconfig`，找到“IPv4 地址”（例如 `192.168.1.10`）。
   - **Linux/Mac**: 输入 `ifconfig` 或 `ip a` 查看。

2. **开放防火墙端口**（关键）：
   - **Windows**:
     1. 搜索“高级安全 Windows Defender 防火墙”。
     2. 点击“入站规则” -> “新建规则”。
     3. 选择“端口” -> “TCP” -> 特定本地端口填 `3000`。
     4. 选择“允许连接” -> 命名为 `OpenWebUI` -> 完成。
   - **Linux (Ubuntu/UFW)**: `sudo ufw allow 3000/tcp`

3. **访问**:
   - 在同一局域网的手机或电脑浏览器输入：`http://172.23.201.21:3000`（替换为你实际的 IP）。

---

## 5. 开发者玩法：API 调用

Ollama 提供了兼容 OpenAI 格式的 API，方便集成到代码中。

### 5.1 简单的 Python 示例
安装 OpenAI SDK：
```bash
pip install openai
```

编写 `test_ollama.py`：
```python
from openai import OpenAI

# 指向本地 Docker 暴露的端口
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama', # key 可以随便填
)

response = client.chat.completions.create(
    model="qwen2.5:7b", # ⚠️ 请修改为你实际下载的模型名称
    messages=[
        {"role": "system", "content": "你是一个乐于助人的助手。"},
        {"role": "user", "content": "用一句话解释什么是 Docker。"},
    ]
)

print(response.choices[0].message.content)
```

### 5.2 常用 API 端点
- **生成补全**: `POST /api/generate`
- **聊天对话**: `POST /api/chat`
- **列出本地模型**: `GET /api/tags`

例如查看本地有哪些模型：
```bash
curl http://localhost:11434/api/tags
```
