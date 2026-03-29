[ollama/ollama - Docker Image](https://hub.docker.com/r/ollama/ollama)
创建名为`ollama-qwen3-coder30b`的容器，其镜像是ollama/ollama
```bash
docker run -d --gpus=all   -v ollama:/root/.ollama   -p 11434:11434   --name ollama-qwen3-coder30b   ollama/ollama
```

进入创建的`ollama-qwen3-coder30b`容器的同时部署名为 `qwen3-coder:30b` 的镜像到该容器中
```bash
docker exec -it ollama-qwen3-coder30b ollama run qwen3-coder:30b
```

其中的`ollama run qwen3-coder:30b`部分**完全发生在该容器内部**，和外面的 Docker 没有直接关系。含义是：
- **`ollama`** (主程序): 这是安装在容器内部的一个可执行二进制程序（就像 `python`, `git`, `vim` 一样）。它是 Ollama 的核心客户端。
- **`run`** (动作/子命令): 这是告诉 `ollama` 主程序要做什么。`run` 的逻辑是：
    1. 先检查本地有没有模型？没有就自动下载 (Pull)。
    2. 加载模型到显存。
    3. 启动交互式对话界面。
- **`qwen3-coder:30b`** (参数/对象): 这是你要运行的具体目标。
    - `qwen3-coder`: 模型家族的名字。
    - `:30b`: **标签 (Tag)**。指定了具体的参数量版本（300亿参数）。Ollama 需要这个标签来精准定位它该去下载哪个文件。

等价于
```bash
# 第一步：先进入容器的 Bash 环境
docker exec -it ollama-qwen3-coder30b /bin/bash

# 第二步：在容器里面手动输入命令
root@container_id:/# ollama run qwen3-coder:30b
```

