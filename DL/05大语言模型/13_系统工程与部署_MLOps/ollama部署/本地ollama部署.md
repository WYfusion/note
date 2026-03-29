注意ollama部署有安全风险，不推荐使用，但是可以在vllm无法兼容的低水平显卡比如1080ti上面部署一些模型。Ollama 的底层是 `llama.cpp`，它对老旧架构（如 Pascal 架构的 1080 Ti）支持极好，而且自带量化（GGUF 格式），能大幅降低显存需求。

---
以下是操作步骤：
### 1. 安装 Ollama
在你的终端直接运行（不需要 conda 环境，它是独立的二进制文件）：

```Bash
curl -fsSL https://ollama.com/install.sh | sh
```
### 2. 启动服务
如若需要调整启动时占用的GPU、模型安装的位置等信息可以先配置：
```bash
	# 创建模型目录(如果还没创建)
mkdir -p /home/wjgzhu/data/ollama

# 启动命令
OLLAMA_MODELS="/home/wjgzhu/data/ollama" \
OLLAMA_HOST="0.0.0.0:11434" \
OLLAMA_ORIGINS="*" \
CUDA_VISIBLE_DEVICES=0,1 \
ollama serve
```
- `OLLAMA_HOST="0.0.0.0:11434"`: 关键参数，允许局域网连接。
- `OLLAMA_ORIGINS="*"`: 防止跨域 (CORS) 报错，建议加上。
开放防火墙端口 (非常重要) 如果不开放端口，客户端会报“连接被拒绝”。
 #### 清理端口占用
```bash
sudo netstat -tunlp | grep 11434
sudo kill -9 <PID>
```


```Bash
# 如果是用 ufw 防火墙
sudo ufw allow 11434/tcp

# 或者如果是 iptables
sudo iptables -I INPUT -p tcp --dport 11434 -j ACCEPT
```
安装完成后，启动 Ollama 的后台服务：
```Bash
ollama serve
```
### 3. 运行 Qwen 模型
打开一个新的终端窗口：
```Bash
ollama run alibayram/Qwen3-30B-A3B-Instruct-2507:latest
```

### 第四步：在局域网另一台电脑上调用
现在服务已经准备好了。
- **API 地址**: `http://172.16.1.95:11434/v1`
- **模型名称**: `alibayram/Qwen3-30B-A3B-Instruct-2507:latest`