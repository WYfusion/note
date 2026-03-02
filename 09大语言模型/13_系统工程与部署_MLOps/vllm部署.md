```bash
python -m vllm.entrypoints.openai.api_server \
    --model /home/rtx5090/data/wy/llm/pre_model_weights/models/Qwen/Qwen3-8B \
    --served-model-name qwen3-8b \
    --max-model-len 8k \
    --host 0.0.0.0 \
    --port 6006 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.8 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser deepseek_r1
```

- 其中的`host`参数是`0.0.0.0`，这可以让同一个局域网的主机调用，但是需要做好**防火墙放行**
### 直接放行 6006 端口 (推荐)
如果远程主机使用的是 `ufw` (常见于 Ubuntu/Debian):
```bash
sudo ufw allow 6006/tcp
sudo ufw reload
```
如果远程主机使用的是 `firewalld` (常见于 CentOS/RHEL):
```bash
sudo firewall-cmd --zone=public --add-port=6006/tcp --permanent
sudo firewall-cmd --reload
```
如果使用 `iptables`:
```bash
sudo iptables -I INPUT -p tcp --dport 6006 -j ACCEPT
```


```python
from openai import OpenAI
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:6006/v1"
# openai_api_base = "http://172.22.176.12:6006/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_reasonse = client.chat.completions.create(
    model="qwen3-8b",
    messages=[{"role":"user","content":"介绍一下你自己"}],
    temperature=1.5,
    top_p=0.8,
    # max_tokens=8192,
    presence_penalty=1.5,
    extra_body={"chat_template_kwargs":{"enable_thinking":False}},
)

print("Chat response:",chat_reasonse)
```