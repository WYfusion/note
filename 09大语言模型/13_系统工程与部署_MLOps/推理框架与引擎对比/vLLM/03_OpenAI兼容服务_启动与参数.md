# OpenAI 兼容服务：启动与参数

vLLM 自带了一个高性能的 API Server，完全兼容 OpenAI 的 Chat Completions API 协议。这意味着你可以直接使用 `openai-python` 库或 LangChain 来调用它。

## 1. 启动服务

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen-Audio-Chat \
    --trust-remote-code \
    --port 8000
```

### 1.1 关键参数
*   `--model`: 模型名称或本地路径。
*   `--trust-remote-code`: 对于 Qwen-Audio 等包含自定义代码的模型，必须开启。
*   `--tensor-parallel-size (tp)`: 张量并行度。如果你有 2 张卡，想跑一个大模型，设为 2。
*   `--gpu-memory-utilization`: 显存占用比例。默认为 0.9。如果出现 OOM，可适当调低（如 0.85）。
*   `--max-model-len`: 最大上下文长度。如果显存不足，可限制此值（如 4096）。

---

## 2. 调用 Audio LLM (以 Qwen-Audio 为例)

虽然 vLLM 的接口是标准的 Chat Completions，但对于多模态模型，我们需要按照特定的格式传入音频信息。

### 2.1 客户端代码示例
Qwen-Audio 接受音频 URL 或本地路径作为输入。

```python
from openai import OpenAI

# 指向 vLLM 服务地址
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

# 构造 Prompt
# Qwen-Audio 的格式通常是：Audio + Text
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Audio 1: "},
            {"type": "audio_url", "audio_url": "http://example.com/test.wav"},
            {"type": "text", "text": "这段音频里的人在说什么？"}
        ]
    }
]

# 注意：vLLM 对多模态输入的格式支持正在快速演进
# 如果上述结构化输入不支持，可能需要回退到纯文本 Prompt 格式：
# "Picture 1: <img>http://...</img>\nDescribe it."
# 具体需参考 vLLM 对应版本的文档。

response = client.chat.completions.create(
    model="Qwen/Qwen-Audio-Chat",
    messages=messages,
    temperature=0.1
)

print(response.choices[0].message.content)
```

---

## 3. 离线推理 (Offline Inference)

如果你不需要 HTTP 服务，只是想跑批量数据（如离线评测），直接使用 Python API 效率更高。

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen-Audio-Chat", trust_remote_code=True)

prompts = [
    "Audio 1: <audio>test1.wav</audio> Transcribe it.",
    "Audio 2: <audio>test2.wav</audio> Analyze emotion."
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```
