# LoRA 在线挂载与多 Adapter 服务

vLLM 的一大杀手锏是支持 **Multi-LoRA Serving**。这意味着你可以在一个 vLLM 实例中加载一个 Base Model，并同时服务多个 LoRA Adapter。

## 1. 应用场景

在语音大模型领域，这非常有用：
*   **Base Model**: Qwen-Audio-Chat (通用能力)。
*   **Adapter A**: 针对医疗领域的微调（Medical ASR）。
*   **Adapter B**: 针对法律领域的微调（Legal ASR）。
*   **Adapter C**: 针对特定方言的微调（Dialect ASR）。

用户请求 A 进来时用 Adapter A，请求 B 进来时用 Adapter B，无需启动多个服务，极大节省显存。

---

## 2. 启用 LoRA

启动服务时需开启 LoRA 支持：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen-Audio-Chat \
    --enable-lora \
    --lora-modules medical=./lora_medical legal=./lora_legal
```

*   `--enable-lora`: 开启 LoRA 功能（会预留一部分显存给 LoRA 权重）。
*   `--lora-modules`: 定义别名和路径。格式为 `name=path`。

---

## 3. 请求级调用

在发送请求时，通过 `model` 参数指定使用哪个 LoRA。

```python
# 使用医疗 LoRA
client.chat.completions.create(
    model="medical",  # 对应启动参数中的别名
    messages=[...]
)

# 使用法律 LoRA
client.chat.completions.create(
    model="legal",
    messages=[...]
)

# 使用基座模型
client.chat.completions.create(
    model="Qwen/Qwen-Audio-Chat",
    messages=[...]
)
```

## 4. 性能影响

vLLM 使用名为 **S-LoRA** 的技术，将 LoRA 权重也纳入 PagedAttention 的管理机制中。
*   **开销**: 极低。切换 LoRA 几乎没有延迟。
*   **限制**: LoRA 的 Rank 不能太大（通常 <= 64），否则会影响吞吐量。
