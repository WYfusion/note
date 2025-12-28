# KV Cache 与 Attention Mask 实践

在自回归生成（Autoregressive Generation）中，KV Cache 是加速推理的关键技术。对于 Whisper、Qwen-Audio 等包含 Decoder 的模型，理解 KV Cache 和 Attention Mask 的工作机制至关重要。

## 1. KV Cache 原理

Transformer Decoder 在生成第 $t$ 个 Token 时，需要计算它与之前所有 Token ($1$ 到 $t-1$) 的 Attention。
如果不缓存，每次生成新 Token 都要重新计算前面所有 Token 的 Key 和 Value 矩阵，计算复杂度为 $O(N^2)$。
使用 KV Cache 后，我们只计算当前 Token 的 K、V，并将其拼接到缓存中，复杂度降为 $O(N)$。

### 1.1 在 Transformers 中启用
默认情况下，`model.generate()` 会自动启用 KV Cache (`use_cache=True`)。

```python
# 显式控制
outputs = model.generate(
    input_features, 
    use_cache=True  # 默认开启
)
```

### 1.2 显存占用分析
KV Cache 占用的显存与以下因素成正比：
$$
\text{Mem} \propto \text{Batch\_Size} \times \text{Seq\_Len} \times \text{Layers} \times \text{Hidden\_Dim}
$$
对于长音频生成（如长篇演讲转录），Seq_Len 会不断增长，导致显存持续增加。

---

## 2. Attention Mask 的作用

Attention Mask 用于告诉模型“哪些部分是有效的，哪些是 Padding”。

### 2.1 Encoder 端的 Mask
对于 Whisper，输入特征通常被 Pad 到 30秒。
*   如果输入音频只有 5秒，剩下的 25秒是 Padding。
*   **问题**: 如果不 Mask，Encoder 会处理这 25秒的静音/噪声，可能导致 Decoder 产生幻觉（Hallucination）。
*   **解决**: 构造 `attention_mask` 传递给模型。

```python
# processor 自动生成 attention_mask
inputs = processor(
    [audio_5s, audio_30s], 
    sampling_rate=16000, 
    return_tensors="pt",
    padding=True
)

# inputs 包含:
# - input_features: [2, 80, 3000]
# - attention_mask: [2, 3000] (0 表示 padding, 1 表示有效)

outputs = model.generate(
    inputs.input_features,
    attention_mask=inputs.attention_mask  # 务必传入！
)
```

### 2.2 Decoder 端的 Mask
在 Batch 推理时，不同样本生成的文本长度不同。
*   Transformers 会自动维护 Decoder 的 Attention Mask，确保预测下一个 Token 时只关注已生成的 Token，而不关注 Padding。

---

## 3. 高级技巧：Prefix Sharing (Prompt Caching)

在构建语音 Agent 时，System Prompt（例如“你是一个语音助手...”）通常是固定的。
*   **Prefix Sharing**：所有用户的请求都可以共享这一段 System Prompt 的 KV Cache Block。
*   对于包含长 System Prompt 的 Audio LLM 尤为有效。

虽然 HuggingFace Transformers 原生对 Prefix Sharing 的支持不如 vLLM 完善，但可以通过手动传递 `past_key_values` 来实现简单的缓存复用。

```python
# 1. 预计算 System Prompt 的 KV Cache
system_prompt = "You are a helpful assistant."
system_inputs = tokenizer(system_prompt, return_tensors="pt")
with torch.no_grad():
    system_outputs = model(
        **system_inputs, 
        use_cache=True
    )
past_key_values = system_outputs.past_key_values

# 2. 处理用户请求时复用
user_input = "Play some music."
user_inputs = tokenizer(user_input, return_tensors="pt")

# 拼接 attention_mask 等逻辑较复杂，通常建议使用 vLLM 处理此类场景
```
