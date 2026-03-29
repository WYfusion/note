# KV Cache 与 Attention Mask 实践

在自回归生成（Autoregressive Generation）中，KV Cache 是加速推理的关键技术。对于 Whisper、Qwen-Audio 等包含 Decoder 的模型，理解 KV Cache 和 Attention Mask 的工作机制至关重要。

## 1. KV Cache 原理

Transformer Decoder 在生成第 $t$ 个 Token 时，需要计算它与之前所有 Token ($1$ 到 $t-1$) 的 Attention。
如果不缓存，每次生成新 Token 都要重新计算前面所有 Token 的 Key 和 Value 矩阵，计算复杂度为 $O(N^2)$。
使用 KV Cache 后，我们只计算当前 Token 的 K、V，并将其拼接到缓存中，复杂度降为 $O(N)$。
### 1. 数学视角的解释：因果关系与矩阵运算

在 Transformer 的自注意力机制（Self-Attention）中，核心公式是：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
假设我们正在生成第 $t$ 个 token（也就是你的描述中的“当前”时刻）：

#### 关于 Q (Query) - 为什么不需要缓存？
- **当前视角**：$Q_t$ 是由当前刚刚生成的 token $x_t$ 经过线性变换 $W_q$ 得到的向量。它的任务是代表“现在”，去之前的历史信息中“搜寻”相关内容。
- **用完即弃**：一旦根据 $Q_t$ 计算出了注意力权重并聚合了信息，进而预测出了下一个 token $x_{t+1}$，那么 $Q_t$ 的使命就彻底结束了。
- **无回溯性**：当我们进行到第 $t+1$ 步时，我们需要的是新的查询 $Q_{t+1}$。我们永远不会再说：“嘿，我想看看第 $t-5$ 步时的那个 token 当时在关注什么”，因为那已经是**既定事实**（已经算完了）。因此，旧的 Q 没有保留价值。

#### 关于 K, V (Key, Value) - 为什么必须缓存？
- **被关注的对象**：当我们在第 $t$ 步时，我们需要计算 $Q_t$ 与之前所有时刻 $1$ 到 $t-1$ 的相似度。这意味着我们需要 $K_1, K_2, ..., K_{t-1}$。
- **重复利用**：
    - 在第 $t$ 步，我们需要 $K_1...K_{t-1}$。
    - 在第 $t+1$ 步，我们需要 $K_1...K_{t-1}, K_t$。
    - 在第 $t+2$ 步，我们需要 $K_1...K_{t-1}, K_t, K_{t+1}$。
- 如果不缓存：每次生成一个新 token，我们都得把前面几百几千个 token 重新拿出来，重新乘一遍投影矩阵 $W_k$ 和 $W_v$。这将导致**巨大的重复计算浪费**。KV Cache 的本质就是**空间换时间**。

### 2. 形象化的比喻：图书馆管理员
想象你在写一篇论文（生成文本）：
- **Q (Query) 是你当下的念头**：
    - 你现在写到了“苹果”这个词，你的脑海里（$Q$）在想：“下个词该接什么？我要找关于‘颜色’或‘口感’的信息。”
    - 当你写完这句，开始写下一句时，你会有**新**的念头。你不需要把“刚才那个念头”存下来，因为那一页已经翻过去了。
- **K (Key) 是书架上书的标题/标签**：
    - 书架上摆着你之前写过的所有章节（历史 Context）。
    - 你需要看书脊（$K$）来判断哪本书跟你现在的念头（$Q$）有关。
- **V (Value) 是书里的具体内容**：
    - 一旦你通过标题（$K$）找到了相关的书，你就把书里的内容（$V$）拿出来参考。
- **KV Cache (缓存)**：
    - **没有缓存**：每写一个字，你都要把书架上那 1000 本书重新买一遍、重新拆封、重新贴标签上架。这太慢了！
    - **有缓存**：书架一直在那里。每当你写出一个新词，你把这个新词做成一本书，贴上标签（计算出 $K, V$），插到书架的最末尾。以后再查阅时，直接扫视整个书架即可。
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

