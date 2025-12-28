# generate 解码参数与采样策略

`model.generate()` 是 Transformers 推理的核心入口。对于语音大模型，除了通用的文本生成参数（如 Temperature），还有一系列控制音频解码行为的专用参数。

## 1. Whisper 专用参数

Whisper 模型的 `generate` 方法封装了复杂的解码逻辑，包括语言识别、任务切换和时间戳预测。

### 1.1 核心控制参数
```python
generated_ids = model.generate(
    input_features,
    max_new_tokens=448,
    # 1. 任务控制
    task="transcribe",      # "transcribe" (转录) 或 "translate" (翻译成英语)
    language="zh",          # 强制指定语言，若不指定则自动检测
    
    # 2. 时间戳控制
    return_timestamps=True, # 输出带时间戳的文本
    
    # 3. 采样策略
    do_sample=False,        # ASR 通常使用 Greedy Search 以保证准确性
    num_beams=1,            # Beam Search 宽度
)
```

### 1.2 强制解码器输入 (`forced_decoder_ids`)
有时我们需要强制模型以特定的 Token 开头（例如强制输出简体中文而非繁体）。
```python
# 获取 tokenizer
tokenizer = processor.tokenizer

# 强制以 <|zh|> 开头
forced_decoder_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")

output = model.generate(
    input_features, 
    forced_decoder_ids=forced_decoder_ids
)
```

---

## 2. 采样策略 (Sampling Strategies)

### 2.1 Greedy Search (贪婪搜索)
*   **配置**: `do_sample=False`, `num_beams=1`
*   **适用场景**: 绝大多数 ASR 任务。语音识别追求的是“准确还原”，而不是“多样性创作”。贪婪搜索通常能提供最稳定的结果。

### 2.2 Beam Search (束搜索)
*   **配置**: `num_beams=5`
*   **适用场景**: 极难识别的音频或需要极高准确率的离线场景。
*   **代价**: 计算量随 Beam 宽度线性增加，显著降低推理速度。
*   **Whisper 特性**: Whisper 论文中提到，在低置信度时会自动回退到 Beam Search (Temperature Fallback)，但在 HF 实现中通常需要手动配置。

### 2.3 Temperature Sampling
*   **配置**: `do_sample=True`, `temperature=0.8`
*   **适用场景**: 语音对话生成（Speech-to-Speech）或多模态创作（Qwen-Audio 续写故事）。此时需要模型有一定的创造性。

---

## 3. 常见解码问题与参数调优

### 3.1 重复生成 (Repetition Loop)
ASR 模型有时会陷入死循环，不断重复同一个短语。
*   **解决**:
    *   `repetition_penalty`: 设置为 1.1 或 1.2（慎用，可能导致漏词）。
    *   `no_repeat_ngram_size`: 禁止重复的 N-gram。
    *   **Whisper 专用**: `condition_on_prev_tokens=False`。Whisper 默认会将上一句的解码结果作为下一句的 Prompt，如果上一句错了，会导致错误累积。关闭此选项可阻断错误传播。

### 3.2 幻觉 (Hallucination)
在静音段，Whisper 可能会“脑补”出奇怪的文本（如 "字幕组制作..."）。
*   **解决**:
    *   使用 VAD (Voice Activity Detection) 预处理，切除静音段。
    *   调高 `logprob_threshold`，过滤低置信度的输出。

---

## 4. 完整示例：带参数的推理

```python
# 准备输入
inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt").to(device)

# 执行生成
predicted_ids = model.generate(
    inputs.input_features,
    max_new_tokens=256,
    language="zh",
    task="transcribe",
    do_sample=False,        # 确定性输出
    num_beams=1,
    return_timestamps=True  # 获取时间戳
)

# 解码为文本
transcription = processor.batch_decode(
    predicted_ids, 
    skip_special_tokens=True,
    output_offsets=True     # 如果需要解析时间戳，这里要配合处理
)

print(transcription)
```
