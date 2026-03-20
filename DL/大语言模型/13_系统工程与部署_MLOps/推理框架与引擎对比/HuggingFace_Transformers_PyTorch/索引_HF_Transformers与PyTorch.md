# HuggingFace Transformers 与 PyTorch 推理

本章节详细介绍了使用原生 Transformers 库进行语音大模型推理的全流程，从模型加载到服务化封装。

## 目录

### [01_模型加载_AutoModel_AutoTokenizer.md](./01_模型加载_AutoModel_AutoTokenizer.md)
- **核心组件**：`AutoProcessor` (Feature Extractor + Tokenizer)。
- **模型类**：`AutoModelForSpeechSeq2Seq` (Whisper), `AutoModelForCausalLM` (Qwen-Audio)。
- **实战**：加载 Whisper Large v3 与 Qwen-Audio-Chat。

### [02_generate解码参数_采样策略.md](./02_generate解码参数_采样策略.md)
- **Whisper 参数**：`task` (transcribe/translate), `language`, `return_timestamps`。
- **采样策略**：Greedy Search (ASR 默认), Beam Search (高精度), Temperature Sampling (创造性)。
- **常见问题**：重复生成与幻觉的抑制。

### [03_批处理与吞吐_DataLoader与动态padding.md](./03_批处理与吞吐_DataLoader与动态padding.md)
- **Padding**：Waveform Padding vs Feature Padding。
- **DataCollator**：自定义 Collator 处理变长音频特征。
- **优化**：Length Bucketing (按长度分桶) 减少无效计算。

### [04_KVCache与attention_mask实践.md](./04_KVCache与attention_mask实践.md)
- **KV Cache**：加速自回归解码，显存占用分析。
- **Attention Mask**：Encoder 端屏蔽静音，Decoder 端屏蔽 Padding。
- **Prefix Sharing**：复用 System Prompt 的 KV Cache。

### [05_推理优化_amp_compile_sdpa_flashattn.md](./05_推理优化_amp_compile_sdpa_flashattn.md)
- **Flash Attention 2**：长序列音频推理的加速利器。
- **torch.compile**：编译 Whisper Encoder 提升速度。
- **AMP**：FP16/BF16 混合精度推理。

### [06_工程化_权重格式与safetensors.md](./06_工程化_权重格式与safetensors.md)
- **Safetensors**：零拷贝加载 (mmap)，提升冷启动速度，杜绝 Pickle 风险。
- **转换与切分**：将旧版 `.bin` 转换为 `.safetensors` 并分片。

### [07_与Serving衔接_fastapi_openai兼容.md](./07_与Serving衔接_fastapi_openai兼容.md)
- **FastAPI**：构建 `/v1/audio/transcriptions` 接口。
- **并发**：`async def` 与 GPU 线程阻塞问题。
- **部署**：Uvicorn Worker 设置。

