# 模型加载：AutoModel 与 AutoProcessor

在 HuggingFace Transformers 生态中，加载模型是推理的第一步。对于语音大模型（Audio LLM），除了常规的 `AutoModel` 和 `AutoTokenizer`，我们更频繁地与 `AutoProcessor` 和 `AutoFeatureExtractor` 打交道，因为语音任务涉及音频信号的预处理。

## 1. 核心组件解析

### 1.1 AutoModel 类族
针对不同的语音任务，Transformers 提供了特定的 Head 封装：
*   **`AutoModel`**: 加载裸模型（无 Head），输出 Hidden States。
*   **`AutoModelForSpeechSeq2Seq`**: 适用于 Whisper 等语音到文本（ASR/ST）模型。
*   **`AutoModelForAudioClassification`**: 适用于音频分类（如情感识别）。
*   **`AutoModelForCausalLM`**: 适用于 Qwen-Audio 等多模态生成模型（本质是 LLM 接了 Audio Encoder）。

### 1.2 预处理组件
语音模型通常需要两个预处理步骤：
1.  **Feature Extractor**: 将原始音频波形（Waveform）转换为声学特征（如 Log-Mel Spectrogram）。
2.  **Tokenizer**: 处理文本输入（Prompt）或解码输出。

**`AutoProcessor`** 是这两者的封装，它将音频处理和文本处理统一到一个接口中，极大简化了代码。

$$
\text{Processor}(Audio, Text) \rightarrow \{ \text{input\_features}, \text{input\_ids}, \text{attention\_mask} \}
$$

---

## 2. 实战：加载 Whisper 模型

Whisper 是典型的 Encoder-Decoder 架构。

### 2.1 标准加载流程
```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

model_id = "openai/whisper-large-v3"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# 1. 加载模型
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, 
    torch_dtype=torch_dtype, 
    low_cpu_mem_usage=True, 
    use_safetensors=True
)
model.to(device)

# 2. 加载处理器 (包含 FeatureExtractor 和 Tokenizer)
processor = AutoProcessor.from_pretrained(model_id)

print(f"Model loaded on {device} with {torch_dtype}")
```

### 2.2 参数详解
*   **`torch_dtype`**: 推荐使用 `float16` 或 `bfloat16` 以节省显存并加速计算（Whisper 对 FP16 鲁棒性很好）。
*   **`low_cpu_mem_usage=True`**: 逐步加载权重到内存，避免一次性占用过大 RAM 导致 OOM。
*   **`use_safetensors=True`**: 优先加载 `.safetensors` 格式权重，加载速度更快且更安全（无 pickle 风险）。

---

## 3. 实战：加载 Qwen-Audio (多模态 LLM)

Qwen-Audio 这类模型本质上是 Causal LM，它将音频特征作为特殊的 Embedding 插入到 LLM 的输入序列中。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

model_id = "Qwen/Qwen-Audio-Chat"

# 1. 加载 Tokenizer (Qwen-Audio 没有独立的 FeatureExtractor，集成在 Tokenizer 中处理)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# 2. 加载模型
# trust_remote_code=True 是必须的，因为 Qwen-Audio 包含自定义的模型代码
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",  # 自动分配 GPU/CPU
    trust_remote_code=True,
    torch_dtype=torch.float16
).eval()

# 3. 验证加载
print(model.generation_config)
```

### 3.1 `device_map="auto"`
对于大模型（如 7B, 70B），单卡可能放不下。`device_map="auto"` 会利用 HuggingFace Accelerate 库自动将模型切分到多张 GPU 甚至 CPU/Disk 上（Model Parallelism）。

---

## 4. 常见坑与解决方案

### 4.1 缺少 `ffmpeg`
`AutoProcessor` 在处理音频文件时通常依赖 `ffmpeg` 进行解码。
*   **报错**: `FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'`
*   **解决**: 系统需安装 ffmpeg (`apt install ffmpeg` 或 `conda install ffmpeg`)。

### 4.2 采样率不匹配
每个模型都有固定的采样率（Whisper 为 16000Hz）。
*   **注意**: `AutoProcessor` 通常不负责重采样（Resampling），或者行为不一致。
*   **最佳实践**: 使用 `librosa` 或 `torchaudio` 加载音频时显式指定采样率。
    ```python
    import librosa
    
    # 强制重采样到 16k
    audio_array, sampling_rate = librosa.load("audio.mp3", sr=16000)
    
    inputs = processor(
        audio_array, 
        sampling_rate=16000, 
        return_tensors="pt"
    ).to(device)
    ```
