## 前置知识

> [!important]
> 
> 本页面向希望**实际使用** Qwen3-TTS 的工程师与研究者。建议先通读主页以了解各变体差异。

---

## 0. 定位

> 本页覆盖 Qwen3-TTS 的开源资源、模型选型、环境搭建、常用 API 调用、微调与评测流程，形成一份**端到端实战手册**。

---

## 1. 开源资源清单

|**资源**|**链接**|**内容**|**论文**|[arXiv:2601.15621](https://arxiv.org/abs/2601.15621)|技术报告全文|
|---|---|---|---|---|---|
|**GitHub**|[QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)|推理代码、示例、配置|**HuggingFace**|[Qwen/qwen3-tts](https://huggingface.co/collections/Qwen/qwen3-tts)|模型权重（8 个变体）|
|**ModelScope**|[Qwen/Qwen3-TTS](https://modelscope.cn/collections/Qwen/Qwen3-TTS)|国内镜像|**许可证**|Apache 2.0|商业可用|

---

## 2. 模型变体选型

![[2026-04-18 09.29.28TTS模型变体选型.excalidraw|1000]]

### 2.1 八个变体一览

|**模型**|**克隆**|**指令**|**首包**|**推荐场景**|
|---|---|---|---|---|
|12Hz-0.6B-Base|✅||97 ms|边缘设备|
|12Hz-1.7B-CustomVoice||✅|~105 ms|定制风格|
|25Hz-0.6B-Base|✅||138 ms|边缘 + 质量|
|25Hz-1.7B-CustomVoice||✅|~155 ms|高质定制|

---

## 3. 环境准备

### 3.1 硬件要求

|**模型规格**|**FP16 显存**|**最低 GPU**|
|---|---|---|
|1.7B|~4 GB|RTX 4080 16GB|
|1.7B + 并发 6|~24 GB|A100 40GB|

### 3.2 依赖安装

```bash
# 创建环境
conda create -n qwen3tts python=3.10 -y
conda activate qwen3tts

# 核心依赖
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.44.0 accelerate
pip install vllm>=0.6.0  # 可选，用于高性能服务

# Qwen3-TTS 仓库
git clone https://github.com/QwenLM/Qwen3-TTS.git
cd Qwen3-TTS
pip install -e .

# 音频处理
pip install soundfile librosa pydub
```

---

## 4. 快速上手

### 4.1 零样本克隆（最常见用法）

```python
import torchaudio
from qwen3_tts import Qwen3TTS

# 加载模型（首次运行自动从 HF 下载）
model = Qwen3TTS.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device="cuda",
    dtype="bfloat16",
)

# 读取 3 秒参考音频（任意采样率，模型内部重采样到 16kHz）
ref_wav, sr = torchaudio.load("my_voice_3s.wav")

# 合成
audio = model.synthesize(
    text="今天的天气真不错，适合出门走走。",
    reference_audio=ref_wav,
    reference_sr=sr,
    language="zh",
)

# 保存（24kHz 输出）
torchaudio.save("output.wav", audio.unsqueeze(0), 24000)
```

### 4.2 Voice Design（自然语言描述造声）

```python
model = Qwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")

audio = model.synthesize(
    text="Good morning, everyone!",
    voice_description="A middle-aged female BBC news anchor with a warm, confident tone and moderate pace.",
    language="en",
)
```

### 4.3 流式输出（低延迟交互）

```python
import sounddevice as sd

# 启用流式迭代器
stream = model.synthesize_streaming(
    text="流式合成可以边生成边播放，用户几乎无感知延迟。",
    reference_audio=ref_wav,
    chunk_size_ms=80,  # 12Hz 一帧 80ms
)

# 逐包播放
with sd.OutputStream(samplerate=24000, channels=1) as out:
    for audio_chunk in stream:
        out.write(audio_chunk.numpy())
```

---

## 5. 细粒度控制（CustomVoice）

### 5.1 控制指令格式

```python
model = Qwen3TTS.from_pretrained("Qwen/Qwen3-TTS-25Hz-1.7B-CustomVoice")

# 使用预置 speaker + 细粒度指令
audio = model.synthesize(
    text="欢迎收听《今日要闻》。",
    speaker_id="anchor_female_01",  # 预置说话人
    instructions={
        "style": "新闻主播",
        "speed": "稍慢",
        "emotion": "庄重",
        "pitch": "中等偏低",
    },
)
```

### 5.2 指令生效机制

![[2026-04-18 09.30.27TTS设计指令生效机制.excalidraw|200]]

> [!important]
> 
> **Thinking Pattern** 有约 20–30% 概率被激活，使模型在生成前用文本**推理**指令意图。这牺牲少量延迟（几十 ms）换取复杂指令的跟随精度。

---

## 6. 微调（Speaker Fine-tuning）

### 6.1 数据准备

|**规模**|**推荐时长**|**预期效果**|轻量（LoRA）|10–30 分钟|音色基本复制|
|---|---|---|---|---|---|
|中等（LoRA + 数据增强）|1–3 小时|韵律+音色稳定|重度（全参微调）|5–10 小时|接近录音棚质量|

### 6.2 LoRA 微调示例

```python
from peft import LoraConfig, get_peft_model
from qwen3_tts import Qwen3TTS

base = Qwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(base.lm, lora_cfg)

# 其余训练循环同标准 HF Trainer
# 损失仅对语音 token 位置计算（text token 位置 mask 掉）
```

### 6.3 数据格式（ChatML）

```jsx
<|im_start|>system
You are a TTS assistant. Speaker: Alice (warm female voice).
<|im_end|>
<|im_start|>user
今天北京的气温是 15 度。
<|im_end|>
<|im_start|>assistant
<|speech_start|>[语音 token 序列]<|speech_end|>
<|im_end|>
```

---

## 7. 评测自己的结果

### 7.1 计算 WER（内容准确性）

```python
import whisper
from jiwer import wer

asr = whisper.load_model("large-v3")

def compute_wer(gen_audio_path: str, reference_text: str, lang: str):
    result = asr.transcribe(gen_audio_path, language=lang)
    hypothesis = result["text"]
    return wer(reference_text, hypothesis)
```

### 7.2 计算 SIM（参考 [[DL/TTS/Qwen3-TTS Technical Report/TTS 评测基准全景指南/Qwen3-TTS 评价指标详解|Qwen3-TTS 评价指标详解]]）

使用该页给出的 WavLM-SV 代码。

### 7.3 计算 UTMOS（自然度）

```python
import torch
from utmos import UTMOS

utmos = UTMOS.load_from_checkpoint("utmos_strong.ckpt").cuda().eval()

@torch.no_grad()
def compute_utmos(wav: torch.Tensor) -> float:
    # wav: (T,), 16kHz
    return utmos(wav.unsqueeze(0).cuda()).item()
```

---

## 8. 常见问题（FAQ）

> [!important]
> 
> **Q1：参考音频最短多长？**
> 
> A：官方推荐 3 秒，最短 1.5 秒也能工作但相似度下降。

> [!important]
> 
> **Q2：支持多少种语言？**
> 
> A：训练明确覆盖 10+ 语种（中、英、意、法、韩、俄、德、西、日、葡等）。未见语种可能仍工作但质量无保证。

> [!important]
> 
> **Q3：能本地部署到 CPU 吗？**
> 
> A：0.6B 配合 `llama.cpp` 量化到 Q4_K_M 后，M2 Max CPU 可达 RTF ≈ 0.8，勉强实时。

> [!important]
> 
> **Q4：是否支持 SSML？**
> 
> A：官方 repo 暂不原生支持 SSML，但可通过 CustomVoice 的 instructions 字段实现类似效果。

> [!important]
> 
> **Q5：可以用于商业产品吗？**
> 
> A：Apache 2.0 许可证，可商用。但训练数据中可能涉及的说话人权益由使用方负责。

---

## 9. 与主流方案的选型对比

|**方案**|**克隆**|**多语言**|**可控**|**首包**|**许可**|
|---|---|---|---|---|---|
|CosyVoice 3|3s|9 + 18 方言|部分|~200 ms|Apache 2.0|
|ElevenLabs|商业|30+|✅|~300 ms|商业 API|
|F5-TTS|3s|中英|否|~500 ms|CC-BY-NC|

---

## 延伸阅读

- [[DL/TTS/Qwen3-TTS Technical Report/Qwen3-TTS Technical Report|Qwen3-TTS Technical Report]]

- [[DL/TTS/Qwen3-TTS Technical Report/Qwen3-TTS 模型组件架构细节/Qwen3-TTS 模型组件架构细节|Qwen3-TTS 模型组件架构细节]]

---

## 参考文献

1. Qwen3-TTS GitHub README & examples/

1. HuggingFace Model Card — Qwen/Qwen3-TTS-12Hz-1.7B-Base

1. Apache 2.0 License Text.