音频嵌入将声音信号映射为向量，覆盖语音内容、说话人身份、声学事件和音乐风格四个细分方向。

<aside>
💡

**通俗理解**：给声音一个"语义指纹"——不只听到了什么词（语音内容），还能分辨是谁在说（说话人），是什么声音（事件），以及什么风格（音乐）。

</aside>

---

## 四大细分方向

### 语音表征嵌入

- **代表**：wav2vec 2.0 系列
- **原理**：自监督预训练，学习语音的通用表征
- **用途**：ASR 前表征、语音分类、语音检索、语音理解

### 说话人嵌入

- **代表**：x-vector、ECAPA-TDNN
- **原理**：从音频中提取说话人身份特征
- **用途**：说话人识别、验证、聚类、分离

### 音频-文本共享嵌入

- **代表**：CLAP、T-CLAP、SLAP
- **原理**：音频和文本编码到同一空间（类似 CLIP 对图像做的事）
- **用途**：text↔audio 检索、开放词汇音频分类、音频生成评估

### 细分对象

- **语音内容嵌入**（what is said）：理解说了什么
- **说话人嵌入**（who speaks）：识别谁在说
- **声学事件嵌入**（what sound/event）：狗叫、警笛、雷声……
- **音乐嵌入**（style / mood / tag）：曲风、情绪、标签

---

## Python 实战：CLAP 音频-文本检索

```python
import torch
import librosa
from transformers import ClapModel, ClapProcessor

# 加载 CLAP 模型
model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

# 加载音频
audio, sr = librosa.load("dog_bark.wav", sr=48000)

# 文本描述候选
texts = ["a dog barking", "a cat meowing", "thunder and rain", "people talking"]

# 编码音频
audio_inputs = processor(audios=[audio], sampling_rate=48000, return_tensors="pt")
with torch.no_grad():
    audio_embed = model.get_audio_features(**audio_inputs)
    audio_embed = audio_embed / audio_embed.norm(dim=-1, keepdim=True)

# 编码文本
text_inputs = processor(text=texts, return_tensors="pt", padding=True)
with torch.no_grad():
    text_embed = model.get_text_features(**text_inputs)
    text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

# 计算相似度
similarity = (audio_embed @ text_embed.T).squeeze()
for text, score in zip(texts, similarity):
    print(f"{text}: {score.item():.4f}")
```

---

## wav2vec 2.0 语音表征提取

```python
import torch
import librosa
from transformers import Wav2Vec2Model, Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

# 加载音频
audio, sr = librosa.load("speech.wav", sr=16000)
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    # 取最后一层隐状态的均值作为句级嵌入
    embedding = outputs.last_hidden_state.mean(dim=1)  # [1, 768]

print(f"语音嵌入维度: {embedding.shape}")
```