视频嵌入将视频片段映射为向量，支持 text→video、video→text 和事件片段检索。

<aside>
💡

**通俗理解**：给视频片段一个"语义指纹"，让你能用文字描述搜到对应的视频片段，或找到内容相似的视频。

</aside>

---

## 两大方向

### 帧池化 / Clip 池化视频嵌入

- **原理**：用图像 backbone（如 ViT）分别编码每帧/每个 clip，然后做时序池化（mean/max pooling、attention pooling）
- **优点**：简单，可复用成熟的图像编码器
- **局限**：时序信息捕获有限

### 视频-文本共享嵌入

- **代表**：VideoCLIP、HowTo100M 系列、Video-ColBERT
- **原理**：视频和文本编码到同一空间，支持跨模态检索
- **用途**：text→video、video→text、事件片段检索、监控检索、教学视频检索

---

## 趋势

- 从"整视频单向量"走向"时空 token / 多向量 / late interaction"
- 更细粒度的时间定位（temporal grounding）
- 与音频、字幕的多模态融合

---

## Python 实战：帧池化视频嵌入

```python
import torch
import cv2
import numpy as np
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def extract_frames(video_path, num_frames=8):
    """均匀采样视频帧"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    return frames

def get_video_embedding(video_path, num_frames=8):
    """帧池化视频嵌入：编码每帧后取均值"""
    frames = extract_frames(video_path, num_frames)
    inputs = processor(images=frames, return_tensors="pt")
    with torch.no_grad():
        frame_embeds = model.get_image_features(**inputs)
        frame_embeds = frame_embeds / frame_embeds.norm(dim=-1, keepdim=True)
    # 均值池化
    video_embed = frame_embeds.mean(dim=0, keepdim=True)
    video_embed = video_embed / video_embed.norm(dim=-1, keepdim=True)
    return video_embed

def text_to_video_search(query, video_paths):
    """文本搜视频"""
    txt_inputs = processor(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        txt_embed = model.get_text_features(**txt_inputs)
        txt_embed = txt_embed / txt_embed.norm(dim=-1, keepdim=True)
    
    scores = []
    for vp in video_paths:
        v_embed = get_video_embedding(vp)
        score = (txt_embed @ v_embed.T).item()
        scores.append(score)
    
    ranked = sorted(zip(video_paths, scores), key=lambda x: -x[1])
    for path, score in ranked:
        print(f"{path}: {score:.4f}")

# 使用示例
text_to_video_search("a person cooking in the kitchen", ["v1.mp4", "v2.mp4", "v3.mp4"])
```