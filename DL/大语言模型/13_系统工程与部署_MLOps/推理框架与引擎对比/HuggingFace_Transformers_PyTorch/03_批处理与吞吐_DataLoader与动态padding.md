# 批处理与吞吐：DataLoader 与动态 Padding

在生产环境中，为了最大化 GPU 利用率（Throughput），我们通常需要将多个音频请求打包成一个 Batch 进行并行推理。

## 1. 音频数据的 Padding 挑战

与文本不同，音频数据的长度差异巨大（有的 1 秒，有的 30 秒）。
*   **Waveform Padding**: 在原始波形层面补 0。
*   **Feature Padding**: 在 Log-Mel Spectrogram 层面补 0。

Whisper 模型通常要求输入特征长度固定（30秒，3000帧）。如果音频短于 30秒，必须 Pad；如果长于 30秒，必须截断或切片。

### 1.1 静态 Padding vs 动态 Padding
*   **静态 Padding**: 所有样本都 Pad 到固定长度（如 30s）。简单，但浪费计算资源。
*   **动态 Padding**: Pad 到当前 Batch 中最长样本的长度。对于 Transformer 类模型（如 Qwen-Audio），这能显著减少计算量。

---

## 2. 实战：构建高效的 DataLoader

我们使用 PyTorch 的 `Dataset` 和 `DataLoader` 配合 Transformers 的 `DataCollator`。

### 2.1 定义 Dataset
```python
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, audio_paths, processor):
        self.audio_paths = audio_paths
        self.processor = processor

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        # 加载音频 (假设已重采样到 16k)
        audio, _ = librosa.load(self.audio_paths[idx], sr=16000)
        
        # 预处理：提取特征
        # 注意：这里不要转 tensor，返回 list 方便 collator 处理
        input_features = self.processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="np" # 返回 numpy
        ).input_features[0] 
        
        return {"input_features": input_features}
```

### 2.2 自定义 DataCollator
对于 Whisper，我们需要将不同长度的特征堆叠起来。虽然 Whisper 默认 Pad 到 30s，但在微调或特定配置下可能需要动态处理。

```python
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 提取 input_features
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        
        # 使用 processor 自带的 pad 方法
        # 对于 Whisper，这通常会将所有样本 Pad 到 30s (3000 frames)
        batch = self.processor.feature_extractor.pad(
            input_features, 
            return_tensors="pt"
        )
        
        return batch
```

### 2.3 启动 DataLoader
```python
dataset = AudioDataset(audio_files, processor)
collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=16,      # 根据显存大小调整
    collate_fn=collator,
    num_workers=4,      # 多进程加载数据，避免 CPU 瓶颈
    pin_memory=True     # 加速 CPU -> GPU 传输
)

# 推理循环
model.eval()
for batch in dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        generated_ids = model.generate(**batch)
    # ... decode ...
```

---

## 3. 吞吐量优化技巧

### 3.1 长度分桶 (Length Bucketing/Sorting)
如果 Batch 内音频长度差异过大（如 1s 和 30s 在一起），短音频会被 Pad 大量的 0，造成浪费。
*   **策略**: 在构建 Batch 前，先按音频长度对数据进行排序。
*   **效果**: 使得同一个 Batch 内的音频长度相近，减少 Padding 比例。

### 3.2 混合精度 (Mixed Precision)
始终使用 `torch.cuda.amp.autocast` 或直接加载 `float16` 模型。
```python
with torch.cuda.amp.autocast(dtype=torch.float16):
    generated_ids = model.generate(**batch)
```

### 3.3 显存估算
Whisper Large v3 (1.5B) 在 Batch Size = 1 时约占用 4GB 显存（FP16）。
随着 Batch Size 增加，显存占用主要来自：
1.  **Input Features**: 相对较小。
2.  **KV Cache**: 随 Batch Size 和生成长度线性增长。
3.  **Intermediate Activations**: 仅在训练时占用，推理时较少。

**经验法则**: 24GB 显存 (3090/4090) 通常可支持 Batch Size 32-64 的 Whisper Large 推理。
