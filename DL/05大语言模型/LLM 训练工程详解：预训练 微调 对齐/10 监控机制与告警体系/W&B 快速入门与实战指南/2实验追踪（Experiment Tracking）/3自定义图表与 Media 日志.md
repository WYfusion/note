# 自定义图表与 Media 日志

W&B 支持远超标量的丰富媒体类型，特别适合 NLP / TTS / 多模态模型的输出可视化。

---

## 支持的媒体类型一览

| **类型** | **API** | **典型场景** |
| --- | --- | --- |
| 图片 | `wandb.Image` | 注意力图、生成图片、混淆矩阵 |
| 音频 | `wandb.Audio` | TTS 合成音频、ASR 原始音频 |
| 视频 | `wandb.Video` | 视频生成、强化学习 Agent |
| 表格 | `wandb.Table` | 预测对比、数据样本展示 |
| HTML | `wandb.Html` | 交互式可视化、自定义渲染 |
| Plotly | `wandb.Plotly` | 交互式图表 |
| 3D 对象 | `wandb.Object3D` | 点云、3D 模型 |
| 直方图 | `wandb.Histogram` | 参数/梯度/激活值分布 |

## 图片日志
```python
import numpy as np

# 从 numpy array
img = np.random.rand(256, 256, 3)
wandb.log({"image": wandb.Image(img, caption="Random noise")})

# 从 PIL Image
from PIL import Image
pil_img = Image.open("attention_map.png")
wandb.log({"attention": wandb.Image(pil_img)})

# 从 matplotlib figure
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(attention_weights, cmap="viridis")
ax.set_title("Attention Map")
wandb.log({"attention_plot": wandb.Image(fig)})
plt.close(fig)

# 批量图片
wandb.log({"samples": [
    wandb.Image(img, caption=f"Sample {i}")
    for i, img in enumerate(batch_images[:8])
]})
```

## 音频日志（TTS/ASR 重点）
```python
import numpy as np

# 从 numpy waveform
waveform = np.random.randn(22050)  # 1 秒 22050Hz
wandb.log({"audio": wandb.Audio(
    waveform,
    sample_rate=22050,
    caption="Generated speech",
)})

# 从文件路径
wandb.log({"reference": wandb.Audio("ref.wav", caption="Reference")})

# TTS 对比：参考音频 vs 生成音频
wandb.log({
    "tts/reference": wandb.Audio(ref_wav, sample_rate=22050, caption="Reference"),
    "tts/generated": wandb.Audio(gen_wav, sample_rate=22050, caption="Generated"),
    "tts/mel_spec": wandb.Image(plot_mel(gen_wav), caption="Mel Spectrogram"),
})
```

## Table（结构化对比）
```python
# 基础用法
table = wandb.Table(columns=["Input", "Prediction", "Ground Truth", "Score"])
for sample in eval_samples:
    table.add_data(
        sample["input"],
        sample["prediction"],
        sample["ground_truth"],
        sample["score"],
    )
wandb.log({"eval_results": table})

# 带媒体的 Table
table = wandb.Table(columns=["Text", "Audio", "Mel", "MOS"])
for s in tts_samples:
    table.add_data(
        s["text"],
        wandb.Audio(s["audio"], sample_rate=22050),
        wandb.Image(s["mel_plot"]),
        s["mos_score"],
    )
wandb.log({"tts_eval": table})
```

## Plotly 交互式图表
```python
import plotly.express as px
import plotly.graph_objects as go

# 超参-指标关系散点图
df = get_sweep_results()  # DataFrame
fig = px.scatter(
    df, x="lr", y="val_loss",
    color="method", size="batch_size",
    log_x=True, title="LR vs Val Loss",
)
wandb.log({"param_analysis": wandb.Plotly(fig)})

# 混淆矩阵
fig = px.imshow(
    confusion_matrix,
    labels=dict(x="Predicted", y="Actual"),
    x=class_names, y=class_names,
)
wandb.log({"confusion_matrix": wandb.Plotly(fig)})
```

## 自定义图表（W&B Custom Charts）
通过 Vega-Lite 规范定义：
```python
# 创建自定义图表数据
data = [[x, y] for x, y in zip(range(100), losses)]
table = wandb.Table(data=data, columns=["step", "loss"])

# 使用内置 preset
wandb.log({
    "loss_over_time": wandb.plot.line(
        table, "step", "loss",
        title="Loss Over Time",
    )
})

# 散点图
wandb.log({
    "lr_vs_loss": wandb.plot.scatter(
        table, "lr", "val_loss",
        title="Learning Rate vs Validation Loss",
    )
})

# PR 曲线
wandb.log({"pr_curve": wandb.plot.pr_curve(
    y_true, y_scores, labels=class_names,
)})

# ROC 曲线
wandb.log({"roc": wandb.plot.roc_curve(
    y_true, y_scores, labels=class_names,
)})
```

## 直方图
```python
# 记录参数/梯度分布
for name, param in model.named_parameters():
    if param.grad is not None:
        wandb.log({
            f"gradients/{name}": wandb.Histogram(param.grad.cpu().numpy()),
            f"params/{name}": wandb.Histogram(param.data.cpu().numpy()),
        })

# 记录 token 长度分布
lengths = [len(tokenizer(t)["input_ids"]) for t in texts]
wandb.log({"token_lengths": wandb.Histogram(lengths, num_bins=50)})
```
---

*← 返回：[[1实验追踪（Experiment Tracking）]]*