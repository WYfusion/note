# 实验追踪（Experiment Tracking）

W&B 最核心的功能——记录训练过程中的一切指标，实现实验可复现、可对比。

---

## 核心 API 一览

| **API** | **作用** | **调用时机** |
| --- | --- | --- |
| `wandb.log(dict)` | 记录标量/图表/媒体到当前 step | 每个 step / epoch |
| `wandb.log(dict, step=N)` | 指定 step 记录（避免 step 递增） | 自定义 step 逻辑 |
| `wandb.define_metric()` | 自定义指标的 X 轴和聚合方式 | `wandb.init` 之后 |
| `run.summary["key"]` | 手动设置 Summary 值（最终指标） | 训练结束 |
| `wandb.watch(model)` | 自动记录梯度和参数分布 | `wandb.init` 之后 |

## 常见指标分组
使用 `/` 分隔符自动分组到不同面板：
```python
wandb.log({
    # 训练指标 → "train" 面板
    "train/loss": 0.35,
    "train/lr": 1e-4,
    "train/grad_norm": 1.2,

    # 验证指标 → "val" 面板
    "val/loss": 0.42,
    "val/accuracy": 0.88,

    # 吞吐量 → "throughput" 面板
    "throughput/tokens_per_sec": 15000,
    "throughput/samples_per_sec": 32,

    # 系统 → "system" 面板
    "system/gpu_util": 95.2,
    "system/gpu_mem_gb": 38.5,
})
```
## 自定义 X 轴

默认 X 轴是 `_step`（自增），可以自定义：

```python
# 让所有 val/* 指标以 epoch 为 X 轴
wandb.define_metric("epoch")
wandb.define_metric("val/*", step_metric="epoch")

for epoch in range(num_epochs):
    # ... 训练 ...
    wandb.log({"epoch": epoch, "val/loss": val_loss})
```

## 媒体日志

W&B 不仅能记录标量，还支持丰富的媒体类型：

```python
# 图片
wandb.log({"samples": [wandb.Image(img, caption="pred") for img in images]})

# 音频（TTS/ASR 场景常用）
wandb.log({"audio": wandb.Audio(waveform, sample_rate=22050, caption="gen")})

# 表格（对比预测结果）
table = wandb.Table(columns=["input", "pred", "label"])
table.add_data("How are you?", "I'm fine.", "I am fine.")
wandb.log({"predictions": table})

# Matplotlib 图
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
wandb.log({"chart": wandb.Image(fig)})
plt.close(fig)
```

## 梯度与参数监控
```python
model = MyModel()
wandb.watch(model, log="all", log_freq=100)
# log="gradients" → 只记录梯度
# log="parameters" → 只记录参数
# log="all" → 梯度 + 参数
# log_freq → 每 N 步记录一次
```

## 系统指标自动采集
W&B 默认每 10 秒自动采集：
- GPU 利用率、显存、温度、功耗
- CPU 利用率、内存
- 磁盘 I/O、网络 I/O

无需额外代码，在 Dashboard 的 "System" 面板中查看。

## 子页面导航

- **[[2wandb log 指标记录全解]]** → 所有参数、step 控制、commit 机制
- **[[3自定义图表与 Media 日志]]** → Table / Image / Audio / Video / 3D / HTML
- **[[4系统监控与 GPU 追踪]]** → 自动采集配置、自定义系统指标

---

*← 上一节：[[W&B 核心概念与环境搭建]]　|　下一节：[[1超参数调优（Sweeps）]] →*
