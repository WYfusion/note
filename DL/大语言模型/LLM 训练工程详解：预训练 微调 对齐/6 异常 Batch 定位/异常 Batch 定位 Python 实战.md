# 异常 Batch 定位 Python 实战

本页面提供完整的异常 batch 检测、日志记录、单卡复现的 Python 实现。

---

## 异常检测触发器

```python
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import json
import time

@dataclass
class StepInfo:
    global_step: int
    micro_step: int
    loss: float
    grad_norm: float
    lr: float
    scale: Optional[float]  # fp16 GradScaler scale
    tokens_per_sec: float
    seq_len_max: int
    seq_len_min: int
    seq_len_mean: float
    sample_ids: list[str]
    shard: str
    file: str
    offset: int
    timestamp: float = field(default_factory=time.time)

class AnomalyDetector:
    """检测 loss spike / grad norm 异常 / nonfinite。"""

    def __init__(self, window=100, loss_k=4.0, grad_threshold=50.0):
        self.loss_history = deque(maxlen=window)
        self.loss_k = loss_k
        self.grad_threshold = grad_threshold
        self.anomalies: list[StepInfo] = []

    def check(self, info: StepInfo) -> list[str]:
        alerts = []

        # 1. nonfinite 检查
        if math.isnan(info.loss) or math.isinf(info.loss):
            alerts.append(f"CRITICAL: nonfinite loss at step {info.global_step}")
        if math.isnan(info.grad_norm) or math.isinf(info.grad_norm):
            alerts.append(f"CRITICAL: nonfinite grad_norm at step {info.global_step}")

        # 2. loss spike 检查
        if len(self.loss_history) >= 10:
            mean = sum(self.loss_history) / len(self.loss_history)
            std = (sum((x - mean) ** 2 for x in self.loss_history) / len(self.loss_history)) ** 0.5
            if info.loss > mean + self.loss_k * max(std, 1e-6):
                alerts.append(
                    f"SPIKE: loss={info.loss:.4f} > {mean:.4f}+{self.loss_k}*{std:.4f} "
                    f"at step {info.global_step}"
                )

        # 3. grad norm 检查
        if info.grad_norm > self.grad_threshold:
            alerts.append(f"HIGH_GRAD: grad_norm={info.grad_norm:.2f} at step {info.global_step}")

        if alerts:
            self.anomalies.append(info)

        self.loss_history.append(info.loss)
        return alerts

    def dump_anomaly(self, info: StepInfo, path: str):
        """保存异常 batch 的完整元信息。"""
        data = {
            "global_step": info.global_step,
            "micro_step": info.micro_step,
            "loss": info.loss,
            "grad_norm": info.grad_norm,
            "lr": info.lr,
            "scale": info.scale,
            "seq_len": {"max": info.seq_len_max, "min": info.seq_len_min, "mean": info.seq_len_mean},
            "sample_ids": info.sample_ids,
            "shard": info.shard,
            "file": info.file,
            "offset": info.offset,
            "timestamp": info.timestamp,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
```

---

## 训练循环集成

```python
def train_with_anomaly_detection(model, optimizer, dataloader, detector):
    for step, batch in enumerate(dataloader):
        # 记录 batch 元信息
        info = StepInfo(
            global_step=step,
            micro_step=0,
            loss=0, grad_norm=0, lr=optimizer.param_groups[0]["lr"],
            scale=scaler.get_scale() if scaler else None,
            tokens_per_sec=0,
            seq_len_max=batch["input_ids"].shape[1],
            seq_len_min=batch["attention_mask"].sum(dim=1).min().item(),
            seq_len_mean=batch["attention_mask"].sum(dim=1).float().mean().item(),
            sample_ids=batch.get("ids", []),
            shard=batch.get("shard", ""),
            file=batch.get("file", ""),
            offset=batch.get("offset", 0),
        )

        # 训练
        t0 = time.time()
        loss, grad_norm = train_step(model, optimizer, batch)
        info.loss = loss.item()
        info.grad_norm = grad_norm
        info.tokens_per_sec = batch["attention_mask"].sum().item() / (time.time() - t0)

        # 异常检测
        alerts = detector.check(info)
        if alerts:
            for a in alerts:
                print(f"🚨 {a}")
            detector.dump_anomaly(info, f"anomaly_step_{step}.json")
```

---

## 单卡复现脚本

```python
import torch

def reproduce_on_single_gpu(model, batch_path, ckpt_path):
    """
    在单卡上复现异常 batch，逐层检查 forward/backward。

    步骤：
    1. 加载 checkpoint
    2. 加载异常 batch
    3. 关闭 dropout
    4. forward 检查 logits/loss
    5. backward 检查各层 grad
    """
    device = torch.device("cuda:0")

    # 1. 加载 checkpoint（不含分布式包装）
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()  # 关闭 dropout

    # 2. 加载异常 batch
    batch = torch.load(batch_path, map_location=device)

    # 3. Forward 检查
    print("=== Forward Pass ===")
    with torch.no_grad():
        outputs = model(**batch)
        logits = outputs.logits
        print(f"logits: min={logits.min():.4f} max={logits.max():.4f} "
              f"has_nan={logits.isnan().any()} has_inf={logits.isinf().any()}")
        print(f"loss: {outputs.loss.item():.4f}")

    # 4. Backward 检查：逐层梯度
    print("\n=== Backward Pass ===")
    model.train()  # 需要开启训练模式做 backward
    for p in model.parameters():
        p.requires_grad_(True)

    outputs = model(**batch)
    outputs.loss.backward()

    print("Per-layer grad norm (top-10 by magnitude):")
    layer_grads = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            gn = param.grad.norm().item()
            has_nan = param.grad.isnan().any().item()
            has_inf = param.grad.isinf().any().item()
            layer_grads.append((name, gn, has_nan, has_inf))

    # 按 grad norm 降序排列
    layer_grads.sort(key=lambda x: x[1], reverse=True)
    for name, gn, has_nan, has_inf in layer_grads[:10]:
        flag = ""
        if has_nan: flag += " ⚠️NaN"
        if has_inf: flag += " ⚠️Inf"
        print(f"  {name}: {gn:.6f}{flag}")

    # 5. 找到第一个 nonfinite 的层
    bad_layers = [(n, gn) for n, gn, nan, inf in layer_grads if nan or inf]
    if bad_layers:
        print(f"\n❌ First nonfinite layer: {bad_layers[0][0]}")
    else:
        print(f"\n✅ All gradients are finite. Global grad norm: "
              f"{sum(gn**2 for _, gn, _, _ in layer_grads)**0.5:.4f}")
```

---

## 替换 Batch 验证

```python
def verify_by_replacing_batch(model, optimizer, bad_batch, good_batch, n_steps=10):
    """
    用正常 batch 替换异常 batch，看训练是否恢复。
    恢复 -> 数据问题；仍异常 -> 超参/实现问题。
    """
    print("--- Training with BAD batch ---")
    for i in range(n_steps):
        loss = train_step(model, optimizer, bad_batch)
        print(f"  step {i}: loss={loss.item():.4f}")

    # 恢复到 checkpoint
    reload_checkpoint(model, optimizer)

    print("--- Training with GOOD batch ---")
    for i in range(n_steps):
        loss = train_step(model, optimizer, good_batch)
        print(f"  step {i}: loss={loss.item():.4f}")
```