# 梯度范数监控 Python 实现

本页面提供梯度范数监控、裁剪、异常检测的完整 Python 实现。

---

## 梯度裁剪与监控封装

```python
import torch
import math
from collections import deque
from typing import Optional

class GradNormMonitor:
    """梯度范数监控器：裁剪 + 趋势检测 + 异常 dump。"""

    def __init__(
        self,
        max_norm: float = 1.0,
        history_window: int = 200,
        spike_k: float = 3.0,
        vanish_threshold: float = 1e-7,
    ):
        self.max_norm = max_norm
        self.spike_k = spike_k
        self.vanish_threshold = vanish_threshold
        self.history = deque(maxlen=history_window)
        self.clip_count = 0
        self.total_count = 0

    def clip_and_monitor(
        self,
        model: torch.nn.Module,
        error_if_nonfinite: bool = True,
    ) -> dict:
        """
        裁剪梯度并返回监控信息。

        Returns:
            {
                "grad_norm": float,       # 裁剪前的原始 grad norm
                "clipped": bool,           # 是否被裁剪
                "is_nonfinite": bool,       # 是否包含 NaN/Inf
                "clip_ratio": float,        # 历史裁剪比例
                "alerts": list[str],        # 异常告警
            }
        """
        self.total_count += 1
        alerts = []

        # 计算原始 grad norm
        params = [p for p in model.parameters() if p.grad is not None]
        if not params:
            return {"grad_norm": 0.0, "clipped": False, "is_nonfinite": False,
                    "clip_ratio": 0.0, "alerts": ["No gradients found"]}

        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), 2) for p in params]), 2
        ).item()

        # nonfinite 检查
        is_nonfinite = math.isnan(total_norm) or math.isinf(total_norm)
        if is_nonfinite:
            alerts.append(f"CRITICAL: nonfinite grad_norm={total_norm}")
            if error_if_nonfinite:
                # 不更新参数，跳过这个 step
                model.zero_grad()
                return {"grad_norm": total_norm, "clipped": False,
                        "is_nonfinite": True, "clip_ratio": self.clip_ratio,
                        "alerts": alerts}

        # 裁剪
        clipped = total_norm > self.max_norm
        if clipped:
            self.clip_count += 1
            torch.nn.utils.clip_grad_norm_(params, self.max_norm)

        # 趋势检测
        if len(self.history) >= 20:
            mean = sum(self.history) / len(self.history)
            std = (sum((x - mean) ** 2 for x in self.history) / len(self.history)) ** 0.5

            # spike 检测
            if total_norm > mean + self.spike_k * max(std, 1e-6):
                alerts.append(f"SPIKE: grad_norm={total_norm:.4f} >> mean={mean:.4f}")

            # 梯度消失检测
            if total_norm < self.vanish_threshold:
                alerts.append(f"VANISH: grad_norm={total_norm:.2e} < threshold")

            # 持续爬升检测（近期均值 > 远期均值 * 1.5）
            recent = list(self.history)[-20:]
            older = list(self.history)[-50:-30] if len(self.history) >= 50 else []
            if older:
                recent_mean = sum(recent) / len(recent)
                older_mean = sum(older) / len(older)
                if recent_mean > older_mean * 1.5:
                    alerts.append(f"CLIMBING: grad_norm trending up {older_mean:.4f}->{recent_mean:.4f}")

        self.history.append(total_norm)

        return {
            "grad_norm": total_norm,
            "clipped": clipped,
            "is_nonfinite": is_nonfinite,
            "clip_ratio": self.clip_ratio,
            "alerts": alerts,
        }

    @property
    def clip_ratio(self) -> float:
        return self.clip_count / max(1, self.total_count)
```

---

## Per-Layer Grad Norm Dump

异常时自动 dump 各层梯度范数，快速定位问题层：

```python
import json
from pathlib import Path

def dump_per_layer_grad_norm(
    model: torch.nn.Module,
    step: int,
    save_dir: str = "grad_dumps",
    top_k: int = 20,
) -> list[tuple[str, float, bool]]:
    """
    Dump 各层 grad norm，返回 top-k 并保存到文件。

    Returns: [(layer_name, grad_norm, has_nonfinite), ...]
    """
    layer_grads = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        gn = param.grad.norm(2).item()
        has_nonfinite = (
            param.grad.isnan().any().item() or
            param.grad.isinf().any().item()
        )
        layer_grads.append((name, gn, has_nonfinite))

    # 按 grad norm 降序
    layer_grads.sort(key=lambda x: x[1] if not math.isnan(x[1]) else float('inf'), reverse=True)

    # 打印 top-k
    print(f"\n=== Grad Norm Dump @ step {step} (top-{top_k}) ===")
    for name, gn, bad in layer_grads[:top_k]:
        flag = " ⚠️ NaN/Inf" if bad else ""
        print(f"  {gn:12.6f}  {name}{flag}")

    # 保存到文件
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    data = [
        {"layer": name, "grad_norm": gn, "nonfinite": bad}
        for name, gn, bad in layer_grads
    ]
    with open(save_path / f"grad_dump_step_{step}.json", "w") as f:
        json.dump(data, f, indent=2)

    return layer_grads[:top_k]
```

---

## Update Norm / Param Norm 监控

衡量每步实际参数更新量与参数本身的比值：

```python
class UpdateNormTracker:
    """跟踪 update_norm / param_norm 比值，评估 LR 是否合理。"""

    def __init__(self):
        self.prev_params = {}  # name -> tensor

    def snapshot(self, model):
        """训练 step 前调用，保存当前参数。"""
        self.prev_params = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def compute(self, model) -> dict:
        """
        训练 step 后调用，计算 update_norm / param_norm。

        Returns:
            {
                "global_update_norm": float,
                "global_param_norm": float,
                "update_to_param_ratio": float,
                "per_layer": [{"name": str, "update_norm": float,
                               "param_norm": float, "ratio": float}]
            }
        """
        if not self.prev_params:
            return {}

        update_norms = []
        param_norms = []
        per_layer = []

        for name, param in model.named_parameters():
            if name not in self.prev_params:
                continue
            update = param.data - self.prev_params[name]
            un = update.norm(2).item()
            pn = param.data.norm(2).item()
            update_norms.append(un ** 2)
            param_norms.append(pn ** 2)
            per_layer.append({
                "name": name,
                "update_norm": un,
                "param_norm": pn,
                "ratio": un / max(pn, 1e-10),
            })

        global_un = sum(update_norms) ** 0.5
        global_pn = sum(param_norms) ** 0.5

        return {
            "global_update_norm": global_un,
            "global_param_norm": global_pn,
            "update_to_param_ratio": global_un / max(global_pn, 1e-10),
            "per_layer": sorted(per_layer, key=lambda x: x["ratio"], reverse=True),
        }
```

---

## 完整训练循环集成

```python
def train_with_grad_monitoring(model, optimizer, dataloader, config):
    monitor = GradNormMonitor(max_norm=config.max_grad_norm)
    tracker = UpdateNormTracker()

    for step, batch in enumerate(dataloader):
        # 快照（可选，每 N 步做一次）
        if step % 100 == 0:
            tracker.snapshot(model)

        # Forward + Backward
        loss = model(**batch).loss
        loss.backward()

        # 梯度监控 + 裁剪
        result = monitor.clip_and_monitor(model)

        # 处理告警
        if result["alerts"]:
            for alert in result["alerts"]:
                print(f"🚨 Step {step}: {alert}")
            # 异常时 dump 各层 grad norm
            dump_per_layer_grad_norm(model, step)

        # 跳过 nonfinite step
        if result["is_nonfinite"]:
            print(f"⚠️ Skipping step {step} due to nonfinite gradients")
            optimizer.zero_grad()
            continue

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Update norm 监控（可选）
        if step % 100 == 0:
            update_info = tracker.compute(model)
            if update_info:
                ratio = update_info["update_to_param_ratio"]
                print(f"Step {step}: update/param ratio = {ratio:.6f}")
                # 比值过大可能 LR 太高
                if ratio > 0.01:
                    print(f"  ⚠️ High update ratio, consider lowering LR")

        # 日志
        if step % 10 == 0:
            print(
                f"Step {step}: loss={loss.item():.4f} "
                f"grad_norm={result['grad_norm']:.4f} "
                f"clipped={result['clipped']} "
                f"clip_ratio={result['clip_ratio']:.2%}"
            )
```