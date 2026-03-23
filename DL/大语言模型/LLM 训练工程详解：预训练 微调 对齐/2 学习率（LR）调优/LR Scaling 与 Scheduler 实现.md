# LR Scaling 与 Scheduler 实现

本页面详细介绍 LR scaling 规则、常用 scheduler 的数学公式与 Python 实现。

---

## LR Scaling 规则

当 batch size 从 $B_0$ 变为 $B_1$ 时，如何调整 LR？

### 线性 Scaling

$\eta_1 = \eta_0 \times \frac{B_1}{B_0}$

适用场景：SGD，小幅 batch 变化。过于激进，大 batch 时易不稳。

### 平方根 Scaling

$\eta_1 = \eta_0 \times \sqrt{\frac{B_1}{B_0}}$

适用场景：Adam 系优化器，大 batch 训练。更保守，工程中更常用。

```python
def scale_lr(base_lr: float, base_bs: int, new_bs: int, rule: str = "sqrt") -> float:
    """根据 batch size 变化缩放 LR。"""
    ratio = new_bs / base_bs
    if rule == "linear":
        return base_lr * ratio
    elif rule == "sqrt":
        return base_lr * (ratio ** 0.5)
    else:
        raise ValueError(f"Unknown rule: {rule}")

# 示例：base_lr=3e-4, batch 从 2048 扩到 4096
print(scale_lr(3e-4, 2048, 4096, "sqrt"))  # 4.24e-4
```

---

## Cosine Scheduler

最常用的 LR 衰减策略，公式如下：

$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t - T_w}{T - T_w}\pi\right)\right)$

其中 $T_w$ 为 warmup 步数，$T$ 为总步数。

```python
import math
import torch
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,  # min_lr = peak_lr * min_lr_ratio
):
    """Cosine scheduler with linear warmup."""
    def lr_lambda(current_step: int) -> float:
        # Warmup 阶段：线性从 0 升到 1
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        # Cosine decay 阶段
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay)

    return LambdaLR(optimizer, lr_lambda)

# 使用示例
model = ...  # your model
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps=2000, total_steps=200000)
```

---

## WSD (Warmup-Stable-Decay) Scheduler

MiniCPM / DeepSeek 等新模型采用的三阶段 scheduler，在 pretrain 和 continual pretrain 间切换更灵活：

```python
def get_wsd_schedule(
    optimizer,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    min_lr_ratio: float = 0.0,
):
    """Warmup -> Stable -> Decay 三阶段 scheduler。"""
    total = warmup_steps + stable_steps + decay_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        elif step < warmup_steps + stable_steps:
            return 1.0
        else:
            decay_progress = (step - warmup_steps - stable_steps) / max(1, decay_steps)
            cosine = 0.5 * (1 + math.cos(math.pi * decay_progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)
```

---

## Linear Decay Scheduler

```python
def get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.0):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))
    return LambdaLR(optimizer, lr_lambda)
```

---

## Scheduler 对比可视化

```python
import matplotlib.pyplot as plt

def plot_schedulers(total_steps=200000, warmup=2000, peak_lr=3e-4):
    """可视化不同 scheduler 的 LR 曲线。"""
    dummy_model = torch.nn.Linear(10, 10)
    steps = list(range(total_steps))

    schedulers = {
        "cosine": get_cosine_schedule_with_warmup,
        "linear": get_linear_schedule_with_warmup,
    }

    fig, ax = plt.subplots(figsize=(12, 5))
    for name, fn in schedulers.items():
        opt = torch.optim.AdamW(dummy_model.parameters(), lr=peak_lr)
        sch = fn(opt, warmup, total_steps, min_lr_ratio=0.1)
        lrs = []
        for _ in steps:
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sch.step()
        ax.plot(steps, lrs, label=name)

    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.legend()
    ax.set_title("LR Scheduler Comparison")
    plt.tight_layout()
    plt.savefig("lr_schedulers.png", dpi=150)
    plt.show()
```

---

## LR Finder（小规模快速定位最优 LR）

经典做法：从极小 LR 指数增长，观察 loss 下降最快的区间。

```python
def lr_range_test(
    model, train_loader, optimizer,
    lr_start=1e-7, lr_end=1e-1, num_steps=200,
):
    """LR Range Test：快速找到最优 LR 区间。"""
    lr_mult = (lr_end / lr_start) ** (1 / num_steps)
    lr = lr_start
    losses, lrs = [], []
    best_loss = float("inf")

    model.train()
    for i, batch in enumerate(train_loader):
        if i >= num_steps:
            break

        # 设置 LR
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        loss = train_step(model, optimizer, batch)
        losses.append(loss.item())
        lrs.append(lr)

        # 如果 loss 爆炸则提前停止
        if loss.item() > best_loss * 4:
            break
        best_loss = min(best_loss, loss.item())

        lr *= lr_mult

    # 最优 LR ≈ loss 下降最陡处 / 10
    return lrs, losses
```