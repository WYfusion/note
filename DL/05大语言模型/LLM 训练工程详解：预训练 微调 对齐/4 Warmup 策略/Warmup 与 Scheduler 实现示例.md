# Warmup 与 Scheduler 实现示例

本页面提供 warmup 策略的具体实现，以及与 scheduler 的集成方式。

---

## 标准 Linear Warmup

```python
class LinearWarmup:
    """独立的 warmup wrapper，可包裹任意 scheduler。"""

    def __init__(self, optimizer, warmup_steps, after_scheduler=None):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.after_scheduler = after_scheduler
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            scale = self.current_step / self.warmup_steps
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg["lr"] = base_lr * scale
        elif self.after_scheduler:
            self.after_scheduler.step()

    def state_dict(self):
        return {
            "current_step": self.current_step,
            "after_scheduler": (
                self.after_scheduler.state_dict()
                if self.after_scheduler else None
            ),
        }

    def load_state_dict(self, state):
        self.current_step = state["current_step"]
        if self.after_scheduler and state["after_scheduler"]:
            self.after_scheduler.load_state_dict(state["after_scheduler"])
```

---

## Warmup 长度自动调整策略

```python
def compute_warmup_steps(
    total_steps: int,
    batch_size: int,
    is_resume: bool = False,
    data_quality: str = "clean",  # "clean" | "noisy"
    base_ratio: float = 0.01,
) -> int:
    """
    根据训练条件自动计算 warmup 步数。

    经验规则：
    - base: total_steps * base_ratio
    - batch 大则加长
    - resume 则缩短
    - 数据脏则加长
    """
    warmup = int(total_steps * base_ratio)

    # batch 越大，warmup 按 sqrt 比例加长
    batch_factor = (batch_size / 2048) ** 0.5
    warmup = int(warmup * batch_factor)

    # resume 且分布接近：缩短到 1/4
    if is_resume:
        warmup = max(100, warmup // 4)

    # 数据脏：加长 50%
    if data_quality == "noisy":
        warmup = int(warmup * 1.5)

    # 下限保护
    warmup = max(100, min(warmup, total_steps // 5))
    return warmup

# 示例
print(compute_warmup_steps(200000, 4096))          # ~2828
print(compute_warmup_steps(200000, 4096, is_resume=True))  # ~707
print(compute_warmup_steps(200000, 4096, data_quality="noisy"))  # ~4243
```

---

## Warmup 健康检查

在 warmup 阶段额外监控，及时发现问题：

```python
class WarmupHealthChecker:
    """在 warmup 阶段密集检查训练稳定性。"""

    def __init__(self, warmup_steps, loss_threshold=20.0, grad_threshold=100.0):
        self.warmup_steps = warmup_steps
        self.loss_threshold = loss_threshold
        self.grad_threshold = grad_threshold
        self.losses = []
        self.grad_norms = []

    def check(self, step, loss, grad_norm):
        if step > self.warmup_steps:
            return  # warmup 后不检查

        self.losses.append(loss)
        self.grad_norms.append(grad_norm)

        # 检查 loss 是否爆炸
        if loss > self.loss_threshold:
            print(f"⚠️ [Warmup] Step {step}: loss={loss:.2f} > threshold={self.loss_threshold}")
            print("  建议：延长 warmup 或降低 LR")

        # 检查 grad norm 是否异常
        if grad_norm > self.grad_threshold:
            print(f"⚠️ [Warmup] Step {step}: grad_norm={grad_norm:.2f} > threshold")
            print("  建议：收紧 grad clipping 或降低 LR")

        # 检查 loss 是否在下降
        if len(self.losses) > 50:
            recent = self.losses[-20:]
            older = self.losses[-50:-30]
            if sum(recent) / len(recent) > sum(older) / len(older) * 1.2:
                print(f"⚠️ [Warmup] Step {step}: loss 未在下降，可能 warmup 不足")
```

---

## 完整训练循环集成

```python
def training_loop(model, optimizer, train_loader, config):
    # 计算 warmup
    warmup_steps = compute_warmup_steps(
        config.total_steps, config.batch_size,
        is_resume=config.resume, data_quality=config.data_quality,
    )
    print(f"Warmup steps: {warmup_steps}")

    # 创建 scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, config.total_steps, min_lr_ratio=0.1
    )

    # 健康检查
    health = WarmupHealthChecker(warmup_steps)

    for step, batch in enumerate(train_loader):
        loss, grad_norm = train_step(model, optimizer, batch)
        scheduler.step()

        # warmup 阶段密集检查
        health.check(step, loss.item(), grad_norm)

        # warmup 阶段密集保存 checkpoint
        if step < warmup_steps and step % 100 == 0:
            save_checkpoint(model, optimizer, scheduler, step)
        elif step % config.save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, step)
```