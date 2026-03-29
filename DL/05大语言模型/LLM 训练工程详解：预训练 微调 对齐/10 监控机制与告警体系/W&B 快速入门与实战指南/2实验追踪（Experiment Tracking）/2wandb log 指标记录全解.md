# wandb.log 指标记录全解

`wandb.log()` 是 W&B 最频繁调用的 API，理解其 step 机制和 commit 行为是高效记录的关键。

---
## 函数签名
```python
wandb.log(
    data: dict,           # 要记录的键值对
    step: int = None,     # 指定 step（否则自增）
    commit: bool = None,  # 是否提交当前 step
)
```
## Step 机制详解
### 自动递增（默认）
```python
wandb.log({"loss": 0.5})   # step=0
wandb.log({"loss": 0.4})   # step=1
wandb.log({"loss": 0.3})   # step=2
```
### 手动指定 step
```python
for global_step in range(1000):
    if global_step % 10 == 0:
        wandb.log({"train/loss": loss}, step=global_step)
    if global_step % 100 == 0:
        wandb.log({"val/loss": val_loss}, step=global_step)
```
### commit 机制
`commit=True`（默认）：立即提交当前 step 的所有数据并推进 step。
`commit=False`：累积数据到当前 step，不推进。
```python
# 同一个 step 分多次 log
wandb.log({"train/loss": 0.5}, commit=False)  # 累积
wandb.log({"train/lr": 1e-4}, commit=False)   # 累积
wandb.log({"train/grad_norm": 1.2})           # 提交并推进 step
```

⚠️**常见陷阱**：不同频率的指标混用自动 step 会导致 X 轴错位。解决方案：使用 `wandb.define_metric()` 或统一 step。
## define_metric 自定义 X 轴
```python
# 基础用法：让 val 指标以 epoch 为 X 轴
wandb.define_metric("epoch")
wandb.define_metric("val/*", step_metric="epoch")

# 设置聚合方式
wandb.define_metric("val/loss", step_metric="epoch", summary="min")
wandb.define_metric("val/accuracy", step_metric="epoch", summary="max")
wandb.define_metric("train/loss", summary="min")

# summary 选项："min", "max", "mean", "last", "best", "copy", "none"
```
## Summary 控制
Summary 是 Run 的最终指标，显示在 Run 列表中用于排序/对比。
```python
# 自动 Summary：wandb.log 记录的每个 key，Summary 默认取最后一次的值
# 手动覆盖：
wandb.run.summary["best_val_loss"] = best_val_loss
wandb.run.summary["total_tokens"] = total_tokens_processed
wandb.run.summary["training_hours"] = (time.time() - start) / 3600

# 删除
del wandb.run.summary["unwanted_key"]
```

## 高频日志优化
大规模训练每秒可能产生大量日志，需要控制频率：
```python
class SmartLogger:
    """智能日志频率控制。"""

    def __init__(self, log_interval=10, eval_interval=500):
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self._acc = {}  # 累积器
        self._count = 0

    def log_train(self, step: int, metrics: dict):
        # 累积指标
        for k, v in metrics.items():
            if k not in self._acc:
                self._acc[k] = []
            self._acc[k].append(v)
        self._count += 1

        # 按频率提交
        if step % self.log_interval == 0:
            averaged = {
                f"train/{k}": sum(v) / len(v)
                for k, v in self._acc.items()
            }
            wandb.log(averaged, step=step)
            self._acc.clear()
            self._count = 0

    def log_eval(self, step: int, metrics: dict):
        eval_data = {f"val/{k}": v for k, v in metrics.items()}
        wandb.log(eval_data, step=step)

# 使用
logger = SmartLogger(log_interval=10)
for step in range(10000):
    loss = train_step()
    logger.log_train(step, {"loss": loss, "lr": get_lr()})
```

## 完整实战模板
```python
import wandb
import time

def train_with_wandb_logging(model, optimizer, train_loader, val_loader, config):
    run = wandb.init(project="llm-sft", config=config)

    # 定义指标
    wandb.define_metric("epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("train/loss", summary="min")
    wandb.define_metric("val/loss", summary="min")

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0
        t0 = time.time()

        for batch in train_loader:
            loss = train_step(model, optimizer, batch)
            epoch_loss += loss.item()
            global_step += 1

            # 每 N 步记录训练指标
            if global_step % 10 == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/epoch": epoch + global_step / len(train_loader),
                }, step=global_step)

        # 每 epoch 记录验证指标
        val_loss = evaluate(model, val_loader)
        wandb.log({
            "epoch": epoch,
            "val/loss": val_loss,
            "val/epoch_time": time.time() - t0,
        }, step=global_step)

        # 更新 best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.run.summary["best_epoch"] = epoch

    wandb.finish()
```

---

*← 返回：[[1实验追踪（Experiment Tracking）]]*
