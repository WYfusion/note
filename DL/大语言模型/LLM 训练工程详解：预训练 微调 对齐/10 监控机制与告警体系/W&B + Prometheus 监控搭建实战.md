# W&B + Prometheus 监控搭建实战

本页面提供三层监控体系的完整 Python 实现：W&B 训练指标 + Prometheus 系统指标 + 自动告警。

---
## 第一层：W&B 训练指标

```python
import wandb
import time
from dataclasses import dataclass

@dataclass
class TrainMetrics:
    step: int
    train_loss: float
    val_loss: float = None
    lr: float = 0.0
    grad_norm: float = 0.0
    clip_ratio: float = 0.0
    overflow_count: int = 0
    tokens_per_sec: float = 0.0
    samples_per_sec: float = 0.0
    mfu: float = 0.0
    step_time: float = 0.0
    data_time: float = 0.0
    fwd_time: float = 0.0
    bwd_time: float = 0.0
    optim_time: float = 0.0
    comm_time: float = 0.0
    ckpt_time: float = 0.0
    gpu_util: float = 0.0
    gpu_mem_gb: float = 0.0

class WandBLogger:
    """封装 W&B 日志，提供统一接口。"""

    def __init__(self, project: str, run_name: str, config: dict):
        self.run = wandb.init(
            project=project,
            name=run_name,
            config=config,
            resume="allow",
        )
        # 定义告警
        wandb.alert(
            title="Training Started",
            text=f"Run {run_name} started",
            level=wandb.AlertLevel.INFO,
        )

    def log(self, metrics: TrainMetrics):
        data = {
            "train/loss": metrics.train_loss,
            "train/lr": metrics.lr,
            "train/grad_norm": metrics.grad_norm,
            "train/clip_ratio": metrics.clip_ratio,
            "train/overflow_count": metrics.overflow_count,
            "throughput/tokens_per_sec": metrics.tokens_per_sec,
            "throughput/samples_per_sec": metrics.samples_per_sec,
            "throughput/mfu": metrics.mfu,
            "time/step_time": metrics.step_time,
            "time/data": metrics.data_time,
            "time/forward": metrics.fwd_time,
            "time/backward": metrics.bwd_time,
            "time/optimizer": metrics.optim_time,
            "time/communication": metrics.comm_time,
            "system/gpu_util": metrics.gpu_util,
            "system/gpu_mem_gb": metrics.gpu_mem_gb,
        }
        if metrics.val_loss is not None:
            data["val/loss"] = metrics.val_loss
        if metrics.ckpt_time > 0:
            data["time/checkpoint"] = metrics.ckpt_time

        wandb.log(data, step=metrics.step)

    def alert(self, title: str, text: str, level="WARN"):
        wandb.alert(
            title=title, text=text,
            level=getattr(wandb.AlertLevel, level),
        )
```

---

## 第二层：Prometheus + Grafana 系统指标

```python
try:
    from prometheus_client import Gauge, Counter, start_http_server
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

class PrometheusMetrics:
    """暴露 Prometheus 指标，供 Grafana 拉取。"""

    def __init__(self, port: int = 8000):
        if not HAS_PROMETHEUS:
            print("⚠️ prometheus_client not installed, skipping")
            return

        # 训练指标
        self.train_loss = Gauge("train_loss", "Training loss")
        self.grad_norm = Gauge("grad_norm", "Gradient norm")
        self.lr = Gauge("learning_rate", "Current learning rate")
        self.tokens_per_sec = Gauge("tokens_per_sec", "Tokens per second")
        self.step_time = Gauge("step_time_seconds", "Time per step")

        # 系统指标
        self.gpu_util = Gauge("gpu_utilization", "GPU utilization %", ["rank"])
        self.gpu_mem = Gauge("gpu_memory_gb", "GPU memory used GB", ["rank"])
        self.data_wait = Gauge("data_wait_ratio", "Data loading wait ratio")

        # 计数器
        self.overflow_total = Counter("overflow_total", "Total overflow events")
        self.spike_total = Counter("spike_total", "Total loss spike events")

        start_http_server(port)
        print(f"✅ Prometheus metrics server on port {port}")

    def update(self, metrics: TrainMetrics, rank: int = 0):
        if not HAS_PROMETHEUS:
            return
        self.train_loss.set(metrics.train_loss)
        self.grad_norm.set(metrics.grad_norm)
        self.lr.set(metrics.lr)
        self.tokens_per_sec.set(metrics.tokens_per_sec)
        self.step_time.set(metrics.step_time)
        self.gpu_util.labels(rank=str(rank)).set(metrics.gpu_util)
        self.gpu_mem.labels(rank=str(rank)).set(metrics.gpu_mem_gb)
        if metrics.step_time > 0:
            self.data_wait.set(metrics.data_time / metrics.step_time)
```

---

## 第三层：自动告警系统

```python
from collections import deque

class AlertManager:
    """基于规则的自动告警系统。"""

    def __init__(self, wandb_logger=None):
        self.logger = wandb_logger
        self.loss_history = deque(maxlen=500)
        self.tps_history = deque(maxlen=100)
        self.val_losses = []

    def check(self, metrics: TrainMetrics) -> list[str]:
        alerts = []

        # === Loss Spike ===
        self.loss_history.append(metrics.train_loss)
        if len(self.loss_history) > 20:
            mean = sum(self.loss_history) / len(self.loss_history)
            std = (sum((x-mean)**2 for x in self.loss_history)/len(self.loss_history))**0.5
            if metrics.train_loss > mean + 4 * max(std, 1e-6):
                alerts.append(f"🔴 Loss spike: {metrics.train_loss:.4f} >> mean {mean:.4f}")

        # === Grad Norm 突变 ===
        if metrics.grad_norm > 50:
            alerts.append(f"🔴 High grad_norm: {metrics.grad_norm:.2f}")

        # === Tokens/sec 下降 ===
        self.tps_history.append(metrics.tokens_per_sec)
        if len(self.tps_history) > 20:
            recent = list(self.tps_history)[-10:]
            baseline = list(self.tps_history)[:10]
            if sum(recent)/len(recent) < sum(baseline)/len(baseline) * 0.8:
                alerts.append(
                    f"🟡 Tokens/sec dropped >20%: "
                    f"{sum(baseline)/len(baseline):.0f} -> {sum(recent)/len(recent):.0f}"
                )

        # === Data Loading 瓶颈 ===
        if metrics.step_time > 0 and metrics.data_time / metrics.step_time > 0.2:
            alerts.append(
                f"🟡 Data loading bottleneck: "
                f"{metrics.data_time/metrics.step_time:.0%} of step time"
            )

        # === GPU Util 低 ===
        if metrics.gpu_util < 60 and metrics.gpu_util > 0:
            alerts.append(f"🟡 Low GPU util: {metrics.gpu_util:.0f}%")

        # === Val Loss 连续恶化 ===
        if metrics.val_loss is not None:
            self.val_losses.append(metrics.val_loss)
            if len(self.val_losses) >= 3:
                if all(self.val_losses[i] > self.val_losses[i-1]
                       for i in range(-2, 0)):
                    alerts.append(
                        f"🔴 Val loss deteriorating: "
                        f"{[f'{v:.4f}' for v in self.val_losses[-3:]]}"
                    )

        # 发送告警
        if alerts and self.logger:
            self.logger.alert(
                title=f"Training Alert @ step {metrics.step}",
                text="\n".join(alerts),
            )

        return alerts
```

---

## Step Time 分解器

```python
class StepTimer:
    """分解单步训练时间到各阶段。"""

    def __init__(self):
        self._timers = {}
        self._results = {}

    def start(self, name: str):
        torch.cuda.synchronize()
        self._timers[name] = time.time()

    def stop(self, name: str) -> float:
        torch.cuda.synchronize()
        elapsed = time.time() - self._timers[name]
        self._results[name] = elapsed
        return elapsed

    def get_results(self) -> dict:
        return dict(self._results)

    def reset(self):
        self._timers.clear()
        self._results.clear()

# 使用示例
timer = StepTimer()

timer.start("data")
batch = next(dataloader_iter)
timer.stop("data")

timer.start("forward")
loss = model(**batch).loss
timer.stop("forward")

timer.start("backward")
loss.backward()
timer.stop("backward")

timer.start("optimizer")
optimizer.step()
timer.stop("optimizer")

# 结果: {"data": 0.01, "forward": 0.15, "backward": 0.30, "optimizer": 0.05}
print(timer.get_results())
```

---

## 完整集成示例

```python
def train_with_full_monitoring(model, optimizer, dataloader, config):
    wb = WandBLogger("llm-training", config.run_name, vars(config))
    prom = PrometheusMetrics(port=8000)
    alert_mgr = AlertManager(wandb_logger=wb)
    timer = StepTimer()

    for step, batch in enumerate(dataloader):
        timer.start("data")
        # ... 数据准备 ...
        timer.stop("data")

        timer.start("forward")
        loss = model(**batch).loss
        timer.stop("forward")

        timer.start("backward")
        loss.backward()
        timer.stop("backward")

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        timer.start("optimizer")
        optimizer.step()
        optimizer.zero_grad()
        timer.stop("optimizer")

        times = timer.get_results()
        metrics = TrainMetrics(
            step=step,
            train_loss=loss.item(),
            lr=optimizer.param_groups[0]["lr"],
            grad_norm=grad_norm.item(),
            tokens_per_sec=batch["attention_mask"].sum().item() / sum(times.values()),
            step_time=sum(times.values()),
            data_time=times.get("data", 0),
            fwd_time=times.get("forward", 0),
            bwd_time=times.get("backward", 0),
            optim_time=times.get("optimizer", 0),
        )

        # 日志 + 监控 + 告警
        wb.log(metrics)
        prom.update(metrics)
        alerts = alert_mgr.check(metrics)
        if alerts:
            for a in alerts:
                print(f"🚨 {a}")

        timer.reset()
```