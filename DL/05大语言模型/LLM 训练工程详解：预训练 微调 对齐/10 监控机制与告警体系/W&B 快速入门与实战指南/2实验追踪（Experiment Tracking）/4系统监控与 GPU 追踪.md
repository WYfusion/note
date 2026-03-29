# 系统监控与 GPU 追踪
W&B 自动采集系统级指标，无需额外代码。掌握配置方法可精准诊断训练瓶颈。

---

## 默认自动采集指标
W&B 每 **10 秒** 自动采集一次，记录到 `system/` 前缀面板：

| **指标** | **含义** | **正常范围** |
| --- | --- | --- |
| `system.gpu.0.gpu` | GPU 计算利用率 % | >90% 为佳 |
| `system.gpu.0.memory` | GPU 显存利用率 % | 60-95% |
| `system.gpu.0.memoryAllocated` | 实际分配显存 GB | — |
| `system.gpu.0.temp` | GPU 温度 ℃ | <85℃ |
| `system.gpu.0.powerWatts` | GPU 功耗 W | — |
| `system.cpu` | CPU 利用率 % | 视 DataLoader workers |
| `system.memory` | 系统内存利用率 % | <90% |
| `system.disk.in` | 磁盘读取 MB/s | — |
| `system.network.sent` | 网络发送 MB/s | — |

## 配置采集频率
```python
import wandb

# 调整采集间隔（秒）
wandb.init(
    settings=wandb.Settings(
        _stats_sample_rate_seconds=5,  # 每 5 秒采集
        _stats_samples_to_average=3,   # 每 3 个样本取平均
    )
)
```
或通过环境变量：
```bash
export WANDB__STATS_SAMPLE_RATE_SECONDS=5
```
## 自定义系统指标
当默认指标不够时，手动采集并上报：
```python
import subprocess
import re

def get_nvidia_smi_metrics():
    """通过 nvidia-smi 获取详细 GPU 指标。"""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    metrics = {}
    for i, line in enumerate(result.stdout.strip().split("\n")):
        vals = [float(x.strip()) for x in line.split(",")]
        metrics[f"gpu/{i}/util"] = vals[0]
        metrics[f"gpu/{i}/mem_used_gb"] = vals[1] / 1024
        metrics[f"gpu/{i}/mem_total_gb"] = vals[2] / 1024
        metrics[f"gpu/{i}/mem_pct"] = vals[1] / vals[2] * 100
        metrics[f"gpu/{i}/temp"] = vals[3]
        metrics[f"gpu/{i}/power_w"] = vals[4]
    return metrics

# PyTorch 原生方法
import torch

def get_torch_gpu_metrics(device=0):
    """通过 PyTorch 获取 GPU 内存。"""
    return {
        "gpu/allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
        "gpu/reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
        "gpu/max_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "gpu/fragmentation": (
            1 - torch.cuda.memory_allocated(device) /
            max(torch.cuda.memory_reserved(device), 1)
        ),
    }

# 在训练循环中周期性上报
if step % 50 == 0:
    system_metrics = get_torch_gpu_metrics()
    wandb.log(system_metrics, step=step)
```

## 瓶颈诊断指南

| **症状** | **W&B 指标表现** | **诊断** | **解决方案** |
| --- | --- | --- | --- |
| GPU 利用率低 | `gpu.util` < 60% | 数据加载瓶颈 | 增加 `num_workers`、启用 `pin_memory` |
| GPU 利用率波动大 | `gpu.util` 锯齿状 | CPU/GPU 交替等待 | 异步数据预取、Prefetcher |
| 显存接近上限 | `gpu.memory` > 95% | OOM 风险 | 减小 batch_size、启用 gradient checkpointing |
| 网络 I/O 高峰 | `network.sent` 突增 | 分布式通信瓶颈 | overlap communication、gradient compression |
| 磁盘 I/O 持续高 | `disk.in` > 500MB/s | 数据读取慢 | 缓存到 SSD/RAM、WebDataset 格式 |

## 与 Step Time 联合分析
结合[[W&B + Prometheus 监控搭建实战#^87ddcf|Step Time 分解器]]，上报到 W&B 形成完整画面：
```python
# 上报 step time 分解
wandb.log({
    "time/total": step_time,
    "time/data": data_time,
    "time/forward": fwd_time,
    "time/backward": bwd_time,
    "time/optimizer": optim_time,
    "time/communication": comm_time,
    # 各阶段占比
    "time_pct/data": data_time / step_time * 100,
    "time_pct/forward": fwd_time / step_time * 100,
    "time_pct/backward": bwd_time / step_time * 100,
    "time_pct/optimizer": optim_time / step_time * 100,
}, step=global_step)
```

在 W&B Dashboard 中将 `time_pct/*` 面板设为堆叠面积图，即可直观看到训练时间分布变化。

---

*← 返回：[[1实验追踪（Experiment Tracking）]]*
