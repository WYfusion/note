# Early Termination 与分布式 Sweep

提前终止避免在差参数上浪费算力，分布式 Sweep 让多台机器并行搜索。

---

## Early Termination 策略

### Hyperband

最常用的提前终止算法，基于 Successive Halving 思想：

**原理**：

1. 启动一批 Run
2. 每经过 `min_iter` 步后，淘汰表现最差的 $1/\eta$ 比例
3. 幸存者继续训练到下一个检查点
4. 重复直到只剩最优 Run

```python
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val/loss", "goal": "minimize"},
    "parameters": {
        "lr": {"min": 1e-5, "max": 1e-2, "distribution": "log_uniform_values"},
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 3,    # 至少跑 3 个 epoch/step 再判断
        "eta": 3,         # 每轮淘汰 2/3
        "s": 2,           # bracket 数量（越大探索越多）
        "max_iter": 27,   # 可选，最大迭代数
    },
}
```

**参数详解**：

| **参数** | **含义** | **建议值** |
| --- | --- | --- |
| `min_iter` | 最少运行多少步才开始判断 | 1-3 个 epoch |
| `eta` | 淘汰比例，每轮保留 $1/\eta$ | 2 或 3 |
| `s` | bracket 数量，越多越偏向探索 | 1-3 |
| `max_iter` | 最大运行步数 | 总 epoch 数 |

**Hyperband 淘汰过程示意**（`eta=3`, `min_iter=1`, `max_iter=27`）：

```
Bracket 0:  27 runs × 1 step → 9 runs × 3 steps → 3 runs × 9 steps → 1 run × 27 steps
Bracket 1:  9 runs × 3 steps → 3 runs × 9 steps → 1 run × 27 steps
Bracket 2:  3 runs × 9 steps → 1 run × 27 steps
```

### Median Stopping

更简单的策略：如果当前 Run 的指标低于所有 Run 同一 step 的中位数，则终止。

```python
"early_terminate": {
    "type": "median",
    "min_iter": 5,     # 至少跑 5 步
    "grace_period": 10, # 宽限期（前 10 步不淘汰）
}
```

### 策略对比

| **策略** | **激进程度** | **适用场景** | **节省算力** |
| --- | --- | --- | --- |
| Hyperband | 高 | 大规模搜索、GPU 紧张 | 50-80% |
| Median | 中 | 稳健搜索、指标波动大 | 30-50% |
| 无 | 无 | 运行次数少、每次很快 | 0% |

## 分布式 Sweep

### 多机并行

一个 Sweep 可以在多台机器上并行执行：

```bash
# 机器 A
wandb agent my-team/llm-sft/sweep_id

# 机器 B（同一个 sweep_id）
wandb agent my-team/llm-sft/sweep_id

# 机器 C
wandb agent my-team/llm-sft/sweep_id
```

**每台机器独立从 Sweep Controller 获取下一组参数**，无需额外协调。

### Python 多 Agent

```python
import wandb
import multiprocessing

def run_agent(sweep_id, count):
    wandb.agent(sweep_id, function=train, count=count)

sweep_id = wandb.sweep(sweep_config, project="llm-sft")

# 本机启动多个 Agent（每个 Agent 用一张 GPU）
processes = []
for gpu_id in range(4):  # 4 张 GPU
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    p = multiprocessing.Process(
        target=run_agent,
        args=(sweep_id, 5),  # 每个 Agent 跑 5 组
    )
    p.start()
    processes.append(p)

for p in processes:
    p.join()
```

### SLURM 集群集成

```bash
#!/bin/bash
#SBATCH --job-name=wandb-sweep
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-19        # 启动 20 个任务
#SBATCH --time=4:00:00

module load cuda/12.1
source activate ml

# 每个 SLURM 任务运行一个 Agent
wandb agent my-team/llm-sft/${SWEEP_ID} --count 1
```

提交：

```bash
# 先创建 Sweep
SWEEP_ID=$(wandb sweep sweep.yaml 2>&1 | grep -oP 'wandb agent \K.*')

# 再提交 SLURM 任务
export SWEEP_ID
sbatch sweep_job.sh
```

### 分布式 Sweep 最佳实践

<aside>
💡

1. **设置 `count`**：每个 Agent 限制运行次数，避免一个 Agent 占满所有实验
2. **配合 Early Termination**：分布式场景下提前终止效果更明显
3. **监控并行数**：W&B UI 可查看当前活跃 Agent 数量
4. **避免 GPU 冲突**：确保每个 Agent 使用独立 GPU
</aside>

---

*← 返回：[[1超参数调优（Sweeps）]]*
