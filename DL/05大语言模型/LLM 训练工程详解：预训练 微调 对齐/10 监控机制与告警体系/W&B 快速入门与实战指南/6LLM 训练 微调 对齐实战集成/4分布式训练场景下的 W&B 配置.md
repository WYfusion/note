# 分布式训练场景下的 W&B 配置

多 GPU / 多节点训练时 W&B 的正确配置方式，覆盖 DeepSpeed、FSDP、Megatron 等主流方案。

---

## 核心原则

<aside>
⚠️

**分布式训练中只有 rank 0 应该初始化 W&B**。HuggingFace Trainer / DeepSpeed / FSDP 默认遵循此行为，无需额外配置。

</aside>

## Trainer + DeepSpeed

### 配置文件

```json
// ds_config.json
{
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "none"},
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8
    },
    "gradient_accumulation_steps": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}
```

### 启动命令

```bash
# W&B 环境变量在所有节点设置
export WANDB_PROJECT="llm-pretrain"
export WANDB_API_KEY="your-key"

# DeepSpeed 启动（Trainer 自动处理 W&B）
deepspeed --num_gpus=8 train.py \
    --deepspeed ds_config.json \
    --report_to wandb \
    --run_name "llama3-8b-deepspeed-z2"
```

Trainer + DeepSpeed 会自动确保：

- 只有 rank 0 初始化 W&B Run
- 所有 rank 的 loss 已聚合后上报
- 系统指标从所有 GPU 采集

## Trainer + FSDP

```bash
# FSDP 配置
export FSDP_SHARDING_STRATEGY="FULL_SHARD"
export FSDP_STATE_DICT_TYPE="FULL_STATE_DICT"

# accelerate 启动
accelerate launch --config_file fsdp_config.yaml train.py \
    --report_to wandb \
    --run_name "llama3-8b-fsdp"
```

```yaml
# fsdp_config.yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: FULL_STATE_DICT
machine_rank: 0
num_machines: 1
num_processes: 8
```

## 原生 PyTorch DDP + W&B

不使用 Trainer 时需要手动控制：

```python
import os
import torch
import torch.distributed as dist
import wandb

def setup_wandb(rank, world_size, config):
    """分布式环境下初始化 W&B。"""
    if rank == 0:
        # 只有 rank 0 初始化
        wandb.init(
            project="llm-pretrain",
            name=config["run_name"],
            config=config,
            group=config.get("group", "ddp-training"),
        )
    else:
        # 其他 rank 禁用
        os.environ["WANDB_MODE"] = "disabled"
        wandb.init(mode="disabled")

def log_distributed(metrics, step, rank):
    """只在 rank 0 记录日志。"""
    if rank == 0:
        wandb.log(metrics, step=step)

def log_all_reduce(metrics_dict, step, rank, world_size):
    """对指标做 all_reduce 后由 rank 0 记录。"""
    for key, value in metrics_dict.items():
        tensor = torch.tensor(value, device=f"cuda:{rank}")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        metrics_dict[key] = tensor.item() / world_size

    if rank == 0:
        wandb.log(metrics_dict, step=step)

# 使用示例
def train_step(model, batch, optimizer, rank, world_size, step):
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # all_reduce loss 后上报
    log_all_reduce(
        {"train/loss": loss.item()},
        step=step,
        rank=rank,
        world_size=world_size,
    )
```

## 多节点训练

### torchrun 方式

```bash
# Node 0 (master)
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=node0 \
    --master_port=29500 \
    train.py

# Node 1
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=node0 \
    --master_port=29500 \
    train.py
```

确保所有节点都设置了 `WANDB_API_KEY` 环境变量。

### SLURM 集群

```bash
#!/bin/bash
#SBATCH --job-name=llm-train
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64

export WANDB_PROJECT="llm-pretrain"
export WANDB_API_KEY="your-key"

# SLURM 自动设置 RANK, WORLD_SIZE 等
srun torchrun \
    --nproc_per_node=8 \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$(scontrol show hostname $SLURM_NODELIST | head -n 1) \
    train.py --report_to wandb
```

## 每个 Rank 独立记录（调试用）

某些场景需要每个 rank 独立上报（如调试通信性能）：

```python
import os

rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

wandb.init(
    project="llm-debug",
    group="comm-profiling",        # 所有 rank 归为一组
    name=f"rank-{rank}-gpu-{local_rank}",
    tags=[f"node-{rank // 8}", f"gpu-{local_rank}"],
    config={
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
    },
)

# 每个 rank 记录自己的指标
wandb.log({
    f"rank_{rank}/loss": local_loss,
    f"rank_{rank}/gpu_util": gpu_util,
    f"rank_{rank}/comm_time": comm_time,
    f"rank_{rank}/nccl_bandwidth_gbps": bandwidth,
})
```

## 通信开销监控

```python
import time
import torch.distributed as dist

class CommProfiler:
    """监控分布式通信开销。"""

    def __init__(self):
        self.all_reduce_time = 0
        self.all_gather_time = 0
        self.broadcast_time = 0
        self._timer = None

    def start(self, op_type):
        torch.cuda.synchronize()
        self._timer = time.time()
        self._op = op_type

    def stop(self):
        torch.cuda.synchronize()
        elapsed = time.time() - self._timer
        if self._op == "all_reduce":
            self.all_reduce_time += elapsed
        elif self._op == "all_gather":
            self.all_gather_time += elapsed
        elif self._op == "broadcast":
            self.broadcast_time += elapsed
        return elapsed

    def log(self, step):
        if int(os.environ.get("RANK", 0)) == 0:
            wandb.log({
                "comm/all_reduce_ms": self.all_reduce_time * 1000,
                "comm/all_gather_ms": self.all_gather_time * 1000,
                "comm/broadcast_ms": self.broadcast_time * 1000,
                "comm/total_ms": (
                    self.all_reduce_time +
                    self.all_gather_time +
                    self.broadcast_time
                ) * 1000,
            }, step=step)
        self.all_reduce_time = 0
        self.all_gather_time = 0
        self.broadcast_time = 0
```

---

*← 返回：[[LLM 训练/微调/对齐实战集成]]*
