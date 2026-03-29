# 分布式 Sharded Checkpoint 实战

本页面详细介绍大模型训练中分布式 checkpoint 的保存、加载与 reshard 实现。涵盖 FSDP 与 DeepSpeed 两大主流方案。

---

## 核心概念

**Sharded Checkpoint** = 各 rank 只保存自己持有的模型分片和 optimizer state 分片，避免单点汇聚成完整模型的显存和 IO 开销。

$\text{Total checkpoint size} = \text{model params} \times (1 + \text{optimizer multiplier}) \times \text{bytes per param}$

AdamW 的 optimizer multiplier = 2（m 和 v 各一份），所以 7B bf16 模型的 checkpoint 约为 7B × 3 × 2 = 42GB。

---

## 方案一：PyTorch FSDP + StateDictType

### 保存 Sharded Checkpoint

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, ShardedStateDictConfig
from torch.distributed.checkpoint import save as dist_save
from pathlib import Path

def save_sharded_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    step: int,
    seen_tokens: int,
    save_dir: str,
):
    """保存 FSDP sharded checkpoint，各 rank 只写自己的分片。"""
    save_path = Path(save_dir) / f"step-{step}"
    save_path.mkdir(parents=True, exist_ok=True)

    # 1. 设置为 SHARDED_STATE_DICT 模式
    sharded_cfg = ShardedStateDictConfig(offload_to_cpu=True)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, sharded_cfg):
        state = {
            "model": model.state_dict(),
            "optimizer": FSDP.optim_state_dict(model, optimizer),
        }

    # 2. 使用 torch.distributed.checkpoint 保存（各 rank 并行写入）
    dist_save(state, checkpoint_id=str(save_path))

    # 3. 只在 rank 0 写非分片元信息
    if dist.get_rank() == 0:
        meta = {
            "step": step,
            "seen_tokens": seen_tokens,
            "lr_scheduler": scheduler.state_dict(),
            "grad_scaler": scaler.state_dict() if scaler else None,
            "rng_cpu": torch.random.get_rng_state(),
        }
        torch.save(meta, save_path / "meta.pt")

    # 4. 保存各 rank 的 CUDA RNG state
    torch.save(
        torch.cuda.get_rng_state(),
        save_path / f"rng_cuda_rank{dist.get_rank()}.pt",
    )
    dist.barrier()
```

### 加载 Sharded Checkpoint

```python
from torch.distributed.checkpoint import load as dist_load

def load_sharded_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    load_dir: str,
):
    """加载 FSDP sharded checkpoint，自动处理 reshard。"""
    load_path = Path(load_dir)

    # 1. 加载模型和 optimizer 分片
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        state = {
            "model": model.state_dict(),
            "optimizer": FSDP.optim_state_dict(model, optimizer),
        }
        dist_load(state, checkpoint_id=str(load_path))
        model.load_state_dict(state["model"])
        FSDP.optim_state_dict_to_load(
            model, optimizer, state["optimizer"]
        )

    # 2. 加载元信息（所有 rank 都读）
    meta = torch.load(load_path / "meta.pt", map_location="cpu")
    scheduler.load_state_dict(meta["lr_scheduler"])
    if scaler and meta["grad_scaler"]:
        scaler.load_state_dict(meta["grad_scaler"])

    # 3. 恢复 RNG state
    torch.random.set_rng_state(meta["rng_cpu"])
    cuda_rng = torch.load(
        load_path / f"rng_cuda_rank{dist.get_rank()}.pt"
    )
    torch.cuda.set_rng_state(cuda_rng)

    return meta["step"], meta["seen_tokens"]
```

---

## 方案二：DeepSpeed ZeRO Checkpoint

### 保存

```python
import deepspeed

def save_deepspeed_checkpoint(model_engine, step, save_dir):
    """DeepSpeed 自动处理 ZeRO optimizer sharding。"""
    tag = f"step-{step}"
    # save_checkpoint 自动保存:
    #   - model weights (sharded per ZeRO stage)
    #   - optimizer states (sharded)
    #   - scheduler state
    #   - random states
    model_engine.save_checkpoint(
        save_dir=save_dir,
        tag=tag,
        client_state={
            "step": step,
            "seen_tokens": model_engine.global_samples * model_engine.train_batch_size(),
        },
    )
```

### 加载（含 world size 变化）

```python
def load_deepspeed_checkpoint(model_engine, load_dir, tag=None):
    """加载 DeepSpeed checkpoint，支持 world size 变化。"""
    # load_checkpoint 自动处理 reshard
    # load_optimizer_states=True 确保 optimizer 也加载
    _, client_state = model_engine.load_checkpoint(
        load_dir=load_dir,
        tag=tag,  # None = 加载最新
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
    )
    return client_state["step"], client_state["seen_tokens"]
```

---

## World Size 变化时的验证

```python
def verify_resume_after_rescale(model, optimizer, ckpt_path, n_verify_steps=5):
    """在 world size 变化后做 dry-run 验证。"""
    # 1. 加载 checkpoint
    step, seen = load_sharded_checkpoint(model, optimizer, ...)

    # 2. 跑几个 step 看是否 loss 连续正常
    losses = []
    for i in range(n_verify_steps):
        batch = get_fixed_verify_batch()  # 用固定 batch 验证
        loss = train_step(model, optimizer, batch)
        losses.append(loss.item())
        print(f"Verify step {i}: loss={loss.item():.4f}")

    # 3. 检查 loss 是否在合理范围
    mean_loss = sum(losses) / len(losses)
    assert all(abs(l - mean_loss) < mean_loss * 0.5 for l in losses), \
        f"Resume verification failed: losses={losses}"
    print(f"✅ Resume verified: mean_loss={mean_loss:.4f}")
```

---

## 异步落盘

```python
import threading
import shutil

class AsyncCheckpointSaver:
    """后台线程将 checkpoint 从高速盘异步拷贝到远程存储。"""

    def __init__(self, remote_dir: str, max_workers: int = 2):
        self.remote_dir = remote_dir
        self._threads: list[threading.Thread] = []

    def schedule_upload(self, local_path: str):
        def _upload():
            remote_path = Path(self.remote_dir) / Path(local_path).name
            shutil.copytree(local_path, remote_path, dirs_exist_ok=True)
            print(f"✅ Uploaded {local_path} -> {remote_path}")

        t = threading.Thread(target=_upload, daemon=True)
        t.start()
        self._threads.append(t)

    def wait_all(self):
        for t in self._threads:
            t.join()
        self._threads.clear()
```

---

## Checkpoint 管理器（last-k + best-k + milestone）

```python
from collections import deque
import json

class CheckpointManager:
    def __init__(self, save_dir, keep_last=3, keep_best=2):
        self.save_dir = Path(save_dir)
        self.keep_last = keep_last
        self.keep_best = keep_best
        self.last_queue = deque(maxlen=keep_last)
        self.best_queue = []  # (val_loss, path)
        self.milestones = set()

    def should_save(self, step, total_steps, milestone_pct=0.1):
        """判断是否需要保存 milestone。"""
        progress = step / total_steps
        for pct in [i * milestone_pct for i in range(1, int(1/milestone_pct)+1)]:
            if abs(progress - pct) < 0.001 and pct not in self.milestones:
                self.milestones.add(pct)
                return True, "milestone"
        return False, None

    def save_and_manage(self, step, val_loss=None, is_milestone=False):
        path = self.save_dir / f"step-{step}"
        # ... 实际保存逻辑 ...

        # 管理 last-k
        if not is_milestone:
            if len(self.last_queue) == self.keep_last:
                old = self.last_queue[0]
                if old not in [p for _, p in self.best_queue]:
                    shutil.rmtree(old, ignore_errors=True)
            self.last_queue.append(str(path))

        # 管理 best-k
        if val_loss is not None:
            self.best_queue.append((val_loss, str(path)))
            self.best_queue.sort(key=lambda x: x[0])
            while len(self.best_queue) > self.keep_best:
                _, worst_path = self.best_queue.pop()
                if worst_path not in self.last_queue:
                    shutil.rmtree(worst_path, ignore_errors=True)
```