# 吞吐优化技术栈 Python 实战

本页面提供各种吞吐优化技术的实现示例，包括序列 packing、MFU 计算、并行策略配置等。

---

## MFU / HFU 计算

**MFU**（Model FLOPs Utilization）= 实际模型计算量 / 硬件峰值算力。

Transformer 的近似 FLOPs 公式：

$\text{FLOPs per token} \approx 6 \times P$

其中 $P$ 为模型参数量（因为 forward + backward ≈ 3x forward，forward ≈ 2P FLOPs）。

```python
def compute_mfu(
    tokens_per_sec: float,
    num_params: int,
    hardware_tflops: float,  # 硬件峰值（bf16 TFLOPS）
) -> float:
    """
    计算 Model FLOPs Utilization。

    Args:
        tokens_per_sec: 全局 tokens/sec
        num_params: 模型参数量
        hardware_tflops: 硬件峰值 bf16 TFLOPS

    硬件参考：
    - A100 SXM: 312 TFLOPS (bf16)
    - H100 SXM: 989 TFLOPS (bf16)
    - H200 SXM: 989 TFLOPS (bf16)
    """
    flops_per_token = 6 * num_params  # forward + backward
    achieved_tflops = (tokens_per_sec * flops_per_token) / 1e12
    mfu = achieved_tflops / hardware_tflops
    return mfu

# 示例：7B 模型在 8×A100 上达到 50K tokens/sec
mfu = compute_mfu(
    tokens_per_sec=50000,
    num_params=7e9,
    hardware_tflops=312 * 8,  # 8 卡 A100
)
print(f"MFU: {mfu:.2%}")  # ~= 40%
```

---

## 序列 Packing

将多条短序列拼接成一条长序列，减少 padding 浪费：

```python
import random

def pack_sequences(
    sequences: list[list[int]],
    max_seq_len: int,
    pad_token_id: int = 0,
    eos_token_id: int = 2,
) -> list[dict]:
    """
    贪心 packing：将多条序列拼接到 max_seq_len。

    Returns:
        [{"input_ids": [...], "attention_mask": [...], "position_ids": [...]}]
    """
    packed = []
    current_ids = []
    current_positions = []
    current_mask = []
    current_len = 0

    for seq in sequences:
        seq_len = len(seq)

        # 如果放不下，先完成当前 pack
        if current_len + seq_len > max_seq_len and current_ids:
            # Pad 到 max_seq_len
            pad_len = max_seq_len - current_len
            current_ids.extend([pad_token_id] * pad_len)
            current_mask.extend([0] * pad_len)
            current_positions.extend([0] * pad_len)
            packed.append({
                "input_ids": current_ids,
                "attention_mask": current_mask,
                "position_ids": current_positions,
            })
            current_ids, current_positions, current_mask = [], [], []
            current_len = 0

        # 添加序列
        current_ids.extend(seq)
        current_positions.extend(list(range(seq_len)))  # 每条序列独立位置编码
        current_mask.extend([1] * seq_len)
        current_len += seq_len

    # 最后一个 pack
    if current_ids:
        pad_len = max_seq_len - current_len
        current_ids.extend([pad_token_id] * pad_len)
        current_mask.extend([0] * pad_len)
        current_positions.extend([0] * pad_len)
        packed.append({
            "input_ids": current_ids,
            "attention_mask": current_mask,
            "position_ids": current_positions,
        })

    return packed

# 示例
seqs = [[1,2,3], [4,5,6,7,8], [9,10]]
result = pack_sequences(seqs, max_seq_len=12)
# 拼接后: [1,2,3,4,5,6,7,8,9,10,0,0]  而不是 3 条各自 padding
```

---

## Length Bucketing

按长度分组，同一 batch 内序列长度接近，减少 padding：

```python
from torch.utils.data import Sampler
import numpy as np

class LengthBucketSampler(Sampler):
    """按序列长度分桶采样，减少 padding 浪费。"""

    def __init__(self, lengths: list[int], batch_size: int, num_buckets: int = 10):
        self.lengths = np.array(lengths)
        self.batch_size = batch_size

        # 按长度排序后分桶
        sorted_indices = np.argsort(self.lengths)
        bucket_size = len(sorted_indices) // num_buckets
        self.buckets = [
            sorted_indices[i * bucket_size:(i + 1) * bucket_size]
            for i in range(num_buckets)
        ]
        # 剩余的放入最后一个桶
        if len(sorted_indices) % num_buckets:
            self.buckets[-1] = np.concatenate([
                self.buckets[-1],
                sorted_indices[num_buckets * bucket_size:]
            ])

    def __iter__(self):
        # 桶内 shuffle，桶间 shuffle
        all_indices = []
        bucket_order = list(range(len(self.buckets)))
        random.shuffle(bucket_order)

        for bi in bucket_order:
            indices = self.buckets[bi].copy()
            np.random.shuffle(indices)
            all_indices.extend(indices.tolist())

        # 按 batch 切分
        batches = [
            all_indices[i:i+self.batch_size]
            for i in range(0, len(all_indices), self.batch_size)
        ]
        random.shuffle(batches)
        return iter([idx for batch in batches for idx in batch])

    def __len__(self):
        return len(self.lengths)
```

---

## DataLoader 优化

```python
from torch.utils.data import DataLoader

def create_optimized_dataloader(
    dataset,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
):
    """创建优化过的 DataLoader，减少数据加载等待。"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,            # 固定内存，加速 CPU->GPU 传输
        prefetch_factor=prefetch_factor,   # 每个 worker 预取几个 batch
        persistent_workers=True,           # worker 常驻，避免重复启动
        drop_last=True,                    # 丢弃最后不完整 batch
    )
```

---

## Activation Checkpointing

```python
from torch.utils.checkpoint import checkpoint

def apply_activation_checkpointing(model, layers_attr="model.layers"):
    """对 Transformer 各层应用 activation checkpointing。"""
    layers = model
    for attr in layers_attr.split("."):
        layers = getattr(layers, attr)

    for i, layer in enumerate(layers):
        # 将每层的 forward 替换为 checkpoint 版本
        original_forward = layer.forward

        def make_ckpt_forward(fn):
            def ckpt_forward(*args, **kwargs):
                return checkpoint(fn, *args, use_reentrant=False, **kwargs)
            return ckpt_forward

        layer.forward = make_ckpt_forward(original_forward)

    print(f"✅ Applied activation checkpointing to {len(layers)} layers")
```

---

## 吞吐诊断器

```python
import torch
import subprocess

class ThroughputDiagnostic:
    """诊断吞吐瓶颈：data vs compute vs communication。"""

    def __init__(self, num_gpus: int, hardware_tflops_per_gpu: float):
        self.num_gpus = num_gpus
        self.hw_tflops = hardware_tflops_per_gpu

    def diagnose(self, step_times: dict, tokens_per_sec: float, num_params: int) -> str:
        """
        Args:
            step_times: {"data": float, "compute": float, "comm": float}
            tokens_per_sec: 全局吞吐
            num_params: 模型参数

        Returns: 诊断报告字符串
        """
        total = sum(step_times.values())
        report = []
        report.append("=== 吞吐诊断报告 ===")

        # 时间分解
        for name, t in step_times.items():
            pct = t / total * 100
            report.append(f"  {name}: {t*1000:.1f}ms ({pct:.1f}%)")
            if name == "data" and pct > 20:
                report.append("    → 瓶颈！建议: 增加 workers / prefetch / pin_memory")
            elif name == "comm" and pct > 30:
                report.append("    → 瓶颈！建议: 调整 TP/PP/DP 比例或开启 overlap")

        # MFU
        mfu = compute_mfu(tokens_per_sec, num_params, self.hw_tflops * self.num_gpus)
        report.append(f"\n  MFU: {mfu:.2%}")
        if mfu < 0.3:
            report.append("    → MFU 偏低，检查并行策略、flash attention、compile")
        elif mfu < 0.5:
            report.append("    → MFU 中等，可尝试 fp8 matmul / fused kernels")
        else:
            report.append("    → MFU 良好 ✅")

        # Scaling 效率
        per_gpu_tps = tokens_per_sec / self.num_gpus
        report.append(f"\n  Per-GPU tokens/sec: {per_gpu_tps:.0f}")

        return "\n".join(report)
```