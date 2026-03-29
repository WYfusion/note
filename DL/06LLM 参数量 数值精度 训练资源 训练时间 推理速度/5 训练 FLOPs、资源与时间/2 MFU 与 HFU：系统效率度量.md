## 概述

MFU（Model FLOPs Utilization）和 HFU（Hardware FLOPs Utilization）是衡量训练系统效率的核心指标。

---

## MFU 定义

$$\text{MFU} = \frac{\text{Model FLOPs per step} / \text{step time}}{G \times F_{peak}}$$

- **分子**：模型理论 FLOPs（$6ND$ 中每步的部分）除以实际耗时 → 实际算力

- **分母**：集群峰值算力

- **含义**：实际使用了多少比例的硬件峰值算力

> [!important]
> 
> MFU 不计入 activation checkpointing 的重算 FLOPs，因此更反映**模型本身**的效率。

---

## HFU 定义

$$\text{HFU} = \frac{\text{Total actual FLOPs (含重算)} / \text{step time}}{G \times F_{peak}}$$

- 包含 activation checkpointing 的额外前向重算

- HFU ≥ MFU

- **含义**：硬件实际在做多少计算

---

## 典型值参考

|场景|MFU 范围|说明|
|---|---|---|
|理想上限|~55-65%|大模型 + 大 batch + 良好 kernel|
|良好|40-55%|多数大规模训练|
|可接受|30-40%|MoE / 长序列 / 复杂并行|
|需优化|<30%|严重通信/I/O 瓶颈|

### 实际案例

- **PaLM-540B**：MFU ~46.2%（A100）

- **LLaMA-2-70B**：MFU ~38-42%（A100）

- **DeepSeek-V3**：据报道 MFU ~30%（H800 FP8，含 MoE 通信开销）

---

## 计算方法

```Python
def compute_mfu(
    N_active: float,      # 每 token 激活参数
    tokens_per_step: int,  # 每步处理的 token 数
    step_time: float,      # 每步耗时 (秒)
    n_gpus: int,           # GPU 数
    peak_flops: float,     # 单卡峰值 FLOPs/s
) -> float:
    """计算 MFU"""
    # 模型 FLOPs per step (6N per token for training)
    model_flops = 6 * N_active * tokens_per_step
    # 实际算力
    actual_flops_per_sec = model_flops / step_time
    # 峰值算力
    peak_total = n_gpus * peak_flops
    return actual_flops_per_sec / peak_total

# 示例：LLaMA-2-7B on 64xA100
mfu = compute_mfu(
    N_active=7e9,
    tokens_per_step=64 * 2048 * 4,  # 64 GPUs × 2048 seq × 4 micro-batch
    step_time=1.5,  # 秒
    n_gpus=64,
    peak_flops=312e12  # A100 BF16
)
print(f"MFU: {mfu:.1%}")  # ~40%
```

---

## MFU 降低的常见原因

1. **通信开销**：TP/PP/EP all-reduce/all-to-all

1. **Bubble**：PP 流水线空泡

1. **小矩阵**：Tensor Core 利用率低

1. **Activation checkpointing**：额外前向计算（MFU 不计入但影响 wall-clock）

1. **I/O**：数据加载、checkpoint 写入

1. **负载不均衡**：MoE expert 偏斜