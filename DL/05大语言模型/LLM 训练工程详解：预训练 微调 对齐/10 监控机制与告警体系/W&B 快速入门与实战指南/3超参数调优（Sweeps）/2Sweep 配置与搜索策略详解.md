# Sweep 配置与搜索策略详解

深入 Sweep 配置的每一个细节，包括分布类型、嵌套参数、条件参数和贝叶斯优化原理。

---
## 完整配置结构
```yaml
# sweep.yaml
program: train.py            # 训练脚本
method: bayes                # bayes / grid / random
name: lr-bs-sweep            # Sweep 名称

metric:
  name: val/loss
  goal: minimize             # minimize / maximize

parameters:
  lr:
    min: 1e-5
    max: 1e-2
    distribution: log_uniform_values
  batch_size:
    values: [8, 16, 32, 64]
  weight_decay:
    min: 0.0
    max: 0.3
    distribution: uniform
  warmup_steps:
    min: 0
    max: 500
    distribution: int_uniform
  lora_r:
    values: [4, 8, 16, 32, 64]
  optimizer:
    values: ["adamw", "adam", "sgd"]
  seed:
    value: 42                # 固定值

early_terminate:
  type: hyperband
  min_iter: 3
  eta: 3

command:
  - ${env}
  - python
  - ${program}
  - ${args}
```
## 分布类型全解

| **分布** | **参数** | **适用场景** | **数学定义** |
| --- | --- | --- | --- |
| `uniform` | `min`, `max` | dropout、weight_decay | $U(a, b)$ |
| `log_uniform_values` | `min`, `max` | **学习率**（推荐） | $\exp(U(\ln a, \ln b))$ |
| `int_uniform` | `min`, `max` | 层数、warmup_steps | $\lfloor U(a, b) \rfloor$ |
| `normal` | `mu`, `sigma` | 已知大致范围的参数 | $N(\mu, \sigma^2)$ |
| `log_normal` | `mu`, `sigma` | 量级跨度大的正值参数 | $\exp(N(\mu, \sigma^2))$ |
| `q_uniform` | `min`, `max`, `q` | 步长为 q 的离散连续值 | $\text{round}(U(a,b) / q) \times q$ |
| `categorical` | `values` | 离散选择（隐含） | 均匀抽取 values 列表 |
| `constant` | `value` | 固定不搜索 | — |

### 学习率为什么用 log_uniform？
学习率搜索范围通常横跨多个量级（如 $10^{-5}$ 到 $10^{-2}$），`uniform` 会导致 99% 的采样落在 $10^{-2}$ 附近，而 `log_uniform_values` 确保每个量级被均匀探索。
## 嵌套参数
```python
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val/loss", "goal": "minimize"},
    "parameters": {
        "model": {
            "parameters": {
                "hidden_size": {"values": [256, 512, 1024]},
                "num_layers": {"min": 2, "max": 12, "distribution": "int_uniform"},
                "dropout": {"min": 0.0, "max": 0.5},
            }
        },
        "training": {
            "parameters": {
                "lr": {"min": 1e-5, "max": 1e-2, "distribution": "log_uniform_values"},
                "batch_size": {"values": [16, 32, 64]},
            }
        },
    },
}

# 在训练函数中访问
def train():
    run = wandb.init()
    config = wandb.config
    hidden_size = config.model["hidden_size"]  # 嵌套访问
    lr = config.training["lr"]
```

## 贝叶斯优化原理

贝叶斯搜索 (`method: "bayes"`) 的核心思想：
1. **高斯过程代理模型**：基于已完成的 Run 拟合一个「指标 vs 超参」的概率模型
2. **采集函数（Acquisition Function）**：综合「预期改进」和「不确定性」选择下一组参数
3. **迭代更新**：每完成一个 Run，更新代理模型，选择更优参数
$$
\alpha(x) = \mathbb{E}[\max(f(x) - f^+, 0)]
$$
其中 $f^+$ 是当前最优值，$alpha(x)$ 是 Expected Improvement。
**优势**：
- 比 Random 快 2-5x 找到好参数
- 自动在探索（exploration）和利用（exploitation）之间平衡
- 随着 Run 增多，搜索越来越精准
## LLM 微调常用 Sweep 模板
```python
llm_sft_sweep = {
    "method": "bayes",
    "metric": {"name": "eval/loss", "goal": "minimize"},
    "parameters": {
        # LoRA 参数
        "lora_r": {"values": [4, 8, 16, 32, 64]},
        "lora_alpha": {"values": [16, 32, 64, 128]},
        "lora_dropout": {"min": 0.0, "max": 0.1, "distribution": "uniform"},

        # 训练参数
        "learning_rate": {"min": 1e-5, "max": 5e-4, "distribution": "log_uniform_values"},
        "warmup_ratio": {"min": 0.0, "max": 0.1},
        "weight_decay": {"min": 0.0, "max": 0.1},
        "lr_scheduler_type": {"values": ["cosine", "linear", "constant_with_warmup"]},

        # 固定参数
        "num_train_epochs": {"value": 3},
        "per_device_train_batch_size": {"value": 4},
        "gradient_accumulation_steps": {"value": 8},
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 1,  # 至少 1 个 epoch
        "eta": 2,
    },
}
```

---

*← 返回：[[1超参数调优（Sweeps）]]*
