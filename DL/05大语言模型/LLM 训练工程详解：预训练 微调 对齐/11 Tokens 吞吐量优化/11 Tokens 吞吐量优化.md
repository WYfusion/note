# 11. Tokens 吞吐量优化

吞吐量 = 单位时间处理的 token 数，直接决定训练成本和时间。优化吞吐是在「保证训练质量」前提下的核心工程目标。

---

## 核心指标

| 指标 | 含义 | 基准 |
| --- | --- | --- |
| `global tokens/sec` | 整个集群每秒处理的 token 数 | 看总训练时间 |
| `per-GPU tokens/sec` | 单卡吞吐 | 衡量单卡效率 |
| `step time` | 单步训练耗时 | 需分解各阶段 |
| `MFU / HFU` | 模型 / 硬件浮点利用率 | 理论峰值的 40%~60% 算良好 |
| `input pipeline stall ratio` | 数据加载等待占比 | < 5% 为健康 |

---

## 不正常信号

- **GPU util 低但 CPU/IO 忙** → 数据加载瓶颈
- **step time 抖动大** → 长序列不均匀 / 通信不稳
- **长序列比例变化导致吞吐塌陷** → 需要 packing 或 bucketing
- **多卡后 scaling 很差** → 通信瓶颈 / 并行策略不当

---

## 常见优化手段

### 计算优化

- `flash attention`：减少显存 + 加速 attention 计算
- `fused optimizer / fused kernels`：减少 kernel launch 开销
- `bf16 / fp8`：低精度加速
- `activation checkpointing`：用计算换显存
- `compile / graph capture`：torch.compile / CUDA graph

### 数据优化

- `序列打包 / packing`：短序列拼接，减少 padding 浪费
- `bucketing by length`：相近长度分组，减少 padding
- `更多 dataloader workers / 预取 / pin_memory`

### 并行与通信

- `并行策略重排`：DP / TP / PP / EP 的最优组合
- `通信重叠`：computation-communication overlap
- `异步 checkpoint`：存盘不阻塞训练

---

## 判别口径

<aside>
📐

1. **先看单卡基线**：单卡 MFU 是否达标
2. **再看多卡 scaling**：线性 scaling 效率
3. **再分解 data / compute / comm**：找到真正瓶颈
</aside>

→ 详见子页面 [[吞吐优化技术栈 Python 实战]]

[吞吐优化技术栈 Python 实战](吞吐优化技术栈%20Python%20实战.md)