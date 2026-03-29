# 1. Checkpoint 与回滚机制

Checkpoint 是训练的「存档点」——训练随时可能因硬件故障、数据异常、超参不当而中断，checkpoint 让你能快速回到安全状态继续训练，而不必从零开始。

---

## 必存内容

完整的 checkpoint 需包含以下所有状态，缺一则无法精确恢复：

| 组件 | 说明 | 缺失后果 |
| --- | --- | --- |
| `model` | 模型权重（含 BN running stats 等） | 无法恢复 |
| `optimizer` | AdamW 的 m/v 状态 | resume 后 LR 行为异常 |
| `lr scheduler` | 当前 step 对应的 LR 值与衰减状态 | LR 重置导致 spike |
| `grad scaler (AMP)` | fp16 下的 loss scale 状态 | overflow 频繁 |
| `RNG states` | Python / NumPy / CUDA 随机种子 | 不可复现 |
| `dataloader / sampler` | 当前数据读取位置 | 重复 / 跳过数据 |
| `global_step / seen_tokens` | 训练进度计数 | scheduler / logging 错位 |
| `config / git commit / tokenizer` | 代码与数据处理版本 | 无法追溯环境差异 |

---

## 保存策略

采用 **step-based 为主 + time-based 为辅** 的双重策略：

- **小环**：每 100~1000 step 存一次 → 用于快速回滚
- **大环**：每 0.5%~2% total steps 存一次 → 用于里程碑对比
- **关键点**：warmup 结束前存更密，稳定后可放稀
- **保留策略**：`last-k` + `best-k` + `milestone`

> 典型配置：last-3 + best-2（按 val loss）+ 每 10% 进度的 milestone
>

---

## 快速回滚

- 最近 **3~5 个 checkpoint** 常驻高速盘（NVMe / 本地 SSD）
- 历史 checkpoint **异步落盘**到对象存储（S3 / GCS / HDFS）
- 出问题时只回滚最近异常前 **1~3 个** checkpoint，不要回滚太远

---

## 分布式 Checkpoint

大模型训练通常跨多卡多节点，checkpoint 需要特殊处理：

- **必须支持 optimizer sharded checkpoint**：FSDP / DeepSpeed ZeRO 下 optimizer state 分片存储，各 rank 只保存自己的分片
- **world size 变化时先验证 resume**：从 8 卡扩到 16 卡，需要先做一次 dry-run 确认 state 能正确 reshard

→ 详见子页面 [[分布式 Sharded Checkpoint 实战]]

[分布式 Sharded Checkpoint 实战](分布式%20Sharded%20Checkpoint%20实战.md)
