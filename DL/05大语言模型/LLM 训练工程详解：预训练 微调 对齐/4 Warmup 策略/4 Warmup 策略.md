# 4. Warmup 策略

Warmup = 训练初期从极低 LR 逐步升到目标 LR 的阶段。目的是限制早期 update 过大——此时 optimizer 的二阶矩估计（如 Adam 的 $v_t$）尚未稳定，大步更新极易导致发散。

---

## 受影响项

warmup 长度需要根据以下因素调整：

- **batch size**：大 batch 梯度更准但 update 也更大 → 需更长 warmup
- **初始化尺度**：若权重初始化偏大，早期梯度也大
- **优化器 beta**：Adam $\beta_2$ 越大，二阶矩预热越慢
- **数据分布切换**：新数据源引入时相当于重新 warmup
- **是否从 checkpoint 继续**：若 resume 且分布接近 → 可更短

---

## 常用设置

| 模式 | 范围 | 适用 |
| --- | --- | --- |
| 按 ratio | 0.5% ~ 3% of total steps | 大规模预训练 |
| 按 step | 几百到几千 steps | 微调 / 小规模训练 |

**经验法则**：

- 大 batch / 全新训练 / 数据更脏 → 更长 warmup
- 继续训练且分布接近 → 可更短甚至跳过

---

## 不足 vs 过长
⚠️
- **Warmup 不足**：前几百 step 爆 loss / grad norm → 可能直接崩溃
- **Warmup 过长**：前期 LR 太低，token 利用率差，浪费算力

→ 详见子页面 [[Warmup 与 Scheduler 实现示例]]

[Warmup 与 Scheduler 实现示例](Warmup%20与%20Scheduler%20实现示例.md)