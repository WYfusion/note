# 2. 学习率（LR）调优

学习率决定每一步参数更新的幅度——太大则训练不稳甚至崩溃，太小则收敛慢浪费算力。LR 不是一个独立参数，它与模型规模、batch size、精度、阶段等多项因素耦合。

---

## 受影响项

LR 的最优值受以下因素共同影响：

- **模型规模**：参数量越大，通常需要更小的 LR
- **global batch size**：batch 越大，梯度估计越准，可适当提高 LR
- **序列长度**：长序列 loss 更大，需谨慎调整
- **优化器**：AdamW / Adafactor / SGD 各有不同最优区间
- **精度**：bf16 / fp16 / fp8 对数值稳定性影响不同
- **梯度裁剪**：clip 阈值与 LR 共同决定实际 update 幅度
- **数据噪声 / 重复率**：脏数据或重复数据增大梯度方差
- **阶段**：pretrain / SFT / DPO / PPO 各阶段 LR 量级差异大

---

## 常用经验

| 场景 | LR 范围 | 备注 |
| --- | --- | --- |
| Full pretrain（7B） | 1e-4 ~ 3e-4 | 先小规模试验再放大 |
| Full finetune | 1e-5 ~ 5e-5 | 通常低于 pretrain |
| LoRA / QLoRA | 1e-4 ~ 3e-4 | 常可高于 full FT（参数少） |
| PPO actor | 1e-6 ~ 5e-6 | 对齐阶段最保守 |

**batch 变大时**：先考虑按 $\sqrt{k}$ 或线性规则上调 LR，但**必须重新验稳**。

---

## 不稳表现

LR 过大的典型信号：

- `loss spike`：loss 突然飙升
- `grad norm 抬升`：梯度范数持续走高
- `overflow / NaN`：fp16 下 loss scale 频繁跌落
- `验证集同步变差`：不是过拟合，而是训练本身不稳

---

## 判别逻辑

<aside>
🔍

- **train loss 抖 + val 也坏** → 更像 LR 过大
- **train loss 正常、偶发单点爆炸** → 更像脏 batch / 数值问题
</aside>

→ 详见子页面 [[LR Scaling 与 Scheduler 实现]]

[LR Scaling 与 Scheduler 实现](LR%20Scaling%20与%20Scheduler%20实现.md)