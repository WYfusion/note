# SFT / DPO / RLHF 对齐训练监控

对齐训练（Alignment）有独特的指标体系，理解这些指标是判断训练质量的关键。

---

## 对齐阶段指标总览

| **阶段** | **核心指标** | **健康信号** | **异常信号** |
| --- | --- | --- | --- |
| SFT | train/loss, eval/loss | 稳步下降，eval 不发散 | eval loss 持续上升 |
| DPO | rewards/margins, rewards/accuracies | margin 增大，accuracy > 0.5 | accuracy 下降或 margin 为负 |
| PPO/RLHF | ppo/mean_scores, ppo/kl | reward 上升，KL 受控 | KL 爆炸或 reward hacking |

## SFT 监控

### 标准指标

```python
# TRL SFTTrainer 自动记录:
# train/loss        - 训练 loss
# eval/loss         - 验证 loss
# train/grad_norm   - 梯度范数
# train/learning_rate - 当前学习率
```

### 自定义 SFT 监控

```python
from transformers import TrainerCallback
import wandb

class SFTMonitorCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.train_losses.append(logs["loss"])

            # Perplexity
            wandb.log({
                "sft/perplexity": 2 ** logs["loss"],
            }, step=state.global_step)

            # Loss spike 检测
            if len(self.train_losses) > 20:
                mean = sum(self.train_losses[-20:]) / 20
                if logs["loss"] > mean * 2:
                    wandb.alert(
                        title="SFT Loss Spike",
                        text=f"Loss {logs['loss']:.4f} >> mean {mean:.4f}",
                        level=wandb.AlertLevel.WARN,
                    )
```

## DPO 监控

### DPO 自动记录的指标

```python
# TRL DPOTrainer 自动记录:
# train/loss                - DPO loss
# rewards/chosen            - chosen response 的隐式 reward
# rewards/rejected          - rejected response 的隐式 reward
# rewards/margins           - chosen - rejected 的差值
# rewards/accuracies        - chosen > rejected 的比例
# logps/chosen              - chosen 的 log probability
# logps/rejected            - rejected 的 log probability
```

### DPO 指标解读

**rewards/margins**（最重要）：

- 应该**逐渐增大** → 模型学会区分好坏
- 如果为负或下降 → 训练有问题

**rewards/accuracies**：

- 理想范围：0.6 - 0.8
- 太高（>0.95） → 过拟合偏好数据
- 太低（<0.5） → 模型没学到偏好

**logps/chosen 和 logps/rejected**：

- 两者都不应该剧烈下降 → 否则模型在遗忘
- 差距应该增大但不能太大

### DPO 告警系统

```python
class DPOMonitorCallback(TrainerCallback):
    def __init__(self):
        self.margins = []
        self.accuracies = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        alerts = []

        # Margin 监控
        margin = logs.get("rewards/margins")
        if margin is not None:
            self.margins.append(margin)
            if margin < 0:
                alerts.append(f"🔴 Negative margin: {margin:.4f}")
            if len(self.margins) > 10:
                recent = sum(self.margins[-5:]) / 5
                earlier = sum(self.margins[-10:-5]) / 5
                if recent < earlier * 0.8:
                    alerts.append(f"🟡 Margin decreasing: {earlier:.4f} → {recent:.4f}")

        # Accuracy 监控
        acc = logs.get("rewards/accuracies")
        if acc is not None:
            self.accuracies.append(acc)
            if acc < 0.5:
                alerts.append(f"🔴 Low accuracy: {acc:.4f}")
            if acc > 0.95:
                alerts.append(f"🟡 Very high accuracy (overfitting?): {acc:.4f}")

        # 额外指标
        chosen_logp = logs.get("logps/chosen")
        rejected_logp = logs.get("logps/rejected")
        if chosen_logp is not None and rejected_logp is not None:
            wandb.log({
                "dpo/logp_gap": chosen_logp - rejected_logp,
                "dpo/chosen_logp": chosen_logp,
                "dpo/rejected_logp": rejected_logp,
            }, step=state.global_step)

        if alerts:
            wandb.alert(
                title=f"DPO Alert @ step {state.global_step}",
                text="\n".join(alerts),
                level=wandb.AlertLevel.WARN,
            )
```

## PPO / RLHF 监控

### PPO 自动记录的指标

```python
# TRL PPOTrainer 自动记录:
# ppo/loss/policy      - 策略损失
# ppo/loss/value       - 价值函数损失
# ppo/loss/total       - 总损失
# ppo/mean_scores      - 平均 reward model 分数
# ppo/kl               - 与参考模型的 KL 散度
# ppo/entropy          - 策略熵
# ppo/clipfrac         - PPO clip 比例
# ppo/approxkl         - 近似 KL
```

### RLHF 关键指标解读

**ppo/mean_scores**：

- 应该**稳步上升** → 模型生成质量在改善
- 如果突然跳升后不再变化 → 可能 reward hacking

**ppo/kl**：

- 应该保持**适度且稳定**
- 如果持续增大 → 模型偏离太远，需要加大 KL 惩罚
- 典型健康范围：0.5 - 10

**ppo/entropy**：

- 逐渐下降是正常的（模型更确定）
- 如果降到接近 0 → 模式坍缩（mode collapse）

### RLHF 告警

```python
class RLHFMonitorCallback:
    def __init__(self, kl_threshold=15, reward_window=50):
        self.kl_threshold = kl_threshold
        self.rewards = []

    def on_step(self, stats):
        alerts = []

        # KL 散度
        kl = stats.get("ppo/kl", 0)
        if kl > self.kl_threshold:
            alerts.append(f"🔴 KL divergence too high: {kl:.2f}")

        # Reward hacking 检测
        reward = stats.get("ppo/mean_scores", 0)
        self.rewards.append(reward)
        if len(self.rewards) > self.reward_window:
            recent = self.rewards[-10:]
            if max(recent) - min(recent) < 0.01 and reward > 0.9:
                alerts.append("🟡 Possible reward hacking: reward saturated")

        # 模式坍缩
        entropy = stats.get("ppo/entropy", 1)
        if entropy < 0.1:
            alerts.append(f"🔴 Mode collapse risk: entropy={entropy:.4f}")

        if alerts:
            wandb.alert(
                title="RLHF Alert",
                text="\n".join(alerts),
                level=wandb.AlertLevel.WARN,
            )
```

## 对齐质量的人工评估集成

```python
# 定期在 W&B Table 中记录生成样本
def log_alignment_samples(model, tokenizer, test_prompts, step):
    table = wandb.Table(columns=[
        "Prompt", "Response", "Length", "Step",
    ])

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=256,
                temperature=0.7, do_sample=True,
            )
        response = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        table.add_data(prompt, response, len(response), step)

    wandb.log({"alignment/samples": table}, step=step)
```

---

*← 返回：[[LLM 训练/微调/对齐实战集成]]*
