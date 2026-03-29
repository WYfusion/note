# HuggingFace Trainer + W&B 深度集成

HuggingFace Trainer 原生支持 W&B，但要发挥全部潜力需要理解 Callback 机制和高级配置。

---

## 零配置集成

只需两步：

```python
import os
os.environ["WANDB_PROJECT"] = "llm-sft"

from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./output",
    report_to="wandb",  # 就这一行
)
```

## 自动记录的指标

| **指标** | **来源** | **记录频率** |
| --- | --- | --- |
| `train/loss` | 训练 loss | 每 `logging_steps` |
| `train/learning_rate` | 当前学习率 | 每 `logging_steps` |
| `train/grad_norm` | 梯度范数 | 每 `logging_steps` |
| `train/global_step` | 全局步数 | 每 `logging_steps` |
| `train/epoch` | 当前 epoch 进度 | 每 `logging_steps` |
| `eval/loss` | 验证 loss | 每 `eval_steps` |
| `eval/runtime` | 评估用时 | 每 `eval_steps` |
| `train/train_runtime` | 训练总时间 | 训练结束 |
| `train/train_samples_per_second` | 吞吐量 | 训练结束 |

## 关键环境变量

```bash
# 必须
export WANDB_PROJECT="llm-sft"        # 项目名

# 推荐
export WANDB_LOG_MODEL="checkpoint"    # 自动上传 checkpoint 作为 Artifact
# 可选值: "false"(不上传), "end"(训练结束上传), "checkpoint"(每个checkpoint上传)

export WANDB_WATCH="gradients"         # 梯度监控
# 可选值: "false", "gradients", "all"(梯度+参数)

export WANDB_DISABLED="false"          # 全局禁用开关
```

## TrainingArguments W&B 相关参数

```python
args = TrainingArguments(
    output_dir="./output",

    # W&B 配置
    report_to="wandb",
    run_name="llama3-8b-lora-r16",   # W&B Run 名称
    logging_steps=10,                 # 每 N 步记录一次
    logging_first_step=True,          # 记录第一步（调试用）

    # 评估配置（影响 eval 指标上报频率）
    eval_strategy="steps",
    eval_steps=500,

    # Checkpoint（配合 WANDB_LOG_MODEL）
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,               # 只保留最近 3 个
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)
```

## 自定义 Callback

### 记录额外训练指标

```python
from transformers import TrainerCallback
import wandb

class CustomMetricsCallback(TrainerCallback):
    """记录 Trainer 默认不记录的额外指标。"""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        custom = {}

        # Perplexity
        if "loss" in logs:
            custom["train/perplexity"] = 2 ** logs["loss"]
        if "eval_loss" in logs:
            custom["eval/perplexity"] = 2 ** logs["eval_loss"]

        if custom:
            wandb.log(custom, step=state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            wandb.log({
                "eval/samples_per_second": metrics.get("eval_samples_per_second", 0),
            }, step=state.global_step)
```

### 记录样本预测对比

```python
class SamplePredictionCallback(TrainerCallback):
    """每次评估时记录样本预测结果到 W&B Table。"""

    def __init__(self, eval_samples, tokenizer, max_new_tokens=128):
        self.eval_samples = eval_samples[:10]  # 取 10 个样本
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return

        table = wandb.Table(columns=["Input", "Prediction", "Reference"])

        model.eval()
        for sample in self.eval_samples:
            inputs = self.tokenizer(
                sample["input"], return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )

            pred = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            table.add_data(
                sample["input"][:200],
                pred[:500],
                sample.get("output", "")[:500],
            )

        wandb.log({"predictions": table}, step=state.global_step)
```

### 使用 Callback

```python
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    callbacks=[
        CustomMetricsCallback(),
        SamplePredictionCallback(eval_samples, tokenizer),
    ],
)
trainer.train()
```

## Checkpoint 管理

当设置 `WANDB_LOG_MODEL="checkpoint"` 时：

```python
# 训练结束后，最佳模型自动上传为 Artifact
# 可通过 API 下载
api = wandb.Api()
artifact = api.artifact("my-team/llm-sft/model-abc123:best")
model_dir = artifact.download()
```

## 完整生产模板

```python
import os
import wandb
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, TrainerCallback,
)
from peft import LoraConfig, get_peft_model

# 环境变量
os.environ["WANDB_PROJECT"] = "llm-sft"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

# 模型 & LoRA
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    torch_dtype=torch.bfloat16,
)
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear")
model = get_peft_model(model, lora_config)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")

# 训练参数
args = TrainingArguments(
    output_dir="./output",
    report_to="wandb",
    run_name="llama3-8b-lora-r16-alpaca",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    bf16=True,
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    callbacks=[CustomMetricsCallback()],
)
trainer.train()
```

---

*← 返回：[[LLM 训练/微调/对齐实战集成]]*
