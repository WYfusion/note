# QLoRA 详解与实战

QLoRA（Quantized LoRA）在 LoRA 基础上，**将底座模型量化到 4-bit** 后再做 LoRA 微调，显存需求降低约 75%。是资源受限下的首选方案。

---

## 核心创新

### NF4 量化

QLoRA 使用 **NormalFloat 4-bit (NF4)** 量化格式：

- 假设预训练权重近似正态分布
- NF4 的量化点分布在正态分布的分位数上
- 信息论上最优的 4-bit 数据类型

### Double Quantization

量化的量化参数也做量化，进一步节省显存：

- 第一层量化：将 fp16 权重量化为 NF4
- 第二层量化：将量化常数（scale factors）从 fp32 量化为 fp8
- 额外节省约 0.5 GB / 每 10 亿参数

### Paged Optimizers

利用 CPU 内存处理梯度检查点中的内存峰值：

- GPU 显存不足时自动将优化器状态转移到 CPU
- 防止 OOM（out-of-memory）

---

## 显存对比

| 方法 | 7B 模型显存 | 70B 模型显存 |
| --- | --- | --- |
| Full Fine-tuning (fp16) | ~100 GB | ~1000 GB |
| LoRA (fp16) | ~16 GB | ~160 GB |
| **QLoRA (NF4)** | **~6 GB** | **~48 GB** |

> QLoRA 让 7B 模型可在 **单张 RTX 3090/4090 (24GB)** 上微调！
>

---

## Python 实战代码

```python
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# 1. 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NF4 量化
    bnb_4bit_compute_dtype=torch.bfloat16, # 计算精度
    bnb_4bit_use_double_quant=True,       # Double Quantization
)

# 2. 加载量化模型
model_name = "meta-llama/Llama-3.1-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 3. 准备量化模型进行训练
model = prepare_model_for_kbit_training(model)

# 4. 配置 LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 5. 训练配置
training_args = TrainingArguments(
    output_dir="./qlora-output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",      # Paged 优化器
)

# 6. 训练
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    max_seq_length=2048,
)
trainer.train()

# 7. 保存 adapter
model.save_pretrained("./qlora-adapter")
```

### 推理（合并需要 dequantize）

```python
# 方法一：直接加载 adapter 推理（不合并）
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "./qlora-adapter")

# 方法二：合并后推理（需要先 dequantize）
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # 不量化
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "./qlora-adapter")
model = model.merge_and_unload()  # 合并
model.save_pretrained("./merged-model")
```

---

## QLoRA vs LoRA 选择指南

| 场景 | 推荐 | 原因 |
| --- | --- | --- |
| 单卡 24GB + 7B 模型 | QLoRA | LoRA 放不下 |
| 多卡 80GB + 7B 模型 | LoRA | 显存够用，LoRA 更快 |
| 单卡 24GB + 70B 模型 | 不可能 | QLoRA 也需要 ~48GB |
| 追求极致精度 | LoRA (fp16) | 量化有精度损失 |
| 追求最低成本 | QLoRA | 显存最省 |

---

## 注意事项

- QLoRA 的训练速度比 LoRA **慢约 30-40%**（量化/反量化有开销）
- 合并权重时需要先将底座 dequantize 回 fp16
- 量化可能导致微小的精度损失，但在大多数任务上可忽略
