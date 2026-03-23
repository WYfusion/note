# DoRA 与 rsLoRA 详解

DoRA（Weight-Decomposed Low-Rank Adaptation）和 rsLoRA（Rank-Stabilized LoRA）是 LoRA 的两个重要改进变体。

---

## DoRA 核心原理

### 动机

研究发现全参数微调的权重更新同时改变了权重的**方向**和**幅度**，而 LoRA 的低秩更新主要改变方向，幅度调整不够充分。

### 数学形式

DoRA 将权重分解为**幅度** $m$ 和**方向** $V$：

$$
W = m \cdot \frac{V}{\|V\|_c} = m \cdot \frac{W_0 + BA}{\|W_0 + BA\|_c}
$$

- $m in mathbb{R}^{1 times k}$：可训练的幅度向量（每列一个缩放值）
- $V = W_0 + BA$：方向矩阵（LoRA 更新方向）
- $|cdot|_c$：按列求范数

### 训练参数

- LoRA 的 $A$ 和 $B$（更新方向）
- 幅度向量 $m$（更新幅度）
- 总参数量比 LoRA 多一个 $m$ 向量（可忽略）

### 效果

- 在多个 benchmark 上超越 LoRA 和 Full FT
- 更接近全参数微调的更新模式
- 代价极小（仅多一个向量）

---

## DoRA 代码实现

```python
from peft import LoraConfig, get_peft_model

# DoRA 在 PEFT 库中通过 use_dora=True 开启
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_dora=True,   # 🔑 开启 DoRA
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 比 LoRA 多一点点参数（幅度向量 m）
```

---

## rsLoRA 核心原理

### 动机

标准 LoRA 的缩放是 $alpha / r$，当增大 rank $r$ 时，缩放变小，导致**高秩的 LoRA 效果反而可能变差**。

### 数学形式

rsLoRA 将缩放因子改为：

$$
\text{LoRA:} \quad \frac{\alpha}{r} \qquad \text{rsLoRA:} \quad \frac{\alpha}{\sqrt{r}}
$$

- 数学推导来自随机矩阵理论
- 当 $r$ 增大时，$1/sqrt{r}$ 比 $1/r$ 下降更慢
- 让高秩 LoRA 能充分利用额外的表达力

### 效果

- 高秩（r ≥ 64）时效果显著提升
- 低秩时与标准 LoRA 相当
- 无额外计算成本

---

## rsLoRA 代码实现

```python
from peft import LoraConfig, get_peft_model

# rsLoRA 在 PEFT 库中通过 use_rslora=True 开启
lora_config = LoraConfig(
    r=64,              # 高秩时 rsLoRA 优势更明显
    lora_alpha=128,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_rslora=True,   # 🔑 开启 rsLoRA
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
```

---

## AdaLoRA 简介

自适应分配不同层的秩：

- 重要的层（如中间层的注意力）分配更高的秩
- 不重要的层分配更低的秩
- 通过 SVD 分解 + 重要性评分实现

```python
from peft import AdaLoraConfig, get_peft_model

config = AdaLoraConfig(
    init_r=12,          # 初始秩
    target_r=8,         # 目标平均秩
    tinit=200,          # 开始裁剪的步数
    tfinal=1000,        # 结束裁剪的步数
    deltaT=10,          # 裁剪间隔
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
```

---

## 方法选择指南

| 场景 | 推荐 |
| --- | --- |
| 首选/默认 | LoRA (r=16, alpha=32) |
| 想要更好效果，几乎无额外成本 | DoRA |
| 需要高秩 LoRA (r≥64) | rsLoRA |
| 想自动化秩的分配 | AdaLoRA |
| 显存极其有限 | QLoRA + 任意变体 |