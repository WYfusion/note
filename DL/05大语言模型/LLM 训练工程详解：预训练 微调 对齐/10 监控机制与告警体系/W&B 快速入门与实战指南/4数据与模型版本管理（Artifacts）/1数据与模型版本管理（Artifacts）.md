# 数据与模型版本管理（Artifacts）

Artifacts 是 W&B 的版本管理系统，追踪数据集、模型、代码之间的依赖关系（血缘）。

---

## Artifact 是什么？

**类比**：Git 管理代码版本 → Artifact 管理数据/模型版本。

每个 Artifact 有：

- **类型（type）**：`dataset` / `model` / `code` / 自定义
- **名称**：如 `sft-dataset`
- **版本**：自动递增 `v0`, `v1`, `v2`...
- **别名（alias）**：如 `latest`, `best`, `production`

## 基本用法

### 上传 Artifact

```python
import wandb

run = wandb.init(project="llm-sft", job_type="data-prep")

# 创建 Artifact
artifact = wandb.Artifact(
    name="sft-dataset",
    type="dataset",
    description="SFT 训练数据 v2, 清洗后",
    metadata={"num_samples": 50000, "format": "jsonl"},
)

# 添加文件
artifact.add_file("data/train.jsonl")
artifact.add_file("data/val.jsonl")
# 或添加整个目录
# artifact.add_dir("data/")

# 记录到 W&B
run.log_artifact(artifact)
wandb.finish()
```

### 下载 Artifact

```python
run = wandb.init(project="llm-sft", job_type="training")

# 声明输入依赖
artifact = run.use_artifact("sft-dataset:latest")
data_dir = artifact.download()  # 下载到本地缓存

# 使用数据
print(f"数据目录: {data_dir}")
```

### 模型 Artifact

```python
# 训练结束后保存模型
model_artifact = wandb.Artifact(
    name="llama-sft",
    type="model",
    metadata={
        "base_model": "meta-llama/Llama-3-8B",
        "method": "LoRA r=16",
        "val_loss": 0.85,
    },
)
model_artifact.add_dir("output/checkpoint-best/")
run.log_artifact(model_artifact, aliases=["best", "latest"])
```

## 血缘追踪（Lineage）

W&B 自动构建 DAG（有向无环图）：

$$
\text{raw-data} \xrightarrow{\text{preprocess}} \text{clean-data} \xrightarrow{\text{train}} \text{model} \xrightarrow{\text{eval}} \text{report}
$$

通过 `use_artifact` + `log_artifact` 自动关联输入输出，在 W&B UI 中可视化完整血缘图。

## Model Registry

Model Registry 是 Artifact 之上的模型管理层：

| **功能** | **说明** |
| --- | --- |
| 模型注册 | 将 model Artifact link 到 Registry |
| 阶段管理 | `staging` → `production` → `archived` |
| 自动化 | Webhook 触发部署流水线 |
| 审计 | 谁在什么时候改了什么 |

```python
# 将模型链接到 Registry
run.link_artifact(
    artifact=model_artifact,
    target_path="model-registry/llama-sft",
)
```

## 子页面导航

- **[[2Artifact 版本管理与血缘追踪]]** → 版本控制细节、增量更新、Reference Artifact
- **[[3Model Registry 与生产部署]]** → 阶段管理、Webhook 集成、CI/CD 联动

---

*← 上一节：[[1超参数调优（Sweeps）]]　|　下一节：[[5可视化与协作报告（Reports）]] →*

[3Model Registry 与生产部署](3Model%20Registry%20与生产部署.md)

[2Artifact 版本管理与血缘追踪](2Artifact%20版本管理与血缘追踪.md)