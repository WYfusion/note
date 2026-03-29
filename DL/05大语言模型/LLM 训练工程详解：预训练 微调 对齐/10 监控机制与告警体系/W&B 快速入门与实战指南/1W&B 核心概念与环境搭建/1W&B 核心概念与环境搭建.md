# W&B 核心概念与环境搭建

掌握 W&B 的基础概念和环境配置，是所有后续功能的前提。

---
## 安装与登录
```bash
pip install wandb
wandb login  # 输入 API Key（从 wandb.ai/authorize 获取）
```
也可通过环境变量设置（适合 CI/CD 和远程集群）：
```bash
export WANDB_API_KEY="your-api-key"
export WANDB_PROJECT="my-project"
export WANDB_ENTITY="my-team"  # 可选，指定团队
```

## 核心概念

| **概念** | **类比** | **说明** |
| --- | --- | --- |
| **Entity** | GitHub 用户/组织 | 个人账号或团队命名空间 |
| **Project** | GitHub Repo | 一组相关实验的集合，如 `llm-sft` |
| **Run** | 一次 Git Commit | 一次完整的训练/评估过程 |
| **Group** | Git Branch | 将多个 Run 归组（如多卡训练的各 rank） |
| **Job Type** | CI Stage | 区分 train / eval / preprocess 等阶段 |
| **Tags** | Git Tag | 自定义标签，如 `baseline`、`lora-r16` |
| **Config** | 配置文件 | 超参数字典，可搜索、对比、过滤 |
| **Summary** | 最终结果 | 每个指标的最终/最优值，用于 Run 列表排序 |

## 最小可运行示例
```python
import wandb
import random

# 1. 初始化 Run
run = wandb.init(
    project="quickstart",
    config={"lr": 0.001, "epochs": 10, "batch_size": 32},
)

# 2. 模拟训练循环
for epoch in range(run.config["epochs"]):
    loss = random.random() * (1 - epoch / run.config["epochs"])
    acc = 1 - loss + random.random() * 0.1
    wandb.log({"epoch": epoch, "loss": loss, "accuracy": acc})

# 3. 结束
wandb.finish()
```
或者官网的例子
```python
import random
import wandb

# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="fusionwy-guangzhou",
    # Set the wandb project where this run will be logged.
    project="my-awesome-project",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    },
)

# Simulate training.
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset
    # Log metrics to wandb.
    run.log({"acc": acc, "loss": loss})
# Finish the run and upload any remaining data.
run.finish()
```
运行后在 [wandb.ai](http://wandb.ai) 即可看到自动生成的：
- Loss / Accuracy 曲线
- 系统资源监控（GPU/CPU/内存）
- 完整的超参数记录

## 离线模式
无网络环境（如内网 GPU 集群）可使用离线模式：
```python
import os
os.environ["WANDB_MODE"] = "offline"

# 正常使用 wandb.init / wandb.log
# 训练结束后同步：
# wandb sync ./wandb/offline-run-*
```

## 子页面导航
- **[[2wandb init 配置详解]]** → 所有参数含义与最佳实践
- **[[3Project-Run-Group 组织架构]]** → 大规模实验管理策略

---

*→ 下一节：[[1实验追踪（Experiment Tracking）]]*