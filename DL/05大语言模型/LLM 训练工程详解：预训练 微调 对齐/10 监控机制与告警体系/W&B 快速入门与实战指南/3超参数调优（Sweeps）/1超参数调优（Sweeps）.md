# 超参数调优（Sweeps）
W&B Sweeps 提供自动化超参数搜索，无需手动写循环，支持贝叶斯优化、提前终止、分布式并行。

---
## Sweep 是什么？
**类比**：你手动跑 10 组不同学习率的实验 → Sweep 帮你自动跑，还能智能选择下一组参数。
**三步流程**：
1. **定义搜索空间**（YAML/dict）
2. **创建 Sweep**（获得 sweep_id）
3. **启动 Agent**（自动执行多次训练）
## 最小示例
```python
import wandb

# Step 1: 定义搜索空间，这里的超参名称是和train_one_epoch中要求的一致的
sweep_config = {
    "method": "bayes",  # bayes / grid / random
    "metric": {"name": "val/loss", "goal": "minimize"}, #不能写成 `min` 或 `max`，必须是完整的 `minimize` 或 `maximize`。

	"parameters": {
	    "lr": {"min": 1e-5, "max": 1e-2, "distribution": "log_uniform_values"},  # 学习率范围
	    "batch_size": {"values": [16, 32, 64]},  # 批次大小的可选值
	    "weight_decay": {"min": 0.0, "max": 0.1},  # 权重衰减范围
	    "warmup_ratio": {"min": 0.0, "max": 0.1},  # 预热比例范围
	},
}

# Step 2: 创建 Sweep
sweep_id = wandb.sweep(sweep_config, project="llm-sft")

# Step 3: 定义训练函数
def train():
    run = wandb.init()
    config = wandb.config  # 自动注入当前搜索参数

    # 用 config.lr, config.batch_size 等进行训练
    for epoch in range(10):
        loss = train_one_epoch(config)
        wandb.log({"val/loss": loss})
    wandb.finish()

# Step 4: 启动 Agent（正式跑 20 组实验）
wandb.agent(sweep_id, function=train, count=20)
```
## 三种搜索策略

| **策略** | **原理** | **适用场景** | **效率** |
| --- | --- | --- | --- |
| `grid` | 穷举所有组合 | 参数少且离散 | 低（指数增长） |
| `random` | 随机采样参数组合 | 探索阶段、参数多 | 中 |
| `bayes` | 高斯过程建模，选最有希望的参数 | 精细调参、预算有限 | 高（推荐） |
## 参数分布类型
```python
"parameters": {
    # 连续均匀分布
    "dropout": {"min": 0.0, "max": 0.5, "distribution": "uniform"},
    # 对数均匀（适合学习率）
    "lr": {"min": 1e-5, "max": 1e-2, "distribution": "log_uniform_values"},
    # 离散选择
    "optimizer": {"values": ["adam", "adamw", "sgd"]},
    # 整数
    "num_layers": {"min": 2, "max": 12, "distribution": "int_uniform"},
    # 固定值（不搜索）
    "seed": {"value": 42},
}
```
## 提前终止（Early Termination）
避免浪费 GPU 在明显差的参数组合上：
```python
sweep_config["early_terminate"] = {
    "type": "hyperband",
    "min_iter": 3,    # 至少跑 3 个 epoch 再判断
    "eta": 3,         # 淘汰比例
    "s": 2,           # bracket 数
}
```
## 命令行方式（推荐用于集群）
```bash
# sweep.yaml 中定义搜索空间
wandb sweep sweep.yaml  # 返回 sweep_id
wandb agent <entity>/<project>/<sweep_id>  # 多台机器分别执行
```
## 子页面导航
- **[[2Sweep 配置与搜索策略详解]]** → 分布类型全解、嵌套参数、条件参数
- **[[3Early Termination 与分布式 Sweep]]** → Hyperband/Median 策略、多机并行

---

*← 上一节：[[1实验追踪（Experiment Tracking）]]　|　下一节：[[1数据与模型版本管理（Artifacts）]] →*
