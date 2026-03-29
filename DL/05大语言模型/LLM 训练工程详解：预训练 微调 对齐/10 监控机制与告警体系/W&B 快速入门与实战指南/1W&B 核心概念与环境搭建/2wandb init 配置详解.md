# wandb.init 配置详解

`wandb.init()` 是每次实验的入口，所有参数决定了 Run 的元数据、行为和存储方式。

---
## 完整参数签名
```python
wandb.init(
    project: str = None,       # 项目名
    entity: str = None,        # 团队/用户命名空间
    name: str = None,          # Run 显示名称
    id: str = None,            # Run 唯一标识（用于恢复）
    resume: str = None,        # 恢复策略
    group: str = None,         # 分组名
    job_type: str = None,      # 任务类型
    tags: list = None,         # 标签列表
    notes: str = None,         # 备注
    config: dict = None,       # 超参数配置
    dir: str = None,           # 本地日志目录
    mode: str = None,          # 运行模式
    save_code: bool = None,    # 是否保存代码
    reinit: bool = None,       # 是否允许重复初始化
    settings: dict = None,     # 高级设置
)
```
## 关键参数详解
### project 与 entity
```python
# 个人项目
wandb.init(project="llm-sft")

# 团队项目
wandb.init(project="llm-sft", entity="my-team")

# 通过环境变量设置（CI/CD 推荐）
os.environ["WANDB_PROJECT"] = "llm-sft"
os.environ["WANDB_ENTITY"] = "my-team"
```
### name 与 id
```python
# name: 人类可读的显示名称（可重复）
wandb.init(name="llama3-8b-lora-r16-lr2e4")

# id: 机器标识（全局唯一，用于恢复训练）
wandb.init(id="abc123", resume="allow")
```

💡**命名规范建议**：`{模型}-{方法}-{关键超参}`，如 `llama3-8b-lora-r16-lr2e4`。避免用日期或随机字符串——W&B 已自动记录时间。
### resume（断点恢复）

| **值** | **行为** | **场景** |
| --- | --- | --- |
| `"allow"` | 如果 id 对应的 Run 存在则恢复，否则创建新 Run | **推荐**，容错性最好 |
| `"must"` | 必须恢复已有 Run，找不到则报错 | 明确恢复训练 |
| `"never"` | 始终创建新 Run，id 冲突则报错 | 确保不覆盖 |
| `"auto"` | W&B 自动判断（检测 checkpoint） | 简单场景 |
| `None`（默认） | 始终创建新 Run | 一般训练 |

**断点恢复完整示例**：
```python
import wandb
import os

# 生成确定性 id（基于实验配置）
run_id = f"sft-llama3-lr2e4-bs32"

run = wandb.init(
    project="llm-sft",
    id=run_id,
    resume="allow",
    config={"lr": 2e-4, "batch_size": 32},
)

# 检查是否为恢复的 Run
if run.resumed:
    print(f"恢复训练，从 step {run.step} 继续")
    # 加载 checkpoint...
else:
    print("开始新训练")
```

### config（超参数）

```python
# 方式 1: 字典传入
wandb.init(config={"lr": 2e-4, "epochs": 3})

# 方式 2: argparse 对象
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=2e-4)
args = parser.parse_args()
wandb.init(config=args)

# 方式 3: OmegaConf / Hydra 配置
from omegaconf import OmegaConf
cfg = OmegaConf.load("config.yaml")
wandb.init(config=OmegaConf.to_container(cfg, resolve=True))

# 方式 4: 后续更新
wandb.config.update({"extra_param": "value"}, allow_val_change=True)
```

### mode（运行模式）

```python
# 正常模式（默认）
wandb.init(mode="online")

# 离线模式（内网/无网络）
wandb.init(mode="offline")
# 后续同步: wandb sync ./wandb/offline-run-*

# 禁用模式（调试时跳过所有 W&B 调用）
wandb.init(mode="disabled")

# 空运行（验证代码但不上传）
wandb.init(mode="dryrun")
```

### save_code

```python
# 自动保存当前脚本到 W&B
wandb.init(save_code=True)

# 保存额外文件
wandb.save("train.py")
wandb.save("configs/*.yaml")
```

## 环境变量速查

| **环境变量** | **等价参数** | **说明** |
| --- | --- | --- |
| `WANDB_API_KEY` | — | API 密钥（CI/CD 必须） |
| `WANDB_PROJECT` | `project` | 项目名 |
| `WANDB_ENTITY` | `entity` | 团队/用户 |
| `WANDB_MODE` | `mode` | `online` / `offline` / `disabled` |
| `WANDB_RUN_GROUP` | `group` | 分组名 |
| `WANDB_TAGS` | `tags` | 逗号分隔的标签 |
| `WANDB_NAME` | `name` | Run 名称 |
| `WANDB_DIR` | `dir` | 本地日志路径 |
| `WANDB_SILENT` | — | `true` 关闭所有 W&B 输出 |

## 生产环境最佳实践模板

```python
import wandb
import os
import hashlib

def init_wandb(config: dict):
    """生产级 W&B 初始化模板。"""
    # 基于配置生成确定性 id
    config_str = str(sorted(config.items()))
    run_id = hashlib.md5(config_str.encode()).hexdigest()[:8]

    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "default"),
        entity=os.getenv("WANDB_ENTITY"),
        name=f"{config.get('model', 'model')}-{config.get('method', 'ft')}",
        id=run_id,
        resume="allow",
        group=config.get("group"),
        job_type=config.get("job_type", "train"),
        tags=config.get("tags", []),
        config=config,
        save_code=True,
    )

    # 定义指标 X 轴
    wandb.define_metric("epoch")
    wandb.define_metric("val/*", step_metric="epoch")

    return run
```

---

*← 返回：[[W&B 核心概念与环境搭建]]*
