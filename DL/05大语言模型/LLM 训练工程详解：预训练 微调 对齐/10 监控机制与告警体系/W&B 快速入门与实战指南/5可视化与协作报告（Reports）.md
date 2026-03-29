# 可视化与协作报告（Reports）

W&B Reports 和 Dashboard 让你从数据面板到交互式文档，实现实验结果的可视化与团队协作。

---

## Dashboard vs Report

| **特性** | **Dashboard** | **Report** |
| --- | --- | --- |
| 用途 | 实时监控训练过程 | 总结、分享、存档实验结论 |
| 内容 | 自动生成的指标面板 | Markdown + 嵌入式图表 + Run 集 |
| 交互 | 拖拽面板、过滤 Run | 评论、@提及、版本历史 |
| 类比 | Grafana Dashboard | Notion 页面 + 数据可视化 |

## Dashboard 定制

### 面板类型

- **Line Plot**：默认，指标随 step 变化
- **Scatter Plot**：超参 vs 指标相关性
- **Bar Chart**：Run 之间指标对比
- **Parallel Coordinates**：多维超参同时对比
- **Table**：Run 列表，可排序/过滤

### 常用操作

1. **分组面板**：用 `train/`, `val/`, `system/` 前缀自动分组
2. **正则过滤**：面板搜索栏支持正则表达式
3. **Run 对比**：选中多个 Run，切换到 "Diff" 视图查看参数差异
4. **分组聚合**：按 Group / Tag 聚合显示均值 ± 标准差

## Report 创建

### 通过 UI

1. 进入 Project → 点击 "Create Report"
2. 添加 Run Set（选择要对比的实验）
3. 嵌入面板 + 写 Markdown 分析
4. 分享链接或导出 PDF

### 通过 API（自动化报告）

```python
import wandb

api = wandb.Api()
runs = api.runs("my-team/llm-sft", filters={"tags": "final"})

# 构建对比表
for run in runs:
    print(f"{run.name}: loss={run.summary['val/loss']:.4f}, "
          f"lr={run.config['lr']}, method={run.config.get('method', 'N/A')}")
```

## W&B API 读取历史数据

```python
api = wandb.Api()
run = api.run("my-team/llm-sft/run-id")

# 获取完整历史
history = run.history()  # pandas DataFrame
print(history[["_step", "train/loss", "val/loss"]].tail())

# 获取 Summary
print(run.summary["val/loss"])

# 获取 Config
print(run.config["lr"])
```

## 实用技巧

<aside>
💡

**最佳实践**：每次重要实验结束后创建 Report，记录：结论 → 关键图表 → 超参配置 → 下一步计划。这比截图+笔记靠谱得多。

</aside>

---

*← 上一节：[[1数据与模型版本管理（Artifacts）]]　|　下一节：[[LLM 训练/微调/对齐实战集成]] →*