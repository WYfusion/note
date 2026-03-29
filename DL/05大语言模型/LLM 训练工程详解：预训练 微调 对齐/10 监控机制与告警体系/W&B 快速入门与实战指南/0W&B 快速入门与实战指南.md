# W&B 快速入门与实战指南

从零开始掌握 Weights & Biases，覆盖实验追踪、调参、版本管理到 LLM 训练全流程。

---

## 什么是 W&B？

Weights & Biases（简称 wandb）是一个 **MLOps 平台**，核心解决机器学习实验中的"记录、对比、复现、协作"问题。可以类比为：

- **TensorBoard 的云端增强版** —— 更强的可视化 + 团队协作
- **Git for ML experiments** —— 自动记录每次实验的代码、超参、指标、环境

## W&B 核心能力一览

| **能力模块** | **解决的问题** | **对应概念** |
| --- | --- | --- |
| Experiment Tracking | 记录每次训练的指标、超参、系统资源 | `wandb.init`  • `wandb.log` |
| Sweeps | 自动化超参数搜索与调优 | `wandb.sweep`  • `wandb.agent` |
| Artifacts | 数据集/模型版本管理与血缘追踪 | `wandb.Artifact` |
| Reports | 交互式可视化报告与团队分享 | Dashboard + Report |
| Alerts | 训练异常自动告警（Slack/Email） | `wandb.alert` |

## 为什么选 W&B？

- **零侵入**：只需 3 行代码即可接入现有训练脚本
- **自动捕获**：GPU/CPU/内存等系统指标自动采集
- **框架无关**：PyTorch、HuggingFace、DeepSpeed、Megatron 均原生支持
- **团队协作**：多人对比实验、共享 Dashboard、评审 Report
- **LLM 友好**：对 SFT / DPO / RLHF / PPO 等对齐流程有成熟集成方案

## 学习路线图

> 按以下顺序阅读子页面，由浅入深：
1. **[[1W&B 核心概念与环境搭建]]** → 安装、登录、Project/Run/Group 等基本概念
2. **[[1实验追踪（Experiment Tracking）]]** → `wandb.log` 指标记录、Media 日志、系统监控
3. **[[1超参数调优（Sweeps）]]** → 自动搜索策略、Early Stopping、分布式 Sweep
4. **[[1数据与模型版本管理（Artifacts）]]** → 数据集版本、模型血缘、Model Registry
5. **[[5可视化与协作报告（Reports）]]** → Dashboard 定制、交互式 Report、团队分享
6. **[[LLM 训练/微调/对齐实战集成]]** → HuggingFace Trainer、SFT/DPO/RLHF、分布式训练