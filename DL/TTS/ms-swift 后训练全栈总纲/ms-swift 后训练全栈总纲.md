## 0. 定位

> 一句话：本页是 **ms-swift（ModelScope SWIFT）** 笔记体系的**入口与目录**，覆盖从数据工程、训练、对齐、评测到部署的完整后训练全栈。

> [!important]
> 
> 本页只做**导航、消歧、决策树、误区清单**。具体技术原理、数学推导、代码实现请进入下方目录中的子页面。

---

## 1. 边界定义与同名消歧

**核心定位**：**ms-swift** 是 ModelScope 团队（Alibaba）开源的「大模型后训练 + 工程部署」一站式框架，AAAI 2025 收录。截至 ms-swift 4.x 版本，支持 **600+ 文本大模型** 与 **300+ 多模态大模型**，并集成 Megatron-LM 实现 MoE 训练 **10x 加速**。[[1]](https://github.com/modelscope/ms-swift)[[2]](https://swift.readthedocs.io/en/latest/GetStarted/Quick-start.html)

---

## 2. 能力地图（一图看懂全栈）

![[ms-swift 后训练全栈总纲 - 2. 能力地图（一图看懂全栈） - 图 01.excalidraw|800]]

---

## 3. 全栈分类框架（按目标维度）

|训练目标|典型方法|所属章节|
|---|---|---|
|领域知识注入|CPT → SFT|§3 训练任务|
|指令遵循|SFT-LoRA|§3 训练任务|
|风格 / 偏好对齐|DPO / SimPO / ORPO / KTO|§4 偏好学习|
|推理能力增强（数学/代码）|GRPO / DAPO / GSPO|§5 强化学习|
|多模态能力|MLLM SFT / 多模态 RLHF|§6 多模态训练|
|检索增强|Embedding + Reranker 微调|§10 Embedding/Reranker|
|低显存训练|QLoRA / FSDP-QLoRA / GaLore|§3 / §7 / §8|
|大模型训练|Megatron-SWIFT (TP/PP/CP/EP)|§7 分布式与并行|
|生产部署压缩|LoRA Merge → AWQ/GPTQ/FP8|§12 量化导出|

---

## 4. 技术选择决策树

![[ms-swift 后训练全栈总纲 - 4. 技术选择决策树 - 图 02.excalidraw|800]]

---

## 5. 常见误区速查（详见 §15）

> [!important]
> 
> 以下 6 类是踩坑率最高的「红线」，进入任何子任务前都应先确认：

1. **混淆 Base 与 Instruct 模型**：Base 模型直接做 SFT 需要更多数据冷启动，Instruct 模型继续 SFT 容易破坏对齐。

1. **chat template 错位**：训练用错 template，推理结果与 loss 都会异常。

1. **DPO 前 SFT 不够好**：DPO 是「对齐微调」，不是「能力训练」，必须先把 SFT 做扎实。

1. **GRPO 用于开放式任务**：GRPO 依赖**可验证奖励**，对开放式生成（如对话、创意写作）效果差。

1. **量化后不重测**：AWQ/GPTQ 后必须用与训练一致的 benchmark 重新评测。

1. **训练集污染评测集**：N-gram 重合检测必须做，否则评测分数完全失真。

---

## 6. 推荐学习路径

|阶段|建议章节顺序|预期产出|
|---|---|---|
|**新手起步**|§1 → §2 → §3.2 (SFT) → §3.3.1 (LoRA) → §11 (推理) → §13 (评测)|能完整跑通 SFT-LoRA → 评测 → 部署|
|**对齐进阶**|§4 (DPO/SimPO) → §5 (GRPO) → §13 (评测)|能选择并落地偏好/RL 微调|
|**大模型工程**|§7 (Megatron) → §8 (加速) → §12 (量化)|能训练 70B+ 模型并部署|
|**多模态**|§6 (MLLM) → §11 (推理)|能微调与部署 VL/Omni 模型|
|**生产化**|§14 (工程化) → §15 (避坑) → §16 (MVP)|能搭建可灰度可回滚的生产链路|

---

## 7. 目录（自动生成）

- [[#0. 定位]]
- [[#1. 边界定义与同名消歧]]
- [[#2. 能力地图（一图看懂全栈）]]
- [[#3. 全栈分类框架（按目标维度）]]
- [[#4. 技术选择决策树]]
- [[#5. 常见误区速查（详见 §15）]]
- [[#6. 推荐学习路径]]
- [[#7. 目录（自动生成）]]
- [[#参考文献]]

---

## 参考文献

1. Zhao Y. et al. _SWIFT: A Scalable lightWeight Infrastructure for Fine-Tuning_. AAAI 2025. [arXiv:2408.05517](https://arxiv.org/abs/2408.05517)

1. ms-swift GitHub: <[https://github.com/modelscope/ms-swift](https://github.com/modelscope/ms-swift)>

1. ms-swift 官方文档: <[https://swift.readthedocs.io/en/latest/](https://swift.readthedocs.io/en/latest/)>

1. Qwen × ms-swift 教程: <[https://qwen.readthedocs.io/en/latest/training/ms_swift.html](https://qwen.readthedocs.io/en/latest/training/ms_swift.html)>

[[1. 框架架构与核心对象]]

[[2. 数据工程]]

[[3. 训练任务（CPT - SFT - PEFT）]]

[[4. 偏好学习（Preference Learning）]]

[[5. 强化学习（GRPO 家族）]]

[[6. 多模态训练（MLLM）]]

[[7. 分布式与并行（Megatron-SWIFT）]]

[[8. 训练加速与显存优化]]

[[9. Agent 与工具调用训练]]

[[10. Embedding 与 Reranker 训练]]

[[11. 推理引擎与部署]]

[[12. 模型导出、合并与量化]]

[[13. 评测体系（EvalScope）]]

[[14. 工程化与生产化]]

[[15. 常见误区与避坑指南]]

[[16. 端到端最小可行路线（MVP）]]

[[17. ASR-TTS-语音多模态]]