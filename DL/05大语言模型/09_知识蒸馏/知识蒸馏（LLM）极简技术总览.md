## 0. 总纲

**知识蒸馏（Knowledge Distillation, KD）**：把教师模型（Teacher）的能力迁移给学生模型（Student），以换取更低参数量、更低延迟、更低显存/算力成本。

> [!info] 核心思想

> 教师模型的输出（logits、hidden states、文本）中包含比硬标签更丰富的"暗知识"（dark knowledge），学生通过拟合这些软信号来获得超越直接训练的效果。

### 本质监督信号 4 类

|类别|监督来源|典型方法|教师可见性|
|---|---|---|---|
|**分布蒸馏**|teacher logits / token probabilities|Hinton KD, MiniLLM, GKD|白盒|
|**表征蒸馏**|hidden states / attention maps / layer relations|TinyBERT, MiniLM|白盒|
|**结果蒸馏**|teacher response / instruction-response data|Alpaca, Self-Instruct, SeqKD|黑盒/白盒|
|**过程与偏好蒸馏**|rationale / preference / reward / judge signal|DPO, RLAIF, Judge Distillation|黑盒/白盒|

### 教师可见性分类

- **白盒蒸馏（White-box）**：可拿到 logits / hidden states / attention → 信号最丰富，效果上限最高

- **黑盒蒸馏（Black-box）**：只能拿 teacher 文本输出 → 工程最灵活，API 教师即可

- **自蒸馏（Self-Distillation）**：teacher 与 student 同源，或 teacher 为同模型强配置/强采样版本

---

## 1. 三阶段蒸馏框架总览

蒸馏按训练阶段可分为三大类，每类蒸的"东西"不同：

|阶段|蒸的是什么|核心目标|详见|
|---|---|---|---|
|**预训练蒸馏**|语言表征|基础模型压缩，保留通用能力|[[1. 预训练阶段蒸馏]]|
|**微调蒸馏（SFT）**|任务行为|指令跟随、垂直任务能力迁移|[[2. 微调阶段蒸馏]]|
|**对齐蒸馏**|偏好与边界|安全、价值观、风格一致性|[[3. 对齐阶段蒸馏]]|

---

## 2. 方案选择速查

|场景|推荐路线|
|---|---|
|只有闭源 API 教师|黑盒数据蒸馏 → SFT + DPO/SimPO|
|有开源白盒教师|logits KD + hidden KD + response KD + on-policy KD|
|压缩基础模型|预训练蒸馏优先|
|快速做垂直小模型|SFT 蒸馏 + 对齐蒸馏|
|做推理小模型|rationale distillation + verifier + hard-example + on-policy|

---

## 3. 一句话结论

当前最实用的落地范式不是单一 KD，而是：

> [!important] 推荐默认范式

> **Pretrain KD（可选）→ SFT 数据蒸馏（核心）→ Preference/Judge Distillation（关键）→ 失败样本迭代回流（增强）**

详见 → [[4. 三阶段综合流程与方案选择速查]]

---

## 子页面导航（完整树状索引）

- [[1. 黑盒数据蒸馏（Self-Instruct / Alpaca-style）]]
    - [[1. Self-Instruct Pipeline 实现与数据过滤]] — 完整 pipeline + 质量控制


- [[2. Response / Rationale / CoT / Step-by-Step Distillation]]

    - [[1. CoT 蒸馏数据构造与训练实现]] — Self-Consistency + 双任务训练


- [[3. 白盒 SFT KD 与 On-policy Distillation（GKD）]]

    - [[1. GKD 原理推导与实现]] — Exposure Bias 解决 + 代码


- [[4. Self-Distillation 技术]]

    - [[1. Best-of-N / Self-Refine 实现]] — 两种自蒸馏范式对比


[[3. 对齐阶段蒸馏（Alignment Distillation）]]

- [[1. 经典预训练蒸馏方案（Hinton KD / DistilBERT / TinyBERT / MiniLM）]]

    - [[1. Soft Targets 与 Temperature Scaling 数学推导]] — 公式推导 + Python 实现

    - [[2. DistilBERT / TinyBERT / MiniLM 架构与实现]] — 三大方案代码对比


- [[2. LLM 预训练蒸馏新范式（PD）]]

    - [[1. Forward KL vs Reverse KL vs Mixed KL 详解]] — 梯度分析 + 场景选择

    - [[2. Logits 裁剪与 Top-k/Top-p 蒸馏实现]] — 存储优化 + 代码


[[2. 微调阶段蒸馏（SFT Distillation）]]

- [[1. RLHF 与 RLAIF / Constitutional AI]]

    - [[1. PPO 训练与 Reward Model 实现]] — Bradley-Terry + PPO 代码

    - [[2. Constitutional AI Pipeline 实现]] — 两阶段自动对齐


- [[2. DPO / ORPO / SimPO / KTO 偏好优化族]]

    - [[1. DPO 数学推导与 Python 实现]] — 从 RLHF 到 DPO 完整推导

    - [[2. ORPO / SimPO / KTO 对比实现]] — 四种偏好方法对比表


- [[3. Judge Distillation]]

    - [[1. LLM-as-Judge Pipeline 实现]] — Pairwise + Pointwise + 偏差消除


[[4. 三阶段综合流程与方案选择速查]]

[[1. 预训练阶段蒸馏（Pre-training Distillation）]]

- [[1. 端到端蒸馏 Pipeline 设计]]

    - [[1. 完整蒸馏工程实战]] — 四阶段实战 + 检查清单


- [[2. 评测闭环与迭代回流]]

    - [[1. 蒸馏质量评测与自动化实现]] — 评测框架 + 效率 benchmark
