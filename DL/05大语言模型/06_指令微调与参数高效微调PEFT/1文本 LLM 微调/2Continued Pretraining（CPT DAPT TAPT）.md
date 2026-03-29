# Continued Pretraining（CPT / DAPT / TAPT）

连续预训练（Continued Pretraining）是在预训练模型基础上，用**领域无标注语料**继续做语言建模训练，注入领域知识。

---

## 三种粒度

| 方法 | 全称 | 语料范围 | 示例 |
| --- | --- | --- | --- |
| **CPT** | Continued Pre-Training | 通用或混合语料 | 通用网页 + 领域文档混合 |
| **DAPT** | Domain-Adaptive Pre-Training | 特定领域的大量文本 | 所有 PubMed 论文（医疗） |
| **TAPT** | Task-Adaptive Pre-Training | 与下游任务直接相关的未标注文本 | 与任务输入格式相似的文本 |

---

## 核心原理

用**语言建模目标**（next token prediction）在领域语料上继续训练：

$$
\mathcal{L}_{\text{CPT}} = -\sum_{t} \log P(x_t | x_{<t}; \theta)
$$

**不需要标注数据**，只需要领域纯文本。

---

## 为什么重要

- 预训练模型可能对特定领域的术语、知识了解不足
- 直接 SFT 只能教模型"怎么回答"，不能教它"知道什么"
- CPT 先注入知识，再 SFT 注入能力，效果显著优于只做 SFT

---

## 关键工程细节

- **学习率**：比预训练低 2-10 倍（通常 5e-5 ~ 2e-4）
- **数据混合**：通常混合一定比例的通用数据，防止灾难性遗忘
- **训练量**：领域语料 1-3 个 epoch
- 通常使用 **LoRA / QLoRA** 做 CPT 以节省资源
- 完成 CPT 后再接 SFT

---

## Long-context Continued Training

一种特殊的 CPT，目标是扩展模型的上下文窗口长度：

- 修改位置编码（如 **RoPE 频率缩放**）
- 用长文档语料继续训练
- 典型案例：将 4K 上下文扩展到 32K / 128K

