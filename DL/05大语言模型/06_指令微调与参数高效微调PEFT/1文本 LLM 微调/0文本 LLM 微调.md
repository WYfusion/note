# 文本 LLM 微调
文本 LLM 微调是最经典、应用最广的微调场景。核心流程为：**连续预训练（可选）→ 监督微调（SFT）→ 蒸馏/压缩（可选）**。

---
## 典型流程

```mermaid
graph LR
    A["预训练底座"] --> B["CPT / DAPT / TAPT"]
    B --> C["Instruction / Task SFT"]
    C --> D["蒸馏 / 压缩"]
    C --> E["部署"]
    D --> E
    style B fill:#e8f4fd
    style C fill:#d4edda
    style D fill:#fff3cd
```

---
## 核心方法概览
### Full Fine-tuning（全参数微调）
更新模型所有参数。效果上限最高，但对显存和算力要求极高（7B fp16 ≈ 56GB，加上优化器状态约 ×4）。适合有充足资源的场景。
### Continued Pretraining（连续预训练）
用目标领域的**无标注语料**以语言建模目标继续训练，让模型内化领域词汇和知识。
- **CPT**：通用连续预训练
- **DAPT**（Domain-Adaptive）：领域级，如医疗、法律、金融
- **TAPT**（Task-Adaptive）：任务级，用任务相关未标注数据

### Instruction / Task SFT（监督微调）
用 `(指令/问题, 期望输出)` 配对数据训练，让模型学会遵循指令和完成特定任务。当前最主流的微调方式。
### PEFT 系列（LoRA / QLoRA 等）
参数高效微调，只更新极少量参数。**QLoRA**（4-bit 量化 + LoRA）是资源受限下的首选。
### MoE 局部专家微调
对 Mixture-of-Experts 模型，只微调被激活的专家模块或新增专家，避免更新全部参数。
### Long-context Continued Training
通过修改位置编码（如 RoPE 频率调整）+ 长文本语料继续训练，扩展模型上下文窗口。

---

## 当前最稳默认组合

✅**通用场景**：Continued Pretraining（可选）+ QLoRA SFT
**企业/垂域**：DAPT/TAPT + LoRA/QLoRA

---
## 📂 子页面导航
- [1Full Fine-tuning 全参数微调](1Full%20Fine-tuning%20全参数微调.md)
- [Continued Pretraining（CPT / DAPT / TAPT）](2Continued%20Pretraining（CPT%20DAPT%20TAPT）.md)
- [Instruction / Task SFT 监督微调](3Instruction%20Task%20SFT%20监督微调.md)
**相关页面**：[PEFT 参数高效微调方案族](PEFT%20%E5%8F%82%E6%95%B0%E9%AB%98%E6%95%88%E5%BE%AE%E8%B0%83%E6%96%B9%E6%A1%88%E6%97%8F%2007bcd7a7aa894f4984c232d57a0e7376.md) · [7训练范式与多阶段流程](7训练范式与多阶段流程.md) · [LLM 微调技术全景指南](LLM%20微调技术全景指南.md)
[Continued Pretraining（CPT / DAPT / TAPT）](2Continued%20Pretraining（CPT%20DAPT%20TAPT）.md)
[1Full Fine-tuning 全参数微调](1Full%20Fine-tuning%20全参数微调.md)
[Instruction / Task SFT 监督微调](3Instruction%20Task%20SFT%20监督微调.md)