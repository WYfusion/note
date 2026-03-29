# Instruction / Task SFT 监督微调

SFT（Supervised Fine-Tuning）是用**有标注的指令-回答数据**训练模型，让它学会遵循指令、完成特定任务。当前最主流的微调方式。

---

## 核心思想

给定训练样本 $(\text{instruction}, \text{output})$，最大化模型在给定指令下生成正确输出的概率：

$$
\mathcal{L}_{\text{SFT}} = -\sum_{t} \log P(y_t | \text{instruction}, y_{<t}; \theta)
$$

**注意**：通常只对 output 部分计算 loss，instruction 部分做 loss masking。

---

## 数据格式

### 单轮指令

```json
{
  "instruction": "把以下英文翻译为中文",
  "input": "Hello, how are you?",
  "output": "你好，你怎么样？"
}
```

### 多轮对话

```json
{
  "conversations": [
    {"role": "user", "content": "什么是 LoRA？"},
    {"role": "assistant", "content": "LoRA 是一种参数高效微调方法..."}
  ]
}
```

---

## 关键配置

| 参数 | 推荐值 | 说明 |
| --- | --- | --- |
| 学习率 | 1e-5 ~ 5e-5 | 比预训练低 10-100 倍 |
| Epoch | 1-3 | 过多会过拟合 |
| Batch size | 64-256 | 可用 gradient accumulation |
| Max seq len | 2048-4096 | 根据任务调整 |
| Loss masking | 只对 output 算 loss | instruction 部分 mask 掉 |

---

## MoE 局部专家微调

对 Mixture-of-Experts 架构的模型：

- 只微调被路由激活的专家（expert）模块
- 或新增 task-specific expert，冻结其余
- 显著减少实际更新的参数量

---

## 📂 子页面（叶子层，含代码示例）

- [SFT 实战：数据准备与训练代码](SFT%20%E5%AE%9E%E6%88%98%EF%BC%9A%E6%95%B0%E6%8D%AE%E5%87%86%E5%A4%87%E4%B8%8E%E8%AE%AD%E7%BB%83%E4%BB%A3%E7%A0%81%203adc179ad44e406baf8a091cb0284686.md) — Alpaca/ChatML 格式 + Loss Masking + TRL+QLoRA 代码

**相关页面**：[文本 LLM 微调](%E6%96%87%E6%9C%AC%20LLM%20%E5%BE%AE%E8%B0%83%20c770f6676f8e449b8febb61c362bbfd3.md) · [1Full Fine-tuning 全参数微调](1Full%20Fine-tuning%20全参数微调.md) · [Continued Pretraining（CPT / DAPT / TAPT）](2Continued%20Pretraining（CPT%20DAPT%20TAPT）.md) · [微调数据构造](%E5%BE%AE%E8%B0%83%E6%95%B0%E6%8D%AE%E6%9E%84%E9%80%A0%20e51f269b452443d9bd0fcb3c901898c0.md) · [LLM 微调技术全景指南](LLM%20微调技术全景指南.md)

[SFT 实战：数据准备与训练代码](SFT%20%E5%AE%9E%E6%88%98%EF%BC%9A%E6%95%B0%E6%8D%AE%E5%87%86%E5%A4%87%E4%B8%8E%E8%AE%AD%E7%BB%83%E4%BB%A3%E7%A0%81%203adc179ad44e406baf8a091cb0284686.md)