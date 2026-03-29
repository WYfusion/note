# LLM 微调技术全景指南
本指南系统性地梳理了当前 LLM（含多模态、语音、向量、重排模型）微调技术的核心概念、经典方案、最新流行方法以及综合实践流程。
>**目标**：保留底座能力 + 注入任务/领域能力
>**本质**：更新全部参数 / 部分参数 / 附加参数 / 连续预训练

---

## 微调技术主线一览

| 主线 | 核心思路 | 典型场景 |
| --- | --- | --- |
| Full Fine-tuning | 更新模型全部参数 | 资源充足、追求极限性能 |
| PEFT（参数高效微调） | 只训练少量新增/选定参数 | 资源受限、快速迭代 |
| Continued Pretraining | 用领域语料继续预训练 | 垂域适配、知识注入 |
| Instruction / Task SFT | 用指令数据监督微调 | 对话、问答、指令遵循 |
| Multimodal Adapter Tuning | 训练模态连接器/适配器 | 图文/视频文多模态 LLM |
| Speech/Audio Joint Tuning | 音频编码器 + LLM 联合微调 | 语音理解/生成 |
| Embedding/Reranker Tuning | 对比学习/排序损失微调 | 检索、重排 |

---
## 📚 分层导航
### 一、基础概念与分类
- [微调核心概念与分类](微调核心概念与分类.md)
### 二、PEFT 参数高效微调方案族
- [PEFT 参数高效微调方案族](PEFT%20%E5%8F%82%E6%95%B0%E9%AB%98%E6%95%88%E5%BE%AE%E8%B0%83%E6%96%B9%E6%A1%88%E6%97%8F%2007bcd7a7aa894f4984c232d57a0e7376.md)
    - [LoRA 系列详解](LoRA%20系列详解.md) · [Prompt 与 Prefix Tuning 系列](Prompt%20与%20Prefix%20Tuning%20系列.md) · [Adapter 系列与其他 PEFT 方法](Adapter%20系列与其他%20PEFT%20方法.md)
### 三、按模型类型分类
- [文本 LLM 微调](%E6%96%87%E6%9C%AC%20LLM%20%E5%BE%AE%E8%B0%83%20c770f6676f8e449b8febb61c362bbfd3.md)
    - [1Full Fine-tuning 全参数微调](1Full%20Fine-tuning%20全参数微调.md) · [Continued Pretraining（CPT / DAPT / TAPT）](2Continued%20Pretraining（CPT%20DAPT%20TAPT）.md) · [Instruction / Task SFT 监督微调](3Instruction%20Task%20SFT%20监督微调.md)
- [多模态 LLM 微调](多模态%20LLM%20微调.md)
    - [多模态架构与连接器详解](%E5%A4%9A%E6%A8%A1%E6%80%81%E6%9E%B6%E6%9E%84%E4%B8%8E%E8%BF%9E%E6%8E%A5%E5%99%A8%E8%AF%A6%E8%A7%A3%2024df6bd2dc7e417f939708b5ed7b5f21.md) · [多模态分阶段训练与视频微调](%E5%A4%9A%E6%A8%A1%E6%80%81%E5%88%86%E9%98%B6%E6%AE%B5%E8%AE%AD%E7%BB%83%E4%B8%8E%E8%A7%86%E9%A2%91%E5%BE%AE%E8%B0%83%2028b0dbec49b34e4c8ddf9246d80be519.md)
- [语音与音频 LLM 微调](语音与音频%20LLM%20微调.md)
    - [语音理解：连续编码器 + Projector + LLM](%E8%AF%AD%E9%9F%B3%E7%90%86%E8%A7%A3%EF%BC%9A%E8%BF%9E%E7%BB%AD%E7%BC%96%E7%A0%81%E5%99%A8%20+%20Projector%20+%20LLM%20ac7418f52ecb48c88b1f9b1ebe5bc031.md) · [Codec Token LM 与语音生成微调](Codec%20Token%20LM%20%E4%B8%8E%E8%AF%AD%E9%9F%B3%E7%94%9F%E6%88%90%E5%BE%AE%E8%B0%83%2093e5c8fda2b64f709dc7aca73235caab.md)
- [向量模型微调](向量模型微调.md)
    - [对比学习与损失函数详解](对比学习与损失函数详解.md) · [多阶段向量微调与进阶技术](多阶段向量微调与进阶技术.md)
- [重排模型微调](%E9%87%8D%E6%8E%92%E6%A8%A1%E5%9E%8B%E5%BE%AE%E8%B0%83%206d652fa8de814c838611ed64426f8b7f.md)
    - [Cross-Encoder 微调与蒸馏详解](Cross-Encoder%20%E5%BE%AE%E8%B0%83%E4%B8%8E%E8%92%B8%E9%A6%8F%E8%AF%A6%E8%A7%A3%207246d864745442f8947d15520bb4b3f9.md)
### 四、微调数据构造
- [微调数据构造](微调数据构造.md)
    - [Hard Negative Mining 与合成数据](Hard%20Negative%20Mining%20%E4%B8%8E%E5%90%88%E6%88%90%E6%95%B0%E6%8D%AE%20a08b2ca52f474463b8490ef6e0efdb0f.md)
### 五、训练范式与多阶段流程
- [7训练范式与多阶段流程](7训练范式与多阶段流程.md)
### 六、工程实践：选型、优化与部署
- [8工程实践：选型、资源优化与部署](8工程实践：选型、资源优化与部署.md)
    - [资源优化与量化部署详解](%E8%B5%84%E6%BA%90%E4%BC%98%E5%8C%96%E4%B8%8E%E9%87%8F%E5%8C%96%E9%83%A8%E7%BD%B2%E8%AF%A6%E8%A7%A3%20ebdbd07e69854b00a56903b10f5df11f.md)

---
## 🗺️ 通用综合流程速览
1. **明确任务** — 生成 / 分类 / 抽取 / 检索 / 重排 / 多模态问答 / 语音理解
2. **选底座** — 文本 LLM / MLLM / SpeechLM / Embedding Model
3. **选微调级别** — full / LoRA / QLoRA / adapter / projector-only
4. **构造数据** — 清洗 → 去重 → 格式统一 → hard negatives / 多模态配对
5. **训练策略** — continued pretraining（可选）→ SFT / task tuning
6. **资源优化** — bf16/fp16/fp8/4bit + grad checkpoint + flash attn + FSDP/ZeRO
7. **验证** — 通用能力回归 + 任务集评测 + 长度/幻觉/格式检查
8. **导出** — adapter / merged model / quantized deploy
9. **上线** — A/B 测试 + 失败样本回流
10. **迭代** — 补数据 → 再微调 → 蒸馏/压缩

---
## ⚡ 选型速记

| 场景 | 推荐方案 |
| --- | --- |
| 数据少、卡少 | QLoRA |
| 想最稳 | LoRA / rsLoRA |
| 领域语言建模 | Continued Pretraining + SFT |
| 多模态 | Projector + LoRA on LLM |
| 语音 | Audio Encoder/Projector Tuning + LoRA |
| 检索向量 | Contrastive Finetune + Hard Negatives |
| 重排 | Cross-Encoder Finetune |
| 超大模型 + 预算足 | Partial / Full Fine-tuning |
