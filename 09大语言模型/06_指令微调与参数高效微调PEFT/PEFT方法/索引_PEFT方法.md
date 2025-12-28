# PEFT方法 (Parameter-Efficient Fine-Tuning)

本章节深入探讨参数高效微调方法，旨在通过调整少量参数实现大模型的高效适配，重点分析了LoRA及其变体，以及Soft Prompt类方法在语音大模型中的应用。

## 目录

### [01_LoRA_QLoRA.md](./01_LoRA_QLoRA.md)
- **LoRA (Low-Rank Adaptation)**：
  - 原理：$W = W_0 + BA$，冻结预训练权重，训练低秩矩阵。
  - 优势：显存占用低，无推理延迟（可合并权重）。
- **QLoRA**：
  - 结合4-bit NormalFloat (NF4) 量化与LoRA。
  - 双重量化（Double Quantization）与分页优化器（Paged Optimizers）。
- **语音应用**：
  - Whisper微调：仅微调Attention层适应新口音/语种。
  - 多模态投影层（Projector）的高效微调。

### [02_PrefixTuning_PromptTuning_P_Tuning.md](./02_PrefixTuning_PromptTuning_P_Tuning.md)
- **Soft Prompt系列**：
  - **Prompt Tuning**：在Embedding层前拼接可学习向量。
  - **P-Tuning v1/v2**：使用LSTM/MLP编码Prompt，解决离散性问题；深层插入（Deep Prompt Tuning）。
  - **Prefix Tuning**：在每一层Attention的Key/Value前拼接可学习前缀。
- **语音特有**：
  - **TTS风格控制**：使用Prefix Token控制说话人情感/音色（如VALL-E中的Prompt）。
  - **多任务适配**：不同Prefix对应ASR、翻译、TTS等不同任务头。
