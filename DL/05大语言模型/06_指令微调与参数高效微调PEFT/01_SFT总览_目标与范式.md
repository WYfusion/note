# SFT 总览：目标与范式

预训练（Pre-training）赋予了模型广泛的知识，而**有监督微调**（Supervised Fine-Tuning, SFT）则教会模型如何遵循指令、进行对话以及完成特定任务。

## 1. SFT 的核心目标

### 1.1 从“续写”到“对话”
*   **Base Model**: 训练目标是 Next Token Prediction。给它“中国的首都是”，它可能会续写“北京是政治中心...”。
*   **Chat Model**: 经过 SFT 后，给它“中国的首都是哪里？”，它会回答“中国的首都是北京。”
*   **对齐 (Alignment)**: SFT 是对齐的第一步，让模型的输出符合人类的意图和格式要求。

### 1.2 数学形式
给定指令数据集 $\mathcal{D} = \{(x_i, y_i)\}$，其中 $x_i$ 是指令/Prompt，$y_i$ 是期望的回答。
SFT 的目标是最大化条件似然概率：
$$ \max_{\theta} \sum_{(x, y) \in \mathcal{D}} \log P(y | x; \theta) $$
注意：Loss 只在 $y$ (Response) 部分计算，而 $x$ (Instruction) 部分通常被 Mask 掉，不参与 Loss 计算。

## 2. 语音大模型的 SFT 范式

语音大模型（Audio LLM）的微调比纯文本复杂，因为涉及多模态交互。

### 2.1 模态对齐微调 (Modality Alignment SFT)
*   **目标**: 让 LLM 理解语音信号。
*   **架构**: Speech Encoder + Projector + LLM。
*   **做法**: 冻结 Speech Encoder 和 LLM，只微调 **Projector** (如 Linear Layer 或 Q-Former)。
*   **数据**: (语音, 文本转录) 对。
*   **例子**: LLaVA-style 的语音模型训练第一阶段。

### 2.2 语音指令微调 (Speech Instruction Tuning)
*   **目标**: 让模型能执行包含语音的复杂指令。
*   **数据**:
    *   Instruction: "请总结这段录音的内容：Audio
    - Response: "这段录音讨论了量子力学的基本原理..."
**全量微调 vs PEFT**: 通常使用 LoRA 微调 LLM 部分，或者同时微调 Projector。
*   **代表模型**: Qwen-Audio, Speech-LLaMA。

### 2.3 语音生成微调 (TTS SFT)
*   **目标**: 让生成模型（如 VALL-E）适应特定说话人或风格。
*   **数据**: 特定说话人的 (文本, 音频) 对。
*   **Few-shot Tuning**: 仅用几分钟数据微调，即可实现高质量的声音克隆。

## 3. 全量微调 vs 参数高效微调 (PEFT)

### 3.1 全量微调 (Full Fine-Tuning)
*   更新模型所有参数。
*   **缺点**: 显存需求大（需要存储优化器状态），容易灾难性遗忘（Catastrophic Forgetting）。
*   **适用**: 数据量极大（如 100B Tokens）或领域跨度极大时。

### 3.2 PEFT (Parameter-Efficient Fine-Tuning)
*   只更新极少量的参数（< 1%）。
*   **优点**: 显存占用低，训练快，不易过拟合。
*   **主流方法**: LoRA, Prefix Tuning, Adapter。
*   **语音领域**: 由于语音数据通常较少，PEFT 是语音大模型微调的主流选择。

