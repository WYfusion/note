# PEFT方法 (Parameter-Efficient Fine-Tuning)

本章节深入探讨***参数高效微调***方法，旨在通过调整少量参数实现大模型的高效适配，重点分析了LoRA及其变体，以及Soft Prompt类方法在语音大模型中的应用。
PEFT 方法仅更新部分权重。它们会冻结模型中的大部分层，只允许训练少量层。其他方法则完全不改变权重，而是向模型中添加新层，并仅训练这些新层。

因此，可训练权重的数量远小于原始 LLM 中的权重数量。这大大降低了训练所需的内存，以至于 PEFT 通常可以在单个 GPU 上完成。由于 LLM 的大部分保持不变，PEFT 也不太容易发生[[索引_SFT数据与格式#^31fc71|灾难性遗忘]] 。


每种参数高效微调方法都在**参数效率**、**内存效率**、**训练速率**、**模型质量**、**推理成本**上做权衡。
#### Selective  选择性：
我们选择一部分初始 LLM 参数进行微调。
选择要微调的参数子集有多种方法。我们可以决定训练：
- 仅模型中的某些组件。
- 模型的特定层。
- 单个参数类型
这些方法以及选择性方法的整体性能参差不齐。参数效率和计算效率之间存在显著的权衡，因此这些方法并不常用。
####   重新参数化
使用低秩表示法对模型权重进行重新参数化。
例如 低秩自适应（LoRA）技术。
####   添加
向模型中添加新的可训练层或参数。
通常有两种方法：
- **适配器** ——新的可训练层被添加到模型中，通常是在编码器或解码器的 FN 或注意力层之后。
- **提示调优** ——模型架构保持不变，而是通过调整输入（提示）来获得更好的性能。这可以通过向提示嵌入添加可训练参数，或者保持输入不变并重新训练嵌入权重来实现。 软提示 就是一种典型的技术。
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

### [02_PrefixTuning_PromptTuning_P_Tuning.md](02_PromptTuning_PrefixTuning_P_Tuning.md)
- **Soft Prompt系列**：
  - **Prompt Tuning**：在Embedding层前拼接可学习向量。
  - **P-Tuning v1/v2**：使用LSTM/MLP编码Prompt，解决离散性问题；深层插入（Deep Prompt Tuning）。
  - **Prefix Tuning**：在每一层Attention的Key/Value前拼接可学习前缀。
- **语音特有**：
  - **TTS风格控制**：使用Prefix Token控制说话人情感/音色（如VALL-E中的Prompt）。
  - **多任务适配**：不同Prefix对应ASR、翻译、TTS等不同任务头。

### [03_Adapter_Tuning.md](./03_Adapter_Tuning.md)
- **Adapter Tuning**：
  - 架构：在Transformer层间插入“降维-非线性-升维”的瓶颈模块。
  - 优劣：参数少，但增加推理延迟（串行计算）。
- **AdapterFusion**：
  - 机制：两阶段学习，先学单任务Adapter，再学Fusion Layer组合它们。
  - 语音应用：多语种ASR的动态切换。

### [04_IA3.md](./04_IA3.md)
- **IA3**：
  - 原理：通过可学习向量对 K, V, FFN 激活值进行缩放 (Rescaling)。
  - 特点：参数量极小 (0.01%)，模拟 In-Context Learning 机制。
  - 场景：适合大规模多任务或端侧个性化微调。
