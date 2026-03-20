# LoRA 与 QLoRA：原理与语音应用

全量微调大模型成本过高，**LoRA (Low-Rank Adaptation)** 通过低秩分解技术，实现了参数高效微调。**QLoRA** 进一步结合量化技术，极大降低了显存门槛。

## 1. LoRA (Low-Rank Adaptation)

### 1.1 核心原理
假设预训练权重为 $W_0 \in \mathbb{R}^{d \times k}$，微调时的更新量为 $\Delta W$。
LoRA 假设 $\Delta W$ 具有**低秩 (Low-Rank)** 属性，可以分解为两个小矩阵的乘积：
$$ W = W_0 + \Delta W = W_0 + BA $$
其中：
*   $B \in \mathbb{R}^{d \times r}$，初始化为 0。
*   $A \in \mathbb{R}^{r \times k}$，初始化为高斯分布。
*   $r \ll \min(d, k)$ 是秩（Rank），通常取 8, 16, 32。

### 1.2 前向传播
$$ h = W_0 x + BA x $$
*   $W_0$ 冻结不更新。
*   只训练 $A$ 和 $B$。
*   **Scaling**: 通常会乘以一个缩放因子 $\frac{\alpha}{r}$，其中 $\alpha$ 是常数。

### 1.3 优势
*   **参数量极小**: 仅需训练原模型 0.01% ~ 1% 的参数，大幅降低显存需求。
*   **无推理延迟 (Zero Inference Latency)**: 训练完成后，可以将 $BA$ 加回 $W_0$（重参数化），推理时结构不变，速度与原模型一致。
*   **高效的多任务切换 (Efficient Task Switching)**:
    *   **原理**: 由于 $W_0$ 是共享且冻结的，我们可以为不同的下游任务（如情感分析、摘要、代码生成）训练不同的 LoRA 模块（$\Delta W_{task1}, \Delta W_{task2}, ...$）。
    *   **部署**: 在生产环境中，只需要加载一份巨大的 Base Model ($W_0$) 到显存中。
    *   **动态切换**: 当请求到来时，根据任务 ID，动态地将对应的轻量级 LoRA 参数 ($A, B$) 加载并参与计算。由于 $A, B$ 非常小（几 MB 到几百 MB），这种切换几乎是瞬时的。
    *   **意义**: 这使得“**一个大模型底座 + N 个垂直领域微调**”的部署模式成为可能，极大地降低了多租户服务的硬件成本。

## 2. QLoRA (Quantized LoRA)

### 2.1 核心技术
*   **4-bit NormalFloat (NF4)**: 一种信息理论上最优的 4-bit 量化数据类型，用于存储冻结的 Base Model 权重。
*   **Double Quantization**: 对量化常数再次进行量化，进一步节省显存。
*   **Paged Optimizers**: 利用 CPU 内存来处理显存峰值，防止 OOM。

### 2.2 效果
可以在单张 24GB 显卡（如 RTX 3090/4090）上微调 33B 甚至 65B 的模型。

## 3. 语音大模型中的 LoRA 应用

语音模型通常包含 Speech Encoder 和 LLM 两部分，LoRA 可以灵活应用。

### 3.1 微调 LLM 部分 (Speech-LLaMA)
*   **场景**: 保持 Speech Encoder 不变，让 LLM 适应语音指令。
*   **做法**: 在 LLM 的 Attention 层（Query, Value）插入 LoRA Adapter。
*   **优势**: 快速赋予 LLM 听觉理解能力，且不破坏其原有的文本能力。

### 3.2 微调 Whisper (Encoder + Decoder)
*   **场景**: 适应特定领域的 ASR（如医学、法律）或特定口音。
*   **做法**:
    *   在 Encoder 的 Self-Attention 层插入 LoRA：适应新的声学环境（噪音、口音）。
    *   在 Decoder 的 Cross-Attention 层插入 LoRA：适应新的术语或语言风格。
*   **实验结论**: 仅微调 Whisper 的 Decoder 部分通常效果不如同时微调 Encoder 和 Decoder。

### 3.3 跨模态对齐 (Projector Tuning)
有时不微调 LLM，而是将 Projector (Linear/Q-Former) 视为一种“Adapter”进行全量训练，这也可以看作广义的 PEFT。

## 4. 工程实践与使用技巧 (Best Practices)

在实际使用 LoRA 进行微调时，参数的选择对效果影响巨大。

### 4.1 关键超参数设置
*   **Rank ($r$)**:
    *   **推荐值**: 通常 **8, 16, 32** 已经足够。对于非常复杂的任务（如学习全新的语言或复杂的逻辑），可以尝试 64 或 128。
    *   **注意**: $r$ 过大不仅增加显存，还可能导致过拟合，且丧失了 LoRA 的参数效率优势。
*   **Alpha ($\alpha$)**:
    *   **作用**: 缩放因子 $\frac{\alpha}{r}$ 控制了 LoRA 权重对最终输出的影响权重。
    *   **推荐值**: 通常设置为 **$r$** 或 **$2r$**。
    *   **技巧**: 在调整 $r$ 时，保持 $\alpha$ 不变，或者保持 $\alpha/r$ 的比例不变，有助于减少调参工作量。
*   **Learning Rate (LR)**:
    *   **推荐值**: LoRA 的学习率通常比全量微调要大。推荐从 **2e-4** 或 **1e-4** 开始尝试（全量微调通常是 1e-5 级别）。

### 4.2 Target Modules (微调哪些层？)
*   **仅 Q, V**: 最早期的 LoRA 论文仅微调 Attention 的 Query 和 Value 矩阵。
*   **All Linear (推荐)**: 现在的最佳实践是微调**所有的线性层**（Q, K, V, O, Gate, Up, Down）。
    *   **效果**: 实验表明，微调所有线性层通常能获得更好的效果，且参数量增加在可接受范围内。
    *   **Embedding**: 通常不需要微调 Embedding 层，除非是学习全新的 Token（如扩展词表）。

### 4.3 常见问题与解决方案
*   **Loss 不下降**: 检查学习率是否过小（LoRA 需要较大的 LR），或者 Rank 是否过小导致容量不足。
*   **灾难性遗忘**: 虽然 LoRA 比全量微调好，但仍可能发生。
    *   **解法**: 混合一部分通用数据（Replay Buffer）一起训练，或者减小 Rank 和 Alpha。
*   **合并权重后的精度损失**:
    *   在使用 4-bit / 8-bit 量化（QLoRA）训练后，如果直接合并权重进行推理，可能会有微小的精度损失。建议保持 LoRA 分离，在推理时动态计算。
