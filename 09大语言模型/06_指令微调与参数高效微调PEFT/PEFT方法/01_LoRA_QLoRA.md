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
*   **参数量极小**: 仅需训练原模型 0.01% ~ 1% 的参数。
*   **无推理延迟**: 训练完成后，可以将 $BA$ 加回 $W_0$（重参数化），推理时结构不变。

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
