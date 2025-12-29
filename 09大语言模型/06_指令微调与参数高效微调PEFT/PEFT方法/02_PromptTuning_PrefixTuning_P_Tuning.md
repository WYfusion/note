#  Prompt Tuning,Prefix Tuning 与 P-Tuning

除了修改模型权重（如 LoRA），另一种思路是**修改输入**，即在输入端添加可学习的“软提示”（Soft Prompts）。
同样，可以按照不同的任务类型设计不同的**修改输入**，使其呈现不同的输出风格。
## 0. 核心辨析：Prompt Engineering vs Prompt Tuning

在深入了解 Prompt Tuning 之前，必须区分它与广为人知的 **Prompt Engineering (提示词工程)** 的本质区别。

| 维度        | Prompt Engineering (提示词工程)         | Prompt Tuning (提示词微调 / Soft Prompt)    |
| :-------- | :--------------------------------- | :------------------------------------- |
| **本质**    | **离散 (Discrete)** 的自然语言文本          | **连续 (Continuous)** 的向量 (Embedding)    |
| **人类可读性** | **可读** (如 "Translate to English:") | **不可读** (一堆浮点数，无法对应具体单词)               |
| **优化方式**  | 人工试错、经验总结、强化学习搜索                   | **反向传播 (Backpropagation)**、梯度下降        |
| **参数更新**  | **无** (模型参数和输入参数都不变)               | **有** (更新 Prompt 向量，冻结模型参数)            |
| **搜索空间**  | 受限于词表和人类语言习惯                       | 在连续的高维向量空间中搜索，解空间更大                    |
| **典型代表**  | CoT, Few-shot, ReAct               | Prompt Tuning, Prefix Tuning, P-Tuning |

*   **Prompt Engineering (Hard Prompt)**: 就像是在教一个孩子，你试图用不同的**话术**（"请仔细思考"、"一步步来"）来引导他输出正确答案。你受限于语言的表达能力。
*   **Prompt Tuning (Soft Prompt)**: 就像是直接通过脑机接口刺激大脑的某个区域。你不需要说具体的词，而是直接寻找能激活模型特定能力的**神经元信号模式**（向量）。这种模式可能不对应任何人类语言，但对模型来说是最优的指令。

---

## 1. Prompt Tuning

### 1.1 原理
在输入的 Embedding 层前面拼接一组可学习的向量 $P \in \mathbb{R}^{l \times e}$。
$$ Input = [P; E(x)] $$
*   只训练 $P$，冻结整个模型。
*   随着模型规模增大（>10B），Prompt Tuning 的效果逐渐逼近全量微调。

## 2. Prefix Tuning

### 2.1 原理
不仅仅在 Embedding 层加 Prompt，而是在**每一层 Transformer** 的 Key 和 Value 之前都拼接可学习的 Prefix 向量。
$$ K = [P_K; K_{original}], \quad V = [P_V; V_{original}] $$
*   **表达能力更强**: 相当于修改了每一层的激活值。
*   **参数量**: 比 Prompt Tuning 多，但仍远少于全量微调。

## 3. P-Tuning v1 / v2

### 3.1 P-Tuning v1
*   发现直接训练 **Soft Prompt 很难收敛**（因为 Embedding 离散且相关性强）。
*   引入一个 LSTM 或 MLP 来生成 Soft Prompt，训练时优化这个生成网络。

### 3.2 P-Tuning v2
*   类似于 Prefix Tuning，将 **Prompt 加到每一层**（Deep Prompt Tuning）。
*   移除了 LSTM，直接优化 Prompt 向量，但在小模型上效果显著提升。

## 4. 语音大模型中的应用

Soft Prompt 在语音领域有独特的应用场景，特别是**控制生成风格**。

### 4.1 风格控制 (Style Control in TTS)
在 VALL-E 或其他 TTS 模型中，可以使用 Soft Prompt 来代表特定的情感或说话风格。
*   **做法**: 训练一组 Prefix 向量，分别对应“开心”、“悲伤”、“愤怒”。
*   **推理**: 输入文本时，拼接对应的 Emotion Prefix，即可生成带有该情感的语音。

### 4.2 模态适配 (Modality Adaptation)
可以将 Speech Encoder 的输出视为一种“Soft Prompt”。
*   **Speech-LLM**: 语音特征经过 Projector 映射后，本质上就是一串连续的向量序列，直接拼接到 LLM 的 Embedding 空间。
*   **观点**: 从这个角度看，多模态微调（Multimodal Tuning）其实就是一种由 Encoder 生成 Prompt 的 Prefix Tuning。

### 4.3 跨语言 ASR
对于 Whisper，可以为每种语言训练一个特定的 Prefix。
*   **Code-Switching**: 在处理中英混合语音时，可以动态切换 Prefix，或者训练一个混合语言的 Prefix。
