# 常见评测基准 (Benchmarks)

基准测试集（Benchmarks）是衡量模型能力的标尺。本节涵盖通用的文本基准以及语音领域的专用基准。

## 1. 通用文本基准

*   **MMLU (Massive Multitask Language Understanding)**: 涵盖 STEM、人文、社科等 57 个学科的选择题，测试世界知识和推理能力。
*   **GSM8K**: 8.5k 个高质量的小学数学应用题，测试多步数学推理能力。
*   **HumanEval**: Python 编程任务，测试代码生成能力。
*   **HellaSwag**: 常识推理，补全句子的结尾。

---

## 2. 语音与音频基准 (Audio Benchmarks)

随着 Speech LLM 的发展，评测基准也在从单一任务向综合能力演进。

### 2.1 传统任务基准
*   **ASR (语音识别)**:
    *   **LibriSpeech**: 英语有声读物，分为 Clean (清晰) 和 Other (噪声) 子集。
    *   **Common Voice**: 多语言、大规模众包数据集。
    *   **Wenetspeech**: 大规模中文 ASR 数据集（10000+ 小时）。
*   **TTS (语音合成)**:
    *   **LJSpeech**: 单一女性说话人，用于测试音质和稳定性。
    *   **VCTK**: 109 个英语说话人，用于测试多说话人建模和口音适应。

### 2.2 生成任务基准
*   **AudioCaps**: 音频描述生成 (Audio Captioning) 和 文本生成音频 (Text-to-Audio) 的标准数据集。
*   **MusicCaps**: 音乐描述与生成数据集，包含专业的音乐术语标注。

### 2.3 Speech LLM 综合基准
针对新一代能够理解和生成语音的大模型。

*   **Dynamic-SUPERB**:
    *   涵盖 30+ 种语音任务，包括情感识别、说话人验证、语音问答等。
    *   特点：动态评估，模型需要根据指令完成未见过的任务。
*   **AIR-Bench (Audio Instruction Repository)**:
    *   专注于评估 Audio LLM 遵循指令的能力。
    *   包含 Chat (对话)、Reasoning (推理)、Generation (生成) 等多个维度。
*   **OpenCompass (Audio)**:
    *   上海人工智能实验室推出的全方位评测体系，包含对 Qwen-Audio, Whisper 等模型的评测。

## 3. 评测指标详解

### 3.1 ASR 指标
$$
\text{WER} = \frac{S + D + I}{N} \times 100\%
$$
*   $S$: 替换 (Substitutions)
*   $D$: 删除 (Deletions)
*   $I$: 插入 (Insertions)
*   $N$: 参考文本的总词数

### 3.2 生成质量指标
*   **FAD (Fréchet Audio Distance)**: 越低越好。计算生成音频与真实音频在 Embedding 空间（通常使用 VGGish 或 CLAP）的分布距离。
*   **KL Divergence**: 衡量生成音频的类别分布与真实分布的差异。
*   **Inception Score (IS)**: 衡量生成样本的多样性和清晰度（在音频领域较少用，逐渐被 FAD 取代）。
