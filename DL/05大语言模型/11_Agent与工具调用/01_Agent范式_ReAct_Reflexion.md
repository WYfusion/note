# Agent 范式：ReAct 与 Reflexion

Agent（智能体）是大语言模型从“被动问答”走向“主动解决问题”的关键形态。本节介绍 Agent 的核心范式，并重点探讨语音交互场景下的 Audio Agent 设计。

## 1. Agent 核心架构

一个通用的 Agent 系统通常由以下组件构成：

$$
\text{Agent} = \text{LLM (Brain)} + \text{Memory} + \text{Planning} + \text{Tools (Action)}
$$

*   **大脑 (Brain)**: 负责推理、规划和决策的核心 LLM。
*   **感知 (Perception)**:
    *   **文本**: 用户 Prompt。
    *   **语音 (Audio Agent)**: 通过 ASR 转录文本，或通过 Audio Encoder (如 CLAP, AudioLM) 直接提取声学特征（语调、情感、背景音）。
*   **工具 (Tools)**: 执行外部操作的能力（API 调用、代码执行）。
*   **规划 (Planning)**: 分解复杂任务（CoT, ToT）。

---

## 2. ReAct 范式 (Reasoning + Acting)

ReAct (Yao et al., 2022) 是最经典的 Agent 范式，它要求模型在执行动作前先进行思考，形成 "Thought-Action-Observation" 的循环。

### 2.1 标准流程
1.  **Thought**: 分析当前状态，决定下一步做什么。
2.  **Action**: 调用工具（如搜索引擎、计算器）。
3.  **Observation**: 获取工具的返回结果。
4.  **Repeat**: 重复上述步骤，直到得出最终答案。

### 2.2 语音场景下的 Audio-ReAct
在语音交互中，Agent 不仅要处理语义，还要处理音频信号。

**案例：制作一段配乐朗诵**
*   **User**: "请用悲伤的语调朗读这首诗，并配上雨声背景音。"
*   **Thought 1**: 用户需要语音合成，且指定了情感（悲伤）和背景音（雨声）。首先需要生成朗读音频。
*   **Action 1**: `TTS_Generate(text="...", emotion="sad")`
*   **Observation 1**: `audio_speech.wav` 生成完毕，时长 15s。
*   **Thought 2**: 现在需要生成背景雨声，时长需覆盖朗读音频。
*   **Action 2**: `Audio_Generation(prompt="rain sound", duration=15s)`
*   **Observation 2**: `audio_bg.wav` 生成完毕。
*   **Thought 3**: 需要将两段音频混合。
*   **Action 3**: `Audio_Mix(foreground="audio_speech.wav", background="audio_bg.wav", balance=0.8)`
*   **Observation 3**: `final_output.wav` 生成完毕。
*   **Final Answer**: "好的，这是为您生成的配乐朗诵。" (播放音频)

---

## 3. Reflexion 范式 (Self-Correction)

Reflexion (Shinn et al., 2023) 引入了“反思”机制。当 Agent 任务失败或效果不佳时，它会生成一段自我反思（Self-Reflection），作为下一次尝试的上下文。

### 3.1 流程
$$
\text{Trial}_t \rightarrow \text{Evaluation} \rightarrow \text{Reflection} \rightarrow \text{Trial}_{t+1}
$$

### 3.2 语音 Agent 的自我修正
语音任务的主观性强（如“声音不够宏大”），Reflexion 尤为重要。

*   **Task**: "生成一段激昂的交响乐。"
*   **Trial 1**: 调用 MusicGen 生成音频。
*   **Evaluation**: (通过 CLAP 评分或用户反馈) "听起来有点像独奏，不够宏大。"
*   **Reflection**: "上一次生成的乐器太单一，Prompt 中应该强调 'orchestra', 'epic', 'many instruments'。"
*   **Trial 2**: 修改 Prompt 为 "Epic orchestral symphony, huge brass section, cinematic percussion"，再次调用 MusicGen。

---

## 4. 语音 Agent 的特殊挑战

### 4.1 跨模态感知与对齐
Audio Agent 需要理解非语言信息。
*   **语调感知**: 用户说 "真是太棒了" (讽刺语气)，Agent 若只看 ASR 文本会误判为夸奖。
*   **解决方案**: 引入 Speech Encoder (如 HuBERT, WavLM) 的特征作为 LLM 的输入，或使用多模态 LLM (如 GPT-4o, Qwen-Audio)。

### 4.2 全双工交互 (Duplex / Turn-taking)
传统的 ReAct 是回合制的 (Turn-based)，而真实的语音对话是流式的。
*   **打断 (Barge-in)**: 用户在 Agent 说话时插话，Agent 需立即停止生成并处理新指令。
*   **VAD (Voice Activity Detection)**: 实时判断用户是否说完，避免抢话或反应迟钝。
*   **Backchannel**: Agent 在用户说话时发出 "嗯"、"对" 等反馈，表示在听。

### 4.3 工具调用的延迟敏感性
语音交互对延迟极度敏感（通常要求 < 500ms）。
*   **Streaming Tool Use**: 工具一边生成音频，Agent 一边播放（流式 TTS）。
*   **Speculative Execution**: 预测用户可能需要的工具，预加载模型。

## 5. 总结
| 特性 | 文本 Agent | 语音 Agent (Audio Agent) |
| :--- | :--- | :--- |
| **输入** | Text | Audio (Speech + Sound + Music) |
| **核心循环** | ReAct (Text Thought) | Audio-ReAct (Multimodal Thought) |
| **反馈机制** | 文本匹配、代码运行结果 | 听感评估、CLAP Score、ASR WER |
| **交互模式** | 异步/回合制 | 同步/流式/全双工 |
