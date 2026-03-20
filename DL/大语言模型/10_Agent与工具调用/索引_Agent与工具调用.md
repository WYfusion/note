# Agent 与工具调用

本章节探讨大语言模型如何进化为能够感知环境、规划任务并调用工具的智能体 (Agent)，并重点介绍语音交互场景下的 Audio Agent 架构。

## 目录

### [01_Agent范式_ReAct_Reflexion.md](./01_Agent范式_ReAct_Reflexion.md)
- **核心架构**：Brain, Perception, Tools, Planning。
- **ReAct 范式**：Thought-Action-Observation 循环。
- **Reflexion 范式**：自我反思与修正。
- **Audio Agent 特性**：
  - **多模态感知**：ASR 与 Audio Encoder。
  - **Audio-ReAct**：处理音频信号的推理循环。
  - **全双工交互**：VAD、打断与流式处理。

## 子模块

### [函数调用与结构化输出](./函数调用与结构化输出/索引_函数调用与结构化输出.md)
- JSON Schema 定义。
- 语音工具 (ASR, TTS, Editing) 的参数设计。
- 结构化输出控制音频参数。

### [规划 / 反思 / 自一致性](./规划_反思_自一致性/索引_规划_反思_自一致性.md)
- CoT 与 ToT 在语音任务中的应用。
- Self-Consistency 提升 ASR 与生成质量。
- 复杂音频工程任务的规划。

### [记忆与状态](./记忆与状态/索引_记忆与状态.md)
- 短期记忆与 Context 管理。
- **Audio Memory**：声纹记忆、偏好记忆、音频资产检索。

