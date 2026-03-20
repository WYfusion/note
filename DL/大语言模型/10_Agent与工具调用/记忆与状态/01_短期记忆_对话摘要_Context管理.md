# 记忆与状态管理

Agent 的智能不仅体现在当前的推理能力，更体现在对历史交互的记忆与状态维护。在语音交互场景中，记忆不仅包含文本信息，还包含声纹、情感和音频偏好等非语言特征。

## 1. 记忆的分类

### 1.1 短期记忆 (Short-term Memory)
通常对应 LLM 的上下文窗口 (Context Window)。
*   **文本**: 最近几轮的对话历史。
*   **语音**: 最近生成的音频片段的临时缓存（用于后续的修改或引用）。

### 1.2 长期记忆 (Long-term Memory)
存储在外部数据库（向量数据库、SQL）中的持久化信息。
*   **知识库**: RAG 检索源。
*   **用户画像**: 用户的长期偏好。

---

## 2. 语音 Agent 的特殊记忆：Audio Memory

除了记住用户“说了什么”（语义），Audio Agent 还需要记住“怎么说的”（声学特征）以及“听到了什么”（音频内容）。

### 2.1 声纹记忆 (Voiceprint Memory)
Agent 需要识别并记住不同用户的声音身份。
*   **实现**: 提取用户的 Speaker Embedding (如 x-vector, d-vector) 并存入向量库。
*   **应用**:
    *   **多说话人场景**: "刚才说话的是张三，他之前提到过不喜欢吃辣。"
    *   **个性化服务**: 听到是老人的声音，自动调大 TTS 音量并放慢语速。

### 2.2 偏好记忆 (Preference Memory)
记录用户对音频生成的偏好参数。
*   **TTS 偏好**: "用户喜欢 'Deep Voice' 音色，语速 1.2x。"
*   **音乐偏好**: "用户倾向于 'Lo-Fi' 风格的背景音乐。"
*   **存储结构**:
    ```json
    {
      "user_id": "u123",
      "voice_preference": {
        "speaker_id": "v_05",
        "speed": 1.1
      },
      "music_preference": ["jazz", "lo-fi", "piano"]
    }
    ```

### 2.3 音频资产记忆 (Asset Memory)
用户可能引用之前生成的音频。
*   **User**: "把上周生成的那段吉他独奏拿出来，加点鼓点。"
*   **实现**:
    1.  生成音频时，计算其 CLAP Embedding 并存入向量库，同时保存元数据（时间、Prompt）。
    2.  解析 "上周生成的那段吉他独奏"，转化为时间过滤 + 语义检索。
    3.  检索到对应的音频文件路径，作为 `add_drums` 工具的输入。

---

## 3. 对话摘要与 Context 管理

由于音频转录的文本可能很长（且包含大量口语赘词），直接塞入 Context Window 效率低下。

### 3.1 文本摘要
*   **Raw ASR**: "呃，那个，我想听一首，就是那种，比较欢快的歌，对，欢快的。"
*   **Summarized**: "用户请求播放欢快的歌曲。"

### 3.2 状态机管理 (State Machine)
对于复杂的语音任务（如订票、游戏），使用有限状态机 (FSM) 管理当前阶段。
*   **State**: `Listening` -> `Processing` -> `Speaking` -> `Idle`.
*   **Interrupt Handling**: 如果在 `Speaking` 状态检测到 VAD 信号，立即切换到 `Listening` 并清空当前的 TTS 缓冲区。

## 4. 总结

| 记忆维度 | 文本 Agent | 语音 Agent |
| :--- | :--- | :--- |
| **身份识别** | User ID (Login) | User ID + Voiceprint (Biometric) |
| **内容记忆** | Text History | Text History + Audio Files |
| **偏好记忆** | Topic, Style | Topic, Style, Timbre, Speed, Volume |
| **检索方式** | Semantic Search | Semantic Search + Audio Similarity Search |
