# 函数调用与结构化输出

Function Calling（函数调用）是 Agent 连接外部世界的桥梁。通过定义清晰的 JSON Schema，大模型可以将自然语言指令转化为可执行的 API 调用。在语音交互场景中，这涉及到对音频文件、流媒体和声学参数的精确控制。

## 1. Function Calling 原理

LLM 本身无法直接执行代码，但它可以被训练为输出特定的 JSON 格式。
流程如下：
1.  **User**: "把这首歌变快一点。"
2.  **System**: 提供可用工具列表（如 `change_speed`, `play_music`）。
3.  **LLM**: 分析意图，输出结构化调用：
    ```json
    {
      "name": "change_speed",
      "arguments": {
        "file_path": "current_song.mp3",
        "factor": 1.2
      }
    }
    ```
4.  **Runtime**: 执行 Python 函数 `change_speed("current_song.mp3", 1.2)`。
5.  **System**: 将执行结果反馈给 LLM。

---

## 2. 定义工具：JSON Schema

OpenAI 等模型使用 JSON Schema 来描述工具。

### 2.1 基础结构
```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "获取当前天气",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {"type": "string"}
      },
      "required": ["location"]
    }
  }
}
```

### 2.2 语音处理工具定义示例

在语音 Agent 中，参数通常涉及文件路径、时间戳、语言代码等。

#### 示例 1：语音转文字 (ASR)
```json
{
  "name": "transcribe_audio",
  "description": "将音频文件转录为文字，支持多种语言。",
  "parameters": {
    "type": "object",
    "properties": {
      "audio_file_path": {
        "type": "string",
        "description": "音频文件的本地绝对路径或 URL。"
      },
      "language": {
        "type": "string",
        "enum": ["zh", "en", "ja", "auto"],
        "description": "音频的语言代码，默认为 auto。"
      },
      "timestamp_granularity": {
        "type": "string",
        "enum": ["word", "sentence", "none"],
        "description": "是否返回时间戳，以及时间戳的粒度。"
      }
    },
    "required": ["audio_file_path"]
  }
}
```

#### 示例 2：语音合成 (TTS) 与情感控制
```json
{
  "name": "generate_speech",
  "description": "将文本转换为语音，可控制情感和语速。",
  "parameters": {
    "type": "object",
    "properties": {
      "text": {
        "type": "string",
        "description": "需要朗读的文本内容。"
      },
      "speaker_id": {
        "type": "string",
        "description": "说话人 ID，如 'xiaoyu_news', 'lili_story'。"
      },
      "emotion": {
        "type": "string",
        "enum": ["neutral", "happy", "sad", "angry", "excited"],
        "description": "语音的情感基调。"
      },
      "speed": {
        "type": "number",
        "minimum": 0.5,
        "maximum": 2.0,
        "description": "语速倍率，1.0 为正常速度。"
      }
    },
    "required": ["text", "speaker_id"]
  }
}
```

---

## 3. 结构化输出 (Structured Output)

除了调用工具，有时我们需要 LLM 直接输出结构化数据，用于控制下游的信号处理算法。

### 3.1 场景：音乐生成参数控制
假设我们有一个基于规则的音乐合成器，需要 LLM 将用户的模糊描述转化为具体参数。

**User**: "来一段快节奏的、有点忧伤的钢琴曲。"

**LLM Output (JSON)**:
```json
{
  "tempo_bpm": 140,
  "key_signature": "C Minor",
  "instrument": "Grand Piano",
  "dynamics": "mf",
  "reverb_level": 0.4
}
```

### 3.2 场景：音频剪辑列表 (EDL)
Agent 规划剪辑任务时，输出剪辑决策列表 (Edit Decision List)。

**User**: "把这段录音里的静音部分都去掉，然后把音量调大一点。"

**LLM Output (JSON)**:
```json
{
  "operations": [
    {
      "type": "remove_silence",
      "threshold_db": -40,
      "min_silence_duration_ms": 500
    },
    {
      "type": "normalize",
      "target_db": -3.0
    }
  ]
}
```

## 4. 最佳实践

1.  **路径处理**: 音频文件通常较大，不适合直接在 JSON 中传递 Base64 编码。推荐传递**文件路径 (File Path)** 或 **对象存储链接 (S3 URL)**。
2.  **容错机制**: ASR 可能会识别错误（如把函数名听错）。在 System Prompt 中加入模糊匹配逻辑或纠错机制。
3.  **流式参数**: 对于流式 TTS，可能需要设计特殊的 chunked JSON 格式，以便一边生成参数一边合成。
