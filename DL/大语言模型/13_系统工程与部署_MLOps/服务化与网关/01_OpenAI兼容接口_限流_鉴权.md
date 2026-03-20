# 服务化与网关

将模型封装为高可用的 API 服务。

## 1. 接口设计 (API Design)

推荐遵循 OpenAI 的 API 规范，以便接入现有的生态工具（如 LangChain）。

### 1.1 语音转文字 (ASR)
```http
POST /v1/audio/transcriptions
Content-Type: multipart/form-data

file: (binary audio)
model: whisper-1
language: zh
response_format: json
```

### 1.2 语音合成 (TTS)
```http
POST /v1/audio/speech
Content-Type: application/json

{
  "model": "tts-1",
  "input": "你好，世界",
  "voice": "alloy"
}
```

---

## 2. 流式协议 (Streaming Protocols)

语音交互对实时性要求极高，传统的 HTTP 请求-响应模式往往不够用。

### 2.1 WebSocket
全双工通信，适合实时对话（Real-time Conversation）。
*   **Client**: 持续发送音频块 (Audio Chunks)。
*   **Server**: 持续接收，VAD 检测到句尾后立即识别，并流式返回生成的音频。

### 2.2 gRPC
高性能 RPC 框架，适合微服务内部通信（如网关 -> ASR 服务 -> LLM 服务 -> TTS 服务）。
*   支持双向流 (Bidirectional Streaming)。

---

## 3. 批处理策略 (Batching Strategy)

### 3.1 动态批处理 (Dynamic Batching)
服务端积攒一小段时间（如 50ms）内的请求，组成一个 Batch 一起推理。

### 3.2 语音数据的 Padding 问题
音频长度差异巨大（有的 1s，有的 30s）。如果直接 Batch，短音频需要 Pad 大量的 0，浪费计算资源。
*   **优化**: 按长度排序 (Bucketing)，将长度相近的音频放在一个 Batch 中。
