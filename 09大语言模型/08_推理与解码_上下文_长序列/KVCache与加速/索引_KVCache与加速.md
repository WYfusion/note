# KV Cache 与加速

本章节深入分析了 KV Cache 的原理与显存占用，以及 vLLM 的 PagedAttention 如何解决显存碎片化问题。针对 Audio LLM 序列长度爆炸的特性，提供了实用的优化策略。

## 目录

### [01_KVCache_原理与显存.md](./01_KVCache_原理与显存.md)
- **KV Cache 原理**：缓存历史 Token 的 Key/Value 向量，避免重复计算。
- **显存计算公式**：$\text{Size}_{KV} = 2 \times L \times N_{head} \times d_{head} \times S \times B \times P_{bytes}$。
- **语音挑战**：
  - 30秒音频 $\approx$ 2250 Token，KV Cache 达 1.1GB（vs 文本 60MB）。
  - MQA/GQA、Window Attention、KV Cache 量化等优化策略。

### [02_PagedAttention_vLLM思路.md](./02_PagedAttention_vLLM思路.md)
- **PagedAttention**：借鉴操作系统分页机制，解决显存碎片化。
- **核心机制**：Block 切分、Block Table 映射、非连续存储。
- **语音应用**：
  - 变长音频请求的零浪费处理。
  - 流式传输（Streaming）降低首字延迟。
  - System Prompt 的跨请求共享（Prefix Sharing）。
