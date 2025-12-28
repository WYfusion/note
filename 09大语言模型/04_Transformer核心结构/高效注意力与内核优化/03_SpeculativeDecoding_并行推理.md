# Speculative Decoding 与并行推理

推理延迟（Latency）是语音交互系统的关键指标。自回归生成（逐个 Token 生成）是主要的瓶颈。

## 1. Speculative Decoding (投机采样 / 推测解码)

### 1.1 核心思想
利用一个小模型（Draft Model）快速生成多个 Token，然后用大模型（Target Model）并行验证这些 Token。

### 1.2 流程
1.  **Draft**: 小模型快速生成 $K$ 个 Token。
2.  **Verify**: 大模型一次性计算这 $K$ 个 Token 的概率分布。
3.  **Accept/Reject**: 根据大模型的概率，决定接受哪些 Token。如果第 $i$ 个被拒绝，则从第 $i$ 个位置重新开始。

### 1.3 优势
*   **无损加速**: 生成结果与直接用大模型生成完全一致（数学上保证）。
*   **利用并行性**: 大模型验证 $K$ 个 Token 是并行的，比串行生成 $K$ 次快得多。

## 2. 并行推理 (Parallel Decoding)

### 2.1 Blockwise Parallel Decoding
在 Encoder-Decoder 模型中，有时可以一次性预测多个 Token。

### 2.2 NAR (Non-Autoregressive) 生成
完全打破自回归依赖，一次性生成所有 Token。
*   **Mask-Predict**: 像 BERT 一样，先生成全 Mask 序列，然后迭代优化。
*   **VALL-E 的 NAR 部分**: VALL-E 的第一层 Quantizer 是 AR 生成的，但后续 7 层 Quantizer 是 NAR 并行生成的，极大地提高了推理速度。

## 3. 语音流式推理 (Streaming Inference)

### 3.1 挑战
用户说话的同时，系统就要开始识别/生成，不能等整句话说完。

### 3.2 策略
*   **Chunk-wise Processing**: 每接收到固定的音频块（如 200ms），就进行一次推理。
*   **KV Cache Management**: 需要高效地管理 KV Cache，既要保留足够的历史上下文，又要避免显存无限增长。
*   **Attention Sink**: 发现保留最初的几个 Token（Sink Tokens）对保持长序列生成的稳定性至关重要。
