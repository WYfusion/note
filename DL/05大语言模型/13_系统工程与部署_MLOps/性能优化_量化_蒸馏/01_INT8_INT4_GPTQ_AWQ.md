# 性能优化：量化与蒸馏

为了在端侧设备或高并发服务端部署语音大模型，必须进行极致的性能优化。

## 1. 模型量化 (Quantization)

将模型权重从 FP16 (16-bit) 压缩到 INT8 (8-bit) 或 INT4 (4-bit)。

### 1.1 主流算法
*   **GPTQ**: 逐层量化，利用 Hessian 矩阵调整权重以最小化误差。
*   **AWQ (Activation-aware Weight Quantization)**: 保护重要的权重（即激活值较大的权重）不被过度量化。

### 1.2 语音模型的量化策略
语音模型通常包含 Encoder (声学特征提取) 和 Decoder (文本/音频生成)。
*   **Decoder (LLM)**: 对量化较鲁棒，可激进地使用 INT4 (AWQ/GPTQ)。
*   **Encoder (Whisper/Audio Encoder)**: 对量化较敏感。
    *   建议保持 **FP16** 或 **INT8**。
    *   如果强行使用 INT4，可能会导致微弱信号（如耳语、背景音）丢失。

---

## 2. 知识蒸馏 (Knowledge Distillation)

通过“老师教学生”的方式，将大模型的能力迁移到小模型。

### 2.1 Distil-Whisper
*   **Teacher**: Whisper-large-v2 (1550M params).
*   **Student**: Distil-large-v2 (756M params).
*   **方法**: 使用伪标签 (Pseudo-labeling) 和 KL 散度 Loss。
*   **效果**: 速度提升 6 倍，WER 仅增加 1%。

### 2.2 语音生成模型的蒸馏
*   **非自回归蒸馏 (NAR Distillation)**: 将自回归 TTS 模型（一步生成一个 Token）蒸馏为非自回归模型（并行生成所有 Token），显著降低延迟。

---

## 3. 流式推理优化 (Streaming Inference)

语音交互要求低延迟，不能等整句话说完再处理。

### 3.1 KV Cache
对于自回归生成的 Audio LLM，KV Cache 是必须的，避免重复计算历史 Token 的 Attention。

### 3.2 流式编码器 (Streaming Encoder)
*   **问题**: 标准 Transformer Encoder 需要全序列 Attention。
*   **解决**:
    *   **Chunk-based Attention**: 将音频切分为固定块（如 2秒），只在块内或相邻块进行 Attention。
    *   **Emformer / Conformer**: 专为流式 ASR 设计的架构，使用带有 Look-ahead 限制的 Attention。
