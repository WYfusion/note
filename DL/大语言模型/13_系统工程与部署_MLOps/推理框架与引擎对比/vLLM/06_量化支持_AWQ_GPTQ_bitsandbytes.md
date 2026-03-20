# 量化支持：AWQ, GPTQ 与 BitsAndBytes

为了在有限的显存中运行更大的模型或提高吞吐量，量化是必不可少的。vLLM 对量化模型有很好的支持，特别是 AWQ。

## 1. AWQ (Activation-aware Weight Quantization)

AWQ 是目前 vLLM 官方推荐的量化格式，因为它在保持精度的同时，推理内核（Kernel）经过了高度优化。

### 1.1 运行 AWQ 模型
假设你已经下载了 `Qwen/Qwen-Audio-Chat-AWQ`（假设存在此版本）。

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen-Audio-Chat-AWQ \
    --quantization awq \
    --dtype half
```

*   **显存节省**: 7B 模型 INT4 量化后权重仅占约 4GB（原 14GB），给 KV Cache 留出了巨大空间，显著提升并发度。

---

## 2. GPTQ

GPTQ 是另一种流行的量化格式，vLLM 同样支持。

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen-Audio-Chat-GPTQ \
    --quantization gptq
```

### 2.1 AWQ vs GPTQ
*   **精度**: 两者在 4-bit 下表现相近。
*   **速度**: 在 vLLM 中，AWQ 的 Kernel 实现通常比 GPTQ 更快。

---

## 3. FP8 (Hopper 架构专用)

如果你使用的是 H100 或 RTX 4090，可以使用 FP8 量化。
*   **优势**: 硬件原生支持 FP8 计算，吞吐量翻倍。
*   **使用**:
    ```bash
    --quantization fp8
    ```

---

## 4. 语音模型的量化陷阱

对于 Audio LLM，量化通常只应用于 LLM 部分（Decoder）。
*   **Audio Encoder**: 通常保持 FP16。因为 Encoder 参数量占比小（如 Whisper Encoder 仅占总参数的 1/3 或更少），但对音频特征的提取精度至关重要。
*   **混合精度**: 优秀的量化模型（如 Qwen-Audio-Int4）通常是混合量化的：Encoder 保持高精度，LLM 部分量化为 Int4。vLLM 会自动处理这种结构。
