# KV Cache 管理与显存估算

vLLM 的核心优势在于对显存的精细化管理。理解其机制有助于避免 OOM 并最大化性能。

## 1. 显存分配策略

vLLM 启动时会预先占用大部分显存。
1.  **加载模型权重**: 固定占用。
2.  **预留 KV Cache**: 将剩余显存的 `gpu_memory_utilization` (默认 90%) 划分为 KV Cache Blocks。

### 1.1 `gpu_memory_utilization`
*   **默认**: 0.9。
*   **调整**:
    *   如果与其他进程（如 TTS 服务）共享 GPU，需调低此值（如 0.6）。
    *   如果遇到 OOM，尝试调低到 0.85。

### 1.2 `block_size`
*   **定义**: PagedAttention 中每个 Block 包含的 Token 数。
*   **默认**: 16。
*   **影响**:
    *   较大的 Block Size (32/64) 可能提高访存效率，但会增加碎片浪费（最后一个 Block 未填满）。
    *   通常保持默认即可。

---

## 2. Swap Space (CPU 卸载)

当 GPU 显存不足以存放所有并发请求的 KV Cache 时，vLLM 会将部分 Block 换出（Swap out）到 CPU 内存。
*   **`--swap-space`**: 每个 GPU 预留的 CPU 交换空间大小（GB）。默认 4GB。
*   **作用**: 防止请求直接失败。虽然速度变慢，但保证了服务的可用性。

---

## 3. Audio LLM 的显存估算案例

假设使用 Qwen-Audio-Chat (7B, FP16)。
*   **权重占用**: $7 \times 10^9 \times 2 \text{ Bytes} \approx 14 \text{ GB}$。
*   **显卡**: RTX 3090 (24 GB)。
*   **剩余显存**: $24 - 14 = 10 \text{ GB}$。
*   **KV Cache 可用**: $10 \times 0.9 = 9 \text{ GB}$。

如果一段音频 Prompt 对应 2000 Tokens，生成 500 Tokens。
*   单请求 KV Cache $\approx 2 \times \text{Layers} \times \text{Hidden} \times \text{Seq} \times 2 \text{ Bytes}$。
*   粗略估算，7B 模型每 1000 Token 约占用 1GB KV Cache (取决于具体架构)。
*   **并发度**: 9GB 显存大约能支持 9 个并发请求（每个请求 1000 Token）。

**结论**: 对于 Audio LLM，由于 Input Sequence 通常很长，显存往往是并发度的瓶颈。
