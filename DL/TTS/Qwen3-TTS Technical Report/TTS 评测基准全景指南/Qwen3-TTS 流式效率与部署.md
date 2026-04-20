## 前置知识

> [!important]
> 
> 阅读本页前建议先读：[[DL/TTS/Qwen3-TTS Technical Report/Qwen3-TTS 模型组件架构细节/Qwen3-TTS 模型组件架构细节|Qwen3-TTS 模型组件架构细节]]、[[DL/TTS/Qwen3-TTS Technical Report/Qwen3-TTS 模型组件架构细节/DiT（Diffusion Transformer）在语音合成中的应用详解/DiT（Diffusion Transformer）在语音合成中的应用详解|DiT（Diffusion Transformer）在语音合成中的应用详解]]、[[DL/TTS/Qwen3-TTS Technical Report/Qwen3-TTS 模型组件架构细节/MTP 模块（Multi-Token Prediction）详解|MTP 模块（Multi-Token Prediction）详解]]。

---

## 0. 定位

> 本页专注于 Qwen3-TTS 的**流式推理效率**：延迟指标的定义、实测数据的解读、延迟来源的拆解、以及生产环境部署的工程优化。

---

## 1. 延迟指标定义

### 1.1 四个关键指标

|**指标**|**全称**|**含义**|**单位**|
|---|---|---|---|
|**TPP**|Time Per Packet|首包之后每个后续包的平均生成耗时|ms|
|**LM TTFP**|LM 部分的首包时间|仅 LM 骨干的首 token 延迟|ms|

### 1.2 RTF 的物理意义

$$\text{RTF} = \frac{T_{\text{compute}}}{T_{\text{audio}}}$$

- $\text{RTF} < 1$：合成速度快于实时，可流式播放不卡顿

- $\text{RTF} = 0.313$（12Hz-1.7B）：合成 1 秒音频仅需 0.313 秒

- **并发** $C$ **可行条件**：$C \times \text{RTF} < 1$（单卡上 12Hz-1.7B 理论最大并发约为 3）

---

## 2. 延迟实测数据

|**模型**|**并发**|**LM TTFP**|**解码**|**TTFP**|**TPP**|**RTF**|
|---|---|---|---|---|---|---|
|**12Hz-1.7B**|1|97|4|**101**|21|0.313|
|25Hz-1.7B|1|125|25|150|56|0.253|
|25Hz-1.7B|6|376|147|523|85|0.725|

> [!important]
> 
> **两个反直觉结论**：
> 
> 1. **0.6B 首包反而比 1.7B 快但 RTF 更低**：0.6B 的单步更快（首包优势）但单 token 计算并未等比缩小（内存访问成为瓶颈）。
> 
> 1. **25Hz 的 RTF 更低却 TTFP 更高**：25Hz 每 token 对应 40ms 音频（12Hz 是 80ms），单位时间生成的音频更多（RTF 低），但首包需等 16 token 的 DiT look-ahead（TTFP 高）。

---

## 3. 延迟来源拆解

### 3.1 12Hz 变体延迟瀑布图

![[2026-04-18 09.31.53Qwen3TTS 12Hz变体延迟图.excalidraw|200]]

**12Hz-1.7B 延迟分解（首包 101ms）**：

- LM Prefill（文本 → KV cache）：~30ms

- LM Decode 第 0 层 token：~60ms（主要开销）

- MTP 15 层残差：2–5ms

- 因果 ConvNet：4ms

- 总和：~97–99ms ≈ 101ms

### 3.2 25Hz 变体延迟瀑布图

![[2026-04-18 09.33.20Qwen3TTS 25Hz 变体延迟图.excalidraw|200]]

**25Hz-1.7B 延迟分解（首包 150ms）**：

- LM 需生成 16 个 token 才能启动解码（DiT look-ahead 需求）：~125ms

- DiT Flow Matching（10–20 步采样）：~15ms

- BigVGAN：~10ms

- 总和：~150ms

### 3.3 关键差异

|**差异点**|**12Hz**|**25Hz**|
|---|---|---|
|解码器类型|轻量因果 ConvNet|DiT + BigVGAN|
|单帧音频|80 ms|40 ms|

> [!important]
> 
> **为什么 25Hz 必须 look-ahead 16 tokens？**
> 
> DiT 的 Chunk-wise 注意力感受野 = 3 历史 chunk + 1 当前 chunk + 1 未来 chunk，每 chunk 8 tokens。首个 chunk 必须等未来 chunk 到齐（即 16 tokens）才能开始去噪。这是**质量/延迟权衡**的必然代价。

---

## 4. 并发场景的延迟放大

### 4.1 观察

从并发=1 到并发=6：

- 12Hz-1.7B：TTFP 101 → 333 ms（**+230%**）

- 25Hz-1.7B：TTFP 150 → 523 ms（**+248%**），解码器从 25 → 147 ms（**+488%**）

### 4.2 原因

![[2026-04-18 09.34.46Qwen3TTS 并发延迟图.excalidraw|600]]

- **LM 部分 batch 友好**：Transformer 的矩阵乘法天然适合 batching，开销近似线性增长。

- **DiT/BigVGAN batch 不友好**：Flow Matching 的多步采样引入同步开销，不同样本的 chunk 边界不对齐。

> [!important]
> 
> **生产部署建议（并发）**：
> 
> 1. **低延迟优先场景** → 选 12Hz，并发 ≤ 3，每请求独占资源
> 
> 1. **吞吐优先场景** → 12Hz 并发 6–8，接受 TTFP 300–400ms
> 
> 1. **长语音批处理** → 25Hz 并发低（2–3），避免 DiT batching 劣势

---

## 5. 推理优化技术

### 5.1 优化栈

|**层级**|**技术**|**收益**|
|---|---|---|
|编译|`torch.compile`|~1.3x speedup|
|精度|FP16 / BF16|2x throughput|
|缓存|KV cache|Decode 阶段复用|

### 5.2 vLLM 部署示例

```python
from vllm import LLM, SamplingParams
import torchaudio

# 启动 Qwen3-TTS-12Hz-1.7B 推理引擎
llm = LLM(
    model="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    dtype="bfloat16",
    enforce_eager=False,  # 启用 CUDA Graph
    gpu_memory_utilization=0.85,
    max_model_len=32768,
)

sampling = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=7500,  # 10 分钟音频 ≈ 7500 个 12Hz token
)

# 构造 ChatML prompt（简化示意）
prompt = build_chatml(
    system="你是一个中文播音员，语速适中",
    text="欢迎收听今日要闻。",
    speaker_wav="ref.wav",
)

# 流式生成
for output in llm.generate_streaming([prompt], sampling):
    speech_tokens = output.outputs[0].token_ids
    # 每累积一帧（80ms）就解码输出
    if len(speech_tokens) % 1 == 0:
        audio_chunk = code2wav_decode(speech_tokens[-1:])
        stream_to_client(audio_chunk)
```

### 5.3 Code2Wav 解码器的 CUDA Graph 优化

```python
import torch

# 首次调用记录 CUDA Graph
g = torch.cuda.CUDAGraph()
static_input = torch.zeros(1, 16, device="cuda", dtype=torch.long)
static_output = torch.zeros(1, 1920, device="cuda", dtype=torch.float32)

# warmup
for _ in range(3):
    static_output.copy_(code2wav(static_input))

# 捕获
with torch.cuda.graph(g):
    static_output.copy_(code2wav(static_input))

# 后续推理只需 replay（省去 kernel launch 开销）
def decode_fast(tokens):
    static_input.copy_(tokens)
    g.replay()
    return static_output.clone()
```

> [!important]
> 
> CUDA Graph 对**小 batch、固定 shape、高频率调用**的 kernel 收益最大。对于 12Hz 因果 ConvNet 每 80ms 调用一次的场景，CUDA Graph 可将解码延迟从 ~8ms 降到 ~4ms，**近乎减半**。

---

## 6. 端到端部署架构

![[2026-04-18 09.35.59TTS端到端部署图.excalidraw|400]]

### 6.1 关键设计

1. **LM 与 Code2Wav 分离部署**：LM 走大显存 GPU，Code2Wav 可放小卡或 CPU。

1. **请求队列 + 背压**：当 RTF 逼近 1/并发 时，队列快速堆积，需要熔断机制。

1. **首包优先调度**：新请求插队首包生成，已在生成的请求走普通优先级，保证用户感知流畅。

### 6.2 SLA 参考

|**场景**|**目标 TTFP**|**推荐配置**|
|---|---|---|
|语音助手响应|< 300 ms|12Hz-1.7B，并发 ≤ 4，A100|
|批量离线合成|无要求|25Hz-1.7B，并发 8+，RTF 优先|

---

## 7. 常见性能误区

> [!important]
> 
> **误区 1：「0.6B 模型一定更快」**
> 
> **反驳**：0.6B 的 TPP 更小（19 vs 21 ms），但 RTF 反而更低（0.288 vs 0.313）。因为 Decode 阶段受**内存带宽**限制，参数量减小带来的计算减少并未等比例反映在 wall-clock 上。批处理场景 1.7B 反而更划算。

> [!important]
> 
> **误区 2：「25Hz look-ahead 意味着它不能流式」**
> 
> **反驳**：25Hz 是**有限 look-ahead**（仅 1 个未来 chunk = 320ms 文本级别的前瞻），首包仍在 150ms 内完成，后续完全流式。

> [!important]
> 
> **误区 3：「RTF 越低越好」**
> 
> **反驳**：RTF 衡量「单请求吞吐」，但首包延迟是用户感知的关键。25Hz-1.7B 的 RTF（0.253）优于 12Hz-1.7B（0.313），但在实时交互场景 12Hz 仍是首选。

---

## 8. 小结

> [!important]
> 
> **延迟核心要点**
> 
> 1. **12Hz 首包快** — 因果设计 + 轻量解码器 + MTP 并行残差
> 
> 1. **25Hz 吞吐高** — 低 RTF，但 look-ahead 推高首包
> 
> 1. **并发放大效应** — DiT 不如 LM batch 友好
> 
> 1. **工程优化叠加** — vLLM + `torch.compile` + CUDA Graph + FlashAttention
> 
> 1. **SLA 驱动选型** — TTFP < 150ms 用 12Hz，否则 25Hz

---

## 延伸阅读

- [[DL/TTS/Qwen3-TTS Technical Report/Qwen3-TTS 模型组件架构细节/MTP 模块（Multi-Token Prediction）详解|MTP 模块（Multi-Token Prediction）详解]]：为什么 12Hz 首包这么快

- [[DL/TTS/Qwen3-TTS Technical Report/Qwen3-TTS 模型组件架构细节/DiT（Diffusion Transformer）在语音合成中的应用详解/DiT（Diffusion Transformer）在语音合成中的应用详解|DiT（Diffusion Transformer）在语音合成中的应用详解]]：25Hz look-ahead 的根源

---

## 参考文献

1. Qwen3-TTS Technical Report, Section 6 — Streaming Efficiency Analysis.

1. Kwon et al. _Efficient Memory Management for LLM Serving with PagedAttention_. SOSP 2023 (vLLM).

1. Dao et al. _FlashAttention-2: Faster Attention with Better Parallelism_. 2023.