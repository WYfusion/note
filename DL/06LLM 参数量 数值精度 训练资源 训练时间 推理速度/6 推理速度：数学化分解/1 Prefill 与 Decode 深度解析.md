## 概述

LLM 推理分为 **Prefill**（处理 prompt）和 **Decode**（生成 token）两个截然不同的阶段，其计算特征和瓶颈完全不同。

---

## Prefill 阶段

### 特征

- **输入**：整个 prompt（$S$ 个 token）

- **计算**：所有 token 可并行处理

- **FLOPs**：$approx 2N times S$（单次前向）

- **特性**：**compute-bound**（大矩阵乘，Tensor Core 利用率高）

### 时间

$$T_{prefill} \approx \frac{2N \times S}{G \times F_{peak} \times \eta_{prefill}}$$

> [!important]
> 
> Prefill 决定了 **TTFT**（Time To First Token）——用户感知的首次响应时间。长 prompt 时 TTFT 可能很高。

---

## Decode 阶段

### 特征

- **输入**：每步仅 1 个 token（+ KV cache）

- **计算**：严格串行——每个 token 依赖前一个

- **FLOPs**：$approx 2N_{active}$ 每 token（GEMM）+ KV attention

- **特性**：**memory-bound**（batch=1 时矩阵太小，算力吃不满）

### 时间

$$T_{decode\_per\_token} \approx \max\left(\frac{2N_{active}}{\text{sustained\_FLOPs}}, \frac{\text{weight\_bytes} + \text{KV\_bytes}}{\text{mem\_bandwidth}}\right)$$

> [!important]
> 
> Decode 决定了 **TPOT**（Time Per Output Token）。单 batch decode 时，GPU 算力利用率可能仅 ~5-15%，因为瓶颈在内存带宽而非算力。

---

## 对比总结

|维度|Prefill|Decode|
|---|---|---|
|并行度|高（$S$ 个 token 并行）|低（每步 1 token）|
|瓶颈|Compute-bound|Memory-bound|
|GEMM 形状|大矩阵 (high utilization)|向量-矩阵乘 (low utilization)|
|关键指标|TTFT|TPOT|
|KV cache|生成 KV|读取 KV（持续增长）|
|优化重点|算力利用率、TP|带宽、batching、KV 量化|

---

## Arithmetic Intensity 分析

$$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes accessed}}$$

### Prefill

$$AI_{prefill} \approx \frac{2N \times S}{N \times b_w / 8} = \frac{16S}{b_w}$$

BF16 ($b_w = 16$)：$AI = S$。当 $S > 100$，已远超 GPU 的 operational intensity 拐点 → **compute-bound**。

### Decode (batch=1)

$$AI_{decode} \approx \frac{2N}{N \times b_w / 8} = \frac{16}{b_w}$$

BF16：$AI = 1$。远低于拐点 → **严格 memory-bound**。

> [!important]
> 
> **提升 decode 利用率的核心方法**：增大 batch size（continuous batching），使 AI 提升到接近拐点。

---

## Python 估算

```Python
def estimate_inference_time(
    N: float,          # 参数量
    S_prompt: int,     # prompt 长度
    S_gen: int,        # 生成长度
    b_w: int = 16,     # 权重 bit
    peak_flops: float = 990e12,    # H100 BF16 TFLOPS
    mem_bw: float = 3.35e12,       # H100 HBM bandwidth bytes/s
    mfu_prefill: float = 0.5,
    mfu_decode: float = 0.05,      # decode batch=1 通常很低
) -> dict:
    weight_bytes = N * b_w / 8
    
    # Prefill
    prefill_flops = 2 * N * S_prompt
    ttft = prefill_flops / (peak_flops * mfu_prefill)
    
    # Decode (memory-bound)
    decode_mem_time = weight_bytes / mem_bw  # 读取所有权重
    decode_compute_time = (2 * N) / (peak_flops * mfu_decode)
    tpot = max(decode_mem_time, decode_compute_time)
    
    total = ttft + tpot * S_gen
    
    return {
        "TTFT_ms": ttft * 1000,
        "TPOT_ms": tpot * 1000,
        "total_s": total,
        "tokens_per_sec": S_gen / (tpot * S_gen),
    }

# LLaMA-2-7B on H100
result = estimate_inference_time(N=7e9, S_prompt=512, S_gen=256)
print(f"TTFT: {result['TTFT_ms']:.1f}ms, TPOT: {result['TPOT_ms']:.1f}ms")
```