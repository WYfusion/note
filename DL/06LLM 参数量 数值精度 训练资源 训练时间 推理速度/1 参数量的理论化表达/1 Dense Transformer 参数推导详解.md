## 概述

本页对 Dense Decoder-only Transformer 的参数量进行**逐矩阵推导**，并用主流开源模型验证近似公式的准确性。

---

## 逐模块参数拆解

### Self-Attention

|矩阵|维度|参数量|说明|
|---|---|---|---|
|$W_Q$|$d \times d$|$d^2$|Query 投影|
|$W_K$|$d \times (h_{kv} \cdot d_{head})$|$d \cdot h_{kv} \cdot d_{head}$|Key 投影（[[06_分组注意力|GQA]] 时 $h_{kv} < h$）|
|$W_V$|$d \times (h_{kv} \cdot d_{head})$|$d \cdot h_{kv} \cdot d_{head}$|Value 投影|
|$W_O$|$d \times d$|$d^2$|Output 投影|

**MHA**（$h_{kv} = h$, $d = h cdot d_{head}$）时：$N_{attn} = 4d^2$

**[[06_分组注意力GQA|GQA]]**（$h_{kv} < h$）时：

$$N_{attn} = d^2 + 2d \cdot h_{kv} \cdot d_{head} + d^2 = 2d^2 + 2d \cdot h_{kv} \cdot d_{head}$$

> [!important]
> 
> 当 $h_{kv} = h$（MHA）时退化为 $4d^2$；当 $h_{kv} = 1$（[[05_多查询注意力MQA|MQA]]）时为 $2d^2 + 2d cdot d_{head} approx 2d^2$。但对**总参数量**影响不大，因 MLP 仍为主项。

### MLP（SwiGLU 变体）

现代 LLM 多使用 SwiGLU（LLaMA/Qwen/Mistral 等）：

|矩阵|维度|参数量|说明|
|---|---|---|---|
|$W_{gate}$|$d \times d_{ff}$|$d \cdot d_{ff}$|Gate 投影|
|$W_{up}$|$d \times d_{ff}$|$d \cdot d_{ff}$|Up 投影|
|$W_{down}$|$d_{ff} \times d$|$d \cdot d_{ff}$|Down 投影|

$$N_{MLP} = 3 \cdot d \cdot d_{ff}$$

当 $d_{ff} = frac{8}{3}d$（LLaMA 系列常用）时：$N_{MLP} = 3 times d times frac{8}{3}d = 8d^2$

### RMSNorm

每层 2 个 RMSNorm（pre-attention + pre-MLP），每个 $d$ 参数：$N_{norm} = 2d$

### Embedding + LM Head

- Embedding：$V times d$

- LM Head：$d times V$（通常与 embedding 共享 = weight tying）

- 共享时仅算一份：$N_{embed} = Vd$

---

## 总参数量公式

### 精确表达

$$N = L \times (N_{attn} + N_{MLP} + N_{norm}) + N_{embed} + N_{final\_norm}$$

### MHA + SwiGLU 近似

$$N \approx L \times (4d^2 + 8d^2 + 2d) + Vd + d = 12Ld^2 + (2L+1)d + Vd$$

当 $d$ 足够大时：

$$\boxed{N_{dense} \approx 12Ld^2 + Vd}$$

---

## 典型模型验证

|模型|$L$|$d$|$d_{ff}$|$h / h_{kv}$|$V$|$12Ld^2 + Vd$ 估算|官方参数量|误差|
|---|---|---|---|---|---|---|---|---|
|LLaMA-2-7B|32|4096|11008|32/32|32000|6.59B|6.74B|~2%|
|LLaMA-2-13B|40|5120|13824|40/40|32000|12.7B|13.0B|~2%|
|LLaMA-2-70B|80|8192|28672|64/8|32000|64.6B|68.9B|~6%|
|Qwen-2.5-7B|28|3584|18944|28/4|152064|4.9B|7.6B|~36%|

> [!important]
> 
> **观察**：
> 
> - 当 $d_{ff} \approx \frac{8}{3}d$ 且使用 MHA 时，$12Ld^2$ 近似非常好（误差 <5%）
> 
> - **Qwen-2.5 偏差大**的原因：$d_{ff} = 18944 gg frac{8}{3} times 3584 = 9557$（MLP 比标准大近 2x），且词表 152K 远大于 LLaMA 的 32K
> 
> - 结论：$12Ld^2$ **是合理的主项近似，但实际计算必须代入真实** $d_{ff}$ **和** $V$

---

## Python 精确计算脚本

```Python
def count_dense_params(
    L: int,           # 层数
    d: int,           # 隐藏维
    d_ff: int,        # FFN 中间维
    h: int,           # query head 数
    h_kv: int,        # KV head 数
    d_head: int,      # 每头维度
    V: int,           # 词表大小
    weight_tying: bool = True,
) -> dict:
    """精确计算 Dense Decoder-only Transformer 参数量"""
    # Attention per layer
    n_q = d * (h * d_head)       # W_Q
    n_k = d * (h_kv * d_head)    # W_K
    n_v = d * (h_kv * d_head)    # W_V
    n_o = (h * d_head) * d       # W_O
    attn_per_layer = n_q + n_k + n_v + n_o

    # MLP per layer (SwiGLU: gate + up + down)
    mlp_per_layer = 3 * d * d_ff

    # RMSNorm per layer (2 norms)
    norm_per_layer = 2 * d

    # Total per layer
    per_layer = attn_per_layer + mlp_per_layer + norm_per_layer

    # Embedding + LM head
    embed = V * d
    lm_head = 0 if weight_tying else V * d
    final_norm = d

    total = L * per_layer + embed + lm_head + final_norm

    return {
        "attn_per_layer": attn_per_layer,
        "mlp_per_layer": mlp_per_layer,
        "per_layer": per_layer,
        "total_layers": L * per_layer,
        "embedding": embed,
        "total": total,
        "total_B": f"{total / 1e9:.2f}B",
    }

# 验证 LLaMA-2-7B
result = count_dense_params(
    L=32, d=4096, d_ff=11008, h=32, h_kv=32, d_head=128, V=32000
)
print(f"LLaMA-2-7B: {result['total_B']}")
# Output: LLaMA-2-7B: 6.61B
```