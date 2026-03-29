## 概述

GQA（Grouped Query Attention）和 MQA（Multi-Query Attention）通过减少 KV head 数来降低 KV cache 和 decode 带宽压力，是当前主流 LLM 的标配。

---

## 从 MHA 到 GQA 到 MQA

### MHA（Multi-Head Attention）

$$h_{kv} = h, \quad \text{KV cache} \propto h \cdot d_{head} \cdot T$$

每个 query head 有独立的 K、V head。

### GQA（Grouped Query Attention）

$$h_{kv} = h / g, \quad \text{KV cache} \propto (h/g) \cdot d_{head} \cdot T$$

每 $g$ 个 query head 共享 1 个 KV head。

### MQA（Multi-Query Attention）

$$h_{kv} = 1, \quad \text{KV cache} \propto d_{head} \cdot T$$

所有 query head 共享同 1 个 KV head。

---

## 对比

|方案|KV cache 缩减|质量影响|decode 带宽节省|代表模型|
|---|---|---|---|---|
|MHA|基准|基准|基准|GPT-3, LLaMA-1|
|GQA-4|4x|极小|~4x|LLaMA-2-70B (g=8)|
|GQA-8|8x|很小|~8x|Qwen-2.5-7B (h=28,h_kv=4)|
|MQA|$h$x|可测量但小|$h$x|Falcon-40B, PaLM-2|

> [!important]
> 
> **为什么 GQA 成为主流**：MQA 在极端情况下会损失一些质量（所有 head 共享同一 KV），GQA 在质量和效率间取得了极好的平衡——几乎无损地获得大幅 KV 缩减。

---

## 对参数量的影响

GQA/MQA 减少的是 KV 投影矩阵，但 MLP 仍为参数主项：

$$\Delta N_{attn} = 2(h - h_{kv}) \cdot d_{head} \cdot d \cdot L$$

对于 70B 模型（$h=64, h_{kv}=8, d=8192, L=80$）：

$$\Delta N = 2 \times 56 \times 128 \times 8192 \times 80 \approx 9.4B$$

减少约 14%，但 **对 KV cache 和 decode 带宽的影响远大于对参数量的影响**。

---

## 实现示例

```Python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GQAAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, head_dim):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_groups = n_heads // n_kv_heads  # 每组的 query head 数
        
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)
    
    def forward(self, x):
        B, S, _ = x.shape
        
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim)
        
        # 将 KV heads 扩展到与 Q heads 相同数量
        k = k.repeat_interleave(self.n_groups, dim=2)
        v = v.repeat_interleave(self.n_groups, dim=2)
        
        # 转置为 (B, heads, S, d)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        
        # FlashAttention via PyTorch
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)

# LLaMA-2-70B 风格
attn = GQAAttention(d_model=8192, n_heads=64, n_kv_heads=8, head_dim=128)
print(f"Q params: {64*128*8192/1e6:.1f}M")
print(f"KV params: {2*8*128*8192/1e6:.1f}M (vs MHA: {2*64*128*8192/1e6:.1f}M)")
```