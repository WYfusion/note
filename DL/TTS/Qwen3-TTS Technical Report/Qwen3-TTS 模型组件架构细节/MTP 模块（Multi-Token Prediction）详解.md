## 前置知识

> [!important]
> 
> 阅读本页前建议先读：[[DL/TTS/Qwen3-TTS Technical Report/Qwen3-TTS 模型组件架构细节/Mimi 音频编解码器架构详解/Mimi 音频编解码器架构详解|Mimi 音频编解码器架构详解]]、[[RVQ 数学原理与训练目标]]。

---

## 0. 定位

> MTP（Multi-Token Prediction）是 Qwen3-TTS **12Hz 变体专属**的轻量预测头，位于 LM 骨干之后、BigVGAN 之前，负责**并行地预测 RVQ 第 1–15 层残差 token**。它是「单 LM 预测语义 + 多码本并行生成声学」这条核心设计的工程落点。

---

## 1. 为什么需要 MTP

### 1.1 多码本预测的本质挑战

12Hz Tokenizer 每帧产出 **16 层 RVQ token**（第 0 层语义 + 第 1–15 层声学残差）。若让 LM 按 token 展平后逐层自回归，代价是：

|**维度**|**LM 逐层自回归**|**LM + MTP 分工**|
|---|---|---|
|单帧前向次数|16 次 LM forward|1 次 LM + 1 次 MTP|
|KV-Cache 长度|展平 16×|保持原长度|
|注意力计算量|$O(256 N^2)$|$O(N^2)$|
|首包延迟（1.7B）|> 500 ms|≈ 101 ms|

### 1.2 核心洞察：RVQ 的能量分布

根据 [[RVQ 数学原理与训练目标]]，残差范数沿层指数衰减 $\|\mathbf{r}_k\|_2 \lesssim \alpha^k \|\mathbf{x}\|_2$，信息量高度集中在第 0 层。

> [!important]
> 
> **分工原则**：第 0 层承担语义，需要 LM 的全局上下文；第 1–15 层只是局部声学残差，轻量网络足够。一个 13B 的 LM 去预测高频噪声尾部在工程上是浪费。

---

## 2. 模块级位置图

![[2026-04-18 10.18.02MTP模块级位置.excalidraw|500]]

> [!important]
> 
> **关键约束**：MTP 的唯一「全局信息源」就是 $h_t$。一旦 LM forward 完成，MTP 本身不再回头看文本或历史音频 token，这是它能保持流式的根本原因。

---

## 3. MTP 内部架构

### 3.1 结构总览

![[2026-04-18 10.20.50MTP内部图.excalidraw|500]]

### 3.2 组件职责

|**组件**|**输入**|**输出**|**参数量占比**|
|---|---|---|---|
|条件融合 $f$|$h_t, e_0$|$z \in \mathbb{R}^{d_{\text{mtp}}}$|&lt; 1%|
|MTP Backbone|$z$|上下文表征 $u$|3–7%|
|分层头 $W_k$（$k=1..15$）|$u, e_{k-1}$|$\text{logits}_k \in \mathbb{R}^{V}$|1–2%|
|**合计**|—|—|**≈ LM 的 5–10%**|

---

## 4. 分层预测的数学形式化

### 4.1 单帧预测递推

设 LM 输出 $h_t$、融合向量 $z = f(h_t, e_0)$、Backbone 输出 $u = \text{MTP}_{\theta}(z)$。对 $k = 1, 2, \dots, 15$：

$$\begin{aligned}  
\text{logits}_k &= W_k \cdot \phi_k(u,\ e_{k-1}) \\  
p(c_k \mid h_t, c_0, c_{<k}) &= \text{softmax}(\text{logits}_k) \\  
c_k &\sim p(c_k \mid \cdot),\qquad e_k = \text{Embed}_k(c_k)  
\end{aligned}$$

其中 $\phi_k$ 可以是简单的 `concat + Linear`，也可以是一个小 MLP。

### 4.2 三种连接拓扑

![[2026-04-18 10.23.41.excalidraw|1200]]

> [!important]
> 
> **为什么选级联而不是完全并行？** RVQ 每层的语义**不独立**——第 $k$ 层建模的是「前 $k-1$ 层量化完之后的残差空间」，因此以 $c_{k-1}$ 为条件可以消除「重复量化」和「层间塌缩」两类常见问题。代价只是 15 次轻量 head 串行，加起来不到 1 ms。

### 4.3 训练目标

$$\mathcal{L}_{\text{MTP}} = \sum_{k=1}^{15} w_k \cdot \text{CE}\!\left( p(c_k \mid h_t, c_0, c_{<k}),\ c_k^{\text{gt}} \right)$$

端到端总损失：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}}^{(0)} + \lambda \cdot \mathcal{L}_{\text{MTP}}$$

- 权重 $w_k$ 通常随 $k$ 递减（高层残差本来就难预测，不给它背过大权重）。

- $\lambda \in [0.3, 0.5]$：保证 LM 主损失仍然主导梯度。

- 训练时采用 **teacher forcing**：$e_{k-1}$ 取真值 embedding；推理时采用 **cascading sampling**：$e_{k-1}$ 取上一步采样结果的 embedding。

---

## 5. 推理流程（单帧步骤级）

![[2026-04-18 10.24.35.excalidraw|200]]

**延迟拆分**（1.7B 模型，单并发，A100）：

|**步骤**|**耗时**|**占比**|
|---|---|---|
|Step 1 LM forward|≈ 85 ms|84%|
|Step 2–3 语义采样 + 融合|≈ 1 ms|1%|
|Step 4 MTP backbone|≈ 3 ms|3%|
|Step 5 15 头级联采样|≈ 2 ms|2%|
|Step 6 ConvNet + BigVGAN|≈ 10 ms|10%|
|**合计 TTFP**|**≈ 101 ms**|**100%**|

---

## 6. PyTorch 参考实现

```python
import torch
import torch.nn as nn

class MTPHead(nn.Module):
    """单层残差预测头：以 (u, e_{k-1}) 为输入，输出 V 维 logits。"""
    def __init__(self, d_mtp: int, d_emb: int, V: int):
        super().__init__()
        self.proj = nn.Linear(d_mtp + d_emb, d_mtp)
        self.out = nn.Linear(d_mtp, V)

    def forward(self, u, e_prev):
        h = torch.relu(self.proj(torch.cat([u, e_prev], dim=-1)))
        return self.out(h)


class MTPModule(nn.Module):
    """Qwen3-TTS MTP: 级联预测第 1..K 层 RVQ token。"""
    def __init__(self, d_model, d_mtp, d_emb, V=2048, K=15, n_layers=3):
        super().__init__()
        self.fuse = nn.Linear(d_model + d_emb, d_mtp)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_mtp, nhead=8, dim_feedforward=4 * d_mtp, batch_first=True
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.embeds = nn.ModuleList([nn.Embedding(V, d_emb) for _ in range(K)])
        self.heads = nn.ModuleList([MTPHead(d_mtp, d_emb, V) for _ in range(K)])
        self.K = K

    def forward(self, h_t, c0, c_gt=None):
        # h_t: (B, T, d_model)，c0: (B, T)
        e0 = self.embeds[0](c0)                              # 用第 0 层 embedding 作为初始条件
        z = self.fuse(torch.cat([h_t, e0], dim=-1))
        u = self.backbone(z)                                 # (B, T, d_mtp)
        logits, e_prev = [], e0
        for k in range(self.K):
            lg = self.heads[k](u, e_prev)                    # (B, T, V)
            logits.append(lg)
            if self.training and c_gt is not None:
                e_prev = self.embeds[k](c_gt[..., k])        # teacher forcing
            else:
                e_prev = self.embeds[k](lg.argmax(-1))       # cascading
        return torch.stack(logits, dim=-2)                   # (B, T, K, V)
```

> [!important]
> 
> **工程提示**
> 
> 1. MTP backbone 用 **非因果** Self-Attn 即可（只看单帧内 16 个 token 的并列位置，不跨时间）。这和 LM 的因果注意力解耦，可单独优化。
> 
> 1. 15 个 head 可用 `torch.jit.script` 融合调用，GPU 内核启动开销从 15 降到 1。
> 
> 1. 级联深度大于 4 时建议加 **dropout(0.1)** 防止层间过拟合。

---

## 7. 与其他方案的取舍

|**方案**|**代表工作**|**延迟**|**质量**|**判断**|
|---|---|---|---|---|
|LM 展平逐层自回归|VALL-E, AudioLM|差|高|精度高但不能上线实时 TTS|
|完全并行 head|SoundStorm 早期|最优|明显下降|噪声/嘶哑可听见|
|MaskGIT / 迭代解码|SoundStorm|中|中高|非流式，不适合 TTFP|
|**级联 MTP**|**Qwen3-TTS**|**优**|**接近自回归**|**质量-延迟双优解**|

> [!important]
> 
> **工程判断链**：追求流式 TTFP → 先 MTP，再考虑并行头；离线生成可接受大延迟 → 选展平自回归（质量上限更高）。

---

## 8. 常见误区

> [!important]
> 
> **误区 1**：「MTP 就是 Medusa / 推测解码」。→ 不是。Medusa 预测的是**时间维度**的下几个 token，MTP 预测的是**同一帧内**不同 RVQ 层的 token，两者目标维度正交。

> [!important]
> 
> **误区 2**：「层数越多，MTP 越难训」。→ 实际反过来。高层残差能量极小（见 [[RVQ 数学原理与训练目标]] §3.5），天然易拟合；真正难的是第 1–3 层主要声学。

> [!important]
> 
> **误区 3**：「可以把 MTP 去掉，用更大的 LM 直接预测 16 层」。→ 不行。无论 LM 多大，逐层自回归都会把序列拉长 16×，KV-Cache 和延迟均不可接受。

---

## 延伸阅读

- [[DL/TTS/Qwen3-TTS Technical Report/Qwen3-TTS 模型组件架构细节/Mimi 音频编解码器架构详解/Mimi 音频编解码器架构详解|Mimi 音频编解码器架构详解]]：MTP 的预测目标来源

- [[RVQ 数学原理与训练目标]]：残差能量分布支撑分层预测的理论依据

- [[DL/TTS/Qwen3-TTS Technical Report/Qwen3-TTS 模型组件架构细节/BigVGAN 通用神经声码器详解|BigVGAN 通用神经声码器详解]]：MTP 输出 token 的下游消费者

---

## 参考文献

1. Qwen Team. _Qwen3-TTS Technical Report_. arXiv:2601.15621, 2026.

1. Défossez et al. _Moshi: a speech-text foundation model for real-time dialogue_. 2024 — Mimi codec 与多码本并行预测头。

1. Wang et al. _VALL-E: Neural Codec Language Models are Zero-Shot TTS Synthesizers_. 2023 — 展平自回归对比基线。

1. Borsos et al. _SoundStorm: Efficient Parallel Audio Generation_. 2023 — 并行/迭代解码对比。

1. Cai et al. _Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads_. 2024 — 时间维度多 token 预测（与 MTP 思路正交）。