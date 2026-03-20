# Scaled Dot-Product Attention (缩放点积注意力)

这是 Transformer 论文中最核心的公式，也是所有现代大模型（包括语音模型）的基石。

## 1. 核心公式

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中：
*   $Q$ (Query): 查询向量矩阵，形状 $[L, d_k]$。
*   $K$ (Key): 键向量矩阵，形状 $[L, d_k]$。
*   $V$ (Value): 值向量矩阵，形状 $[L, d_v]$ (通常 $d_v=d_k$)。
*   $d_k$: Key 的维度。

## 2. 逐步推导与物理意义

### 第一步：相似度计算 ($QK^T$)
我们想知道序列中第 $i$ 个 Token 和第 $j$ 个 Token 有多相关。
*   向量点积 $q_i \cdot k_j$ 衡量了两个向量的方向一致性。
*   $QK^T$ 得到一个 $[L, L]$ 的矩阵，称为 **Attention Score Matrix**。
*   $A_{ij}$ 表示 Query $i$ 对 Key $j$ 的关注程度。

### 第二步：缩放 (Scaling by $\frac{1}{\sqrt{d_k}}$)
**为什么要除以 $\sqrt{d_k}$？**
*   假设 $q$ and $k$ 的分量是均值为 0，方差为 1 的独立随机变量。
*   它们的点积 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 的均值为 0，但方差会变成 $d_k$。
*   当 $d_k$ 很大时（如 128），点积的值会非常大（正负都有）。
*   **梯度消失问题**: Softmax 函数对非常大的输入极其敏感，梯度会趋近于 0（饱和区）。
*   除以 $\sqrt{d_k}$ 将方差拉回 1，保证梯度流动的稳定性。

### 第三步：归一化 (Softmax)
$$ \alpha_{ij} = \frac{\exp(A_{ij})}{\sum_{k} \exp(A_{ik})} $$
*   将分数转换为概率分布，使得每一行的权重之和为 1。
*   这决定了 Query $i$ 应该从 Value $j$ 中提取多少信息。

### 第四步：加权求和 ($ \cdot V$)
$$ \text{Output}_i = \sum_{j} \alpha_{ij} v_j $$
*   根据注意力权重，聚合上下文信息。

## 3. 代码实现 (PyTorch)

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.size(-1)
    
    # 1. QK^T
    # query: [B, H, L, d_k]
    # key:   [B, H, L, d_k] -> transpose -> [B, H, d_k, L]
    scores = torch.matmul(query, key.transpose(-2, -1)) 
    
    # 2. Scaling
    scores = scores / math.sqrt(d_k)
    
    # 3. Masking (Optional)
    if mask is not None:
        # mask 为 0 的位置填入 -inf，Softmax 后变为 0
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 4. Softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # 5. Weighted Sum
    output = torch.matmul(attn_weights, value)
    
    return output, attn_weights
```

## 4. 语音模型中的应用

在语音识别（如 Whisper）中，Attention 机制的作用尤为直观：
*   **Cross-Attention**: Decoder（文本生成器）作为 Query，Encoder（音频特征）作为 Key/Value。
*   **对齐 (Alignment)**: Attention 权重图 ($\alpha_{ij}$) 实际上展示了**文本 Token 与音频时间片之间的对齐关系**。
*   如果你可视化 Whisper 的 Cross-Attention 权重，你会看到一条清晰的对角线——随着文本生成，模型关注的音频区域也在随时间向后移动。
