# Word Embedding 回顾

Token ID 只是一个整数（如 `1024`），神经网络无法直接计算整数。我们需要将其映射为稠密的实数向量（Embedding）。

## 1. One-hot vs Distributed Representation

### 1.1 One-hot Encoding
*   向量维度等于词表大小 $V$。
*   只有第 $i$ 位是 1，其余为 0。
*   **缺点**: 维度灾难，无法衡量词之间的相似度（任意两个 One-hot 向量正交）。

### 1.2 Distributed Representation (Embedding)
*   将每个词映射到低维空间 $d_{model}$ (如 512, 4096)。
*   **Lookup Table**: 本质上是一个 $V \times d_{model}$ 的矩阵 $W_E$。
*   获取 Token $i$ 的向量等于取出矩阵的第 $i$ 行。

## 2. 静态 vs 动态 Embedding

### 2.1 静态 Embedding (Word2Vec, GloVe)
*   训练完成后，每个词的向量是固定的。
*   **问题**: 多义词无法区分。 "Bank" (银行) 和 "Bank" (河岸) 是同一个向量。

### 2.2 动态 Embedding (Contextualized)
*   BERT / GPT 的 Embedding 层输出只是初始状态。
*   经过 Transformer 层层计算后，输出的 Hidden State 是**上下文相关**的。
*   在顶层，"Bank" 在不同语境下的向量表示是完全不同的。

## 3. LLM 中的 Embedding 层

在 PyTorch 中实现非常简单：

```python
import torch.nn as nn

# vocab_size=32000, dim=4096
embedding = nn.Embedding(32000, 4096)

input_ids = torch.LongTensor([1, 500, 1024])
vectors = embedding(input_ids) 
# shape: [3, 4096]
```

### 3.1 权重绑定 (Weight Tying)
许多模型（如 GPT-2, Llama）在**输入 Embedding 层**和**输出 Softmax 层**共享权重矩阵。
*   原因：输入和输出的词表是同一个，语义空间应该一致。
*   好处：减少参数量，防止过拟合。

### 3.2 缩放 (Scaling)
在 Transformer 中，Embedding 向量通常会乘以 $\sqrt{d_{model}}$。
*   原因：保持加法操作（Embedding + Positional Encoding）后的方差稳定。
