https://doi.org/10.48550/arXiv.2305.13245

图1![[Pasted image 20251227215047.png|800]]
图二![[Pasted image 20251227214523.png|800]]
**多头注意力机制**（MHA）包含 H 个查询头、键头和值头。
**分组查询注意力机制**（GQA）则为每个查询头组共享同一个键头和值头，介于多头注意力机制和多查询注意力机制之间。
**多查询注意力机制**（MQA）在所有查询头之间共享同一个键头和值头。

### 多头注意力机制 (MHA)
详细推导见：[[Multi-Head Self-Attention]]
采用多头注意力机制（MHA）的语言模型检查点可以通过升级训练（Komatsuzaki et al., 2022）来使用多查询注意力机制（MQA），且仅需少量原始训练计算资源。

### 多查询注意力机制 (Multi-Query Attention, MQA)
MQA 是 MHA 的一种变体，旨在减少推理过程中的内存带宽需求（特别是 KV Cache 的大小）。在 MQA 中，所有的查询头（Query Heads）共享**同一组**键（Key）和值（Value）头。

#### 数学推导过程
设定输入 $X \in \mathbb{R}^{N \times L \times d_{model}}$。
假设有 $h$ 个查询头，每个头的维度为 $d_k = \frac{d_{model}}{h}$。

1.  **线性变换生成 Q, K, V**
    -   **查询 (Query)**: 与 MHA 一样，保持 $h$ 个独立的投影矩阵。
        $$Q_i = X W^Q_i, \quad i \in [1, h]$$
        其中 $W^Q_i \in \mathbb{R}^{d_{model} \times d_k}$。
    -   **键 (Key) 和 值 (Value)**: 不同于 MHA，MQA 只有**一个**键投影矩阵和**一个**值投影矩阵。
        $$K = X W^K$$
        $$V = X W^V$$
        其中 $W^K, W^V \in \mathbb{R}^{d_{model} \times d_k}$。
        注意：这里 $K, V \in \mathbb{R}^{N \times L \times d_k}$，不再有下标 $i$。

2.  **计算注意力分数**
    对于每一个查询头 $i$，它与**共享**的 $K$ 进行交互：
    $$A_i = \text{softmax}\left( \frac{Q_i K^\top}{\sqrt{d_k}} \right)$$
    -   $Q_i \in \mathbb{R}^{N \times L \times d_k}$
    -   $K^\top \in \mathbb{R}^{N \times d_k \times L}$ (广播机制：所有 $Q_i$ 复用同一个 $K$)
    -   $A_i \in \mathbb{R}^{N \times L \times L}$

3.  **加权求和**
    使用计算出的注意力权重 $A_i$ 对**共享**的 $V$ 进行加权：
    $$\text{head}_i = A_i V$$
    -   $V \in \mathbb{R}^{N \times L \times d_k}$ (广播机制：所有 $head_i$ 复用同一个 $V$)

4.  **拼接与输出**
    $$\text{MultiQuery}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W_O$$
    -   拼接后的维度为 $N \times L \times (h \cdot d_k) = N \times L \times d_{model}$。
    -   $W_O \in \mathbb{R}^{d_{model} \times d_{model}}$。

**优点**：可以显著降低加载键值对所需的内存带宽，它使用多个查询头，但每个键和值都使用单个头。
**缺点**：多查询注意力机制（MQA）可能导致质量下降和训练不稳定，而且训练针对质量和推理分别优化的独立模型可能并不现实。

### 分组查询注意力机制 (Grouped-Query Attention, GQA)

^21f310

GQA 是 MHA 和 MQA 的折中方案。它将 $h$ 个查询头分成 $g$ 个组（Group），每个组内的查询头共享同一组 K 和 V。
当 $g=1$ 时，GQA 退化为 MQA；当 $g=h$ 时，GQA 等同于 MHA。

#### 数学推导过程
设定输入 $X \in \mathbb{R}^{N \times L \times d_{model}}$。
总查询头数为 $h$，组数为 $g$。
每个组包含的查询头数量为 $m = h / g$。

1.  **线性变换生成 Q, K, V**
    -   **查询 (Query)**: 保持 $h$ 个独立的投影矩阵。
        $$Q_i = X W^Q_i, \quad i \in [1, h]$$
    -   **键 (Key) 和 值 (Value)**: 共有 $g$ 个独立的键值对头。
        $$K_j = X W^K_j, \quad j \in [1, g]$$
        $$V_j = X W^V_j, \quad j \in [1, g]$$
        其中 $W^K_j, W^V_j \in \mathbb{R}^{d_{model} \times d_k}$。

2.  **分组映射**
    对于第 $i$ 个查询头，它属于第 $j$ 个组，其中 $j = \lceil \frac{i}{m} \rceil$ (假设索引从1开始)。
    或者表示为映射函数 $G(i)$ 指示查询头 $i$ 对应的 KV 组索引。

3.  **计算注意力分数**
    查询头 $Q_i$ 与其所属组对应的 $K_{G(i)}$ 进行交互：
    $$A_i = \text{softmax}\left( \frac{Q_i K_{G(i)}^\top}{\sqrt{d_k}} \right)$$

4.  **加权求和**
    $$\text{head}_i = A_i V_{G(i)}$$

5.  **拼接与输出**
    $$\text{GroupedQuery}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W_O$$


