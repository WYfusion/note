这里的多头的头数量是一个超参数，引入不同的头可以在同一时间内从不同的角度理解输入的数据，增强特征提取的能力。头的数量 **h** 越多，每个头捕获的特征维度越少，但可以并行捕获更多特征。 ^59530b

常见的头数量一般是8个、12个、16个等等，可以根据**算力**和**需求**平衡头的数量选取。

**Multi-Head Self-Attention 对比 Self-Attention  的主要区别**：
引入了**多头机制**，将输入的特征维度拆分成多份（即多个头），**每个头内的特征分别独立进行 Self-Attention 的操作**，最后再将这些头的输出拼接起来并进行线性变换。主要**增加了一种并行化和多视角的机制**。 ^61f90e

具体过程：

设定$X  \in \mathbb{R}^{N\times L \times d_{model}}$

$N$：批量大小（batch size）。

$L$：序列长度。

$d_{model}$：输入特征的维度。

###### **特征划分为多个头**

- 假设有 **$h$** 个$head$头，将输入的特征维度 **$d_{model}$**，也即将每个$\text{token}$的维度平均分配到每个头中，每个头的特征维度为：$d_k = d_v = \frac{d_{model}}{h}$
- 通过独立的权重矩阵 **$W_i^Q$, $W_i^K$, $W_i^V$**，生成每个头的 **$Q_i$, $K_i$, $V_i$**，每个头的查询、键和值的特征维度是 **$d_k$**。注意$h$有多少，$\text{i}$就有多大。**$W_i^Q$, $W_i^K$, $W_i^V$** ,不共享权重，不存在“统一的权重矩阵”。
- 对于每一个$head$，都有各自的
$$\begin{aligned}Q_i=XW^q_i\\K_i=XW^k_i\\V_i=XW^v_i\end{aligned}$$
其中： ^0726e2
- $i \in [1,h]$
- $W^q_i,W^k_i,W^v_i∈\mathbb{R}^{d_{model}\times d_k}$ 是每个头的独立权重矩阵。矩阵形状变化：$(N\times L\times d_{model})\times (d_{model}\times d_k)=(N\times L\times d_k)$
- $Q_i, K_i, V_i \in \mathbb{R}^{N \times L \times d_k}$ 是每个头的查询、键和值。
![[Pasted image 20250314210038.png|600]]
###### **每个头独立进行 Self-Attention**

每个头在其子空间中分别计算 Self-Attention。每个头的输出维度为 **$(N, L, d_k)$**（其中 **$N$** 是批量大小，**$L$** 是序列长度，**$d_k$** 是每个头的特征维度）。
因为每个头的特征维度较低（相比于单头的 **$d_{model}$**），每个头可以专注于捕获输入序列的不同信息。
计算注意力权重：

  $A_i = \text{softmax}\left( \frac{Q_i K_i^\top}{\sqrt{d_k}} \right)$ ^572aa1
  - $Q_i \in \mathbb{R}^{N \times L \times d_k}$
  - $K_i^\top \in \mathbb{R}^{N \times d_k \times L}$
  - $A_i \in \mathbb{R}^{N \times L \times L}$是注意力权重矩阵，**表示序列中每个元素对其他元素的相关性**。
###### 加权求和值
  - $\text{head}_i = A_i V_i$
  - $A_i \in \mathbb{R}^{N \times L \times L}$
  - $V_i \in \mathbb{R}^{N \times L \times d_k}$
  - $\text{head}_i \in \mathbb{R}^{N \times L \times d_k}$
![[Pasted image 20250314212805.png|1500]] ^20ef97
###### 拼接所有头

- 将 **$h$** 个头的输出拼接到一起，得到维度为 **$(N, L, h \cdot d_k)$** 的矩阵。注意这里：$h \cdot d_k = d_{model}$
- $\text{Multi\ Head}(Q,K,V)=\text{Concat}(\text{head}_1,\text{head}_2,…,\text{head}_h)$
- $\text{Multi\ Head}\in \mathbb{R}^{N\times L \times d_{model}}$

###### **线性变换恢复特征维度**

- 最后，通过一个线性变换（权重矩阵 **$W_O$**）将拼接的结果映射回原始的特征维度 **$d_{model}$**。
- $Y=\text{Multi\ Head}(Q,K,V)\times W_O$
- $W_o \in \mathbb{R}^{d_{model}×d_{model}}$
- 输出的最终维度为 **$(N, L, d_{model})$**，与输入的特征维度一致。$X  \in \mathbb{R}^{N\times L \times d_{model}}$


## 二次注意力（Quadratic Attention）
**二次注意力**是传统 Transformer 中自注意力机制的标准实现，其核心特点是计算复杂度随输入序列长度 $L$ 呈 **二次增长**（$O(L^2 d)$），因此得名。具体来说，它通过计算所有 token 对之间的相似度矩阵（注意力矩阵），再加权聚合值向量（Value），完整保留序列中任意两个 token 的交互信息。
#### 复杂度分析
##### 矩阵乘法$QK^T$
- 得到相似度矩阵 (A)：形状为 $L \times L$，每个元素 $A_{i,j}$ 表示第 $i$ 个 token 与第 $j$ 个 token 的相似度（如注意力分数）。$A_{i,j}=\sum\limits_{k=1}^dQ_{i,k}\cdot K_{j,k}$，其中 $k$ 是特征维度索引，共需 $d$ 次乘法和 $d-1$ 次加法。相似矩阵有 $L \times L$ 个元素。
- 总乘法次数：每个元素 $d$ 次乘法 → 总乘法次数为  $L\times L\times d=L^2d$
- 加法次数：每个元素 $d-1$ 次加法 → 总加法次数为 $T^2 (d-1)$，但大 $O$ 符号忽略常数和低阶项，故简化为 $O(T^2 d)$。
硬件加速器(GPU/TPU)的**乘法操作**是计算瓶颈
