[[2202.10447] Transformer Quality in Linear Time](https://arxiv.org/abs/2202.10447)

---
**门控注意力单元**结合了**门控线性单元GLU**和[[Multi-Head Self-Attention]]
![[Pasted image 20250511215424.png|800]]
# 门控线性单元
## Vanilla MLP 基本多层感知机
输入张量$X\in\mathbb{R}^{T\times d}$，$X$ 表示输入序列的特征矩阵，其中包含 $T$ 个 $\text{tokens}$（Token 数量），每个 token 的特征维度为 d（模型维度隐藏层大小，即每个 token 的向量表示维度）。
输出张量$O=\phi(XW_u)W_o$，**逐元素激活函数**（如 ReLU、GELU 等），用于引入非线性变换。
权重矩阵 $W_u$ 形状为$\mathbb{R}^{d\times e}$，将输入张量投射到更高的 $e$ 维度。扩展特征空间，允许 MLP 捕捉更复杂的模式。
权重矩阵 $W_o$ 形状为$\mathbb{R}^{e\times d}$，将中间张量投射到原始的 $d$ 维度。与输入维度保持一致，便于后续层（如残差连接）处理。
## GLU门控线性单元主体
GLU是作为增强了门控的改进型多层感知器 (MLP)
$$\begin{aligned}&U=\phi_{u}(XW_{u}),\quad V=\phi_{v}(XW_{v})&&\in\mathbb{R}^{T\times e}\\&O=(U\odot V)W_{o}&&\in\mathbb{R}^{T\times d}\end{aligned}$$
其中 $⊙$ 表示逐元素乘法。在 GLU 中，每个表示 $u_i$ 都受到与同一 token 关联的另一个表示 $v_i$ 的门控。$w_u$、$w_v$、$w_o$的逻辑和尺寸同Vanilla MLP中的一致。

# 门控注意单元容许使用比MHSA更简单的更弱的注意力机制作为前置
关键思想是将注意力机制和 GLU 统一为一个层，并尽可能地共享它们的计算，这不仅可以提高参数/计算效率，还能自然地实现强大的注意力门控机制。具体来说，GAU 将公式 $O=(U\odot V)W_{o} \ , O \in\mathbb{R}^{T\times d}$ 做了推广，形成了$O=(U\odot\widehat{V})W_o\quad\mathrm{where}\quad\widehat{V}=AV$，这里涉及的$A\in\mathbb{R}^{T\times T}$ 包含词元间的注意力权重。与 GLU 始终使用 $v_i$ 来门控 $u_i$ （两者均与同一词元相关）不同，GAU 将 $v_i$ 替换为一个可能更相关的表示 $\hat{v}_i=\sum\limits_ja_{ij}v_j$ , 该表示使用注意力机制从所有可用 $\text{tokens}$ 中“检索”而来。当 A 为单位矩阵时，上述过程将简化为 GLU。
门控的存在允许使用比 MHSA ![[Multi-Head Self-Attention#^572aa1]] 更简单/更弱的注意力机制，而不会造成质量损失
$$Z=\phi_{z}(XW_{z})\in\mathbb{R}^{T\times s} \ \ \ \ \  A=\mathrm{relu}^{2}\left(\mathrm{Q}(Z)\mathrm{K}(Z)^{\top}+b\right)\in\mathbb{R}^{T\times T}$$ ^55baef
# 混合块注意力
提出了混合块注意力机制 ，它融合了部分注意力机制和线性注意力机制的优点。但是同年就已经不是很优了。
![[Pasted image 20250512201224.png|500]]
混合块注意力是论文提出的核心创新模块，旨在解决长序列 Transformer 的效率问题。它通过**分块策略**将序列划分为局部块，结合**块内二次注意力**（精确捕捉局部依赖）和**块间线性注意力**（高效建模长距离依赖），实现了线性复杂度与高性能的统一。
## 预处理：分块与特征生成
### 输入形状
输入序列 $x$ 形状为 $[B, T, d]$（批次、序列长度、模型维度），分块后为 $[B, G, C, d]$，其中 $G=T/C$ 为块数，$C$ 为块长（默认 $256$）。
### 特征生成
对每个块 g，通过 GAU 的线性变换生成：
- $U_g$, $V_g \in \mathbb{R}^{C \times e}$（门控机制的输入，$e=2d$ 为扩展维度），
- $Z_g \in \mathbb{R}^{C \times s}$（共享表示，$s=128$ 为缩减维度）， 再通过轻量变换（缩放 + 偏移）生成 4 组查询 / 键：$Q_g^{\text{quad}}, K_g^{\text{quad}}$（块内二次注意力）和 $Q_g^{\text{lin}}, K_g^{\text{lin}}$（块间线性注意力）。
## 块内二次注意力
使用门控注意力，对于每一个块内Token计算其交互：$\hat{V}_g^\mathrm{quad}=\mathrm{relu}^2\left(Q_g^\mathrm{quad}{K_g^{\mathrm{quad}^{\top}}}+b\right)V_g$，$Q_{g}^{\mathrm{quad}}K_{g}^{\mathrm{quad}^{\top}}\in\mathbb{R}^{C\times C}$,相对位置偏差 b（RoPE 编码）
复杂度：单块复杂度 $O(C^2 d)$，总复杂度 $O(G C^2 d) = O(T C d)$（线性于 $T$，因 $C$ 固定）。
## 块间线性注意力
对块 $g$，聚合所有前序块（因果场景）或所有块（非因果）的键值对：$$\begin{aligned}\text{Non-Causal:}&\hat{V}_g^{\mathrm{lin}}=Q_g^{\mathrm{lin}}\left(\sum_{h=1}^GK_h^{\mathrm{lin}\top}V_h\right), && \text{（非因果场景，如BERT）} \\\mathrm{Causal:}&\hat{V}_g^{\mathrm{lin}}=Q_g^{\mathrm{lin}}\left(\sum_{h=1}^{g-1}K_h^{\mathrm{lin}\top}V_h\right).&& \text{（因果场景，如自回归）} \end{aligned}$$
将传统线性注意力的 T 步串行累加（每步处理 1 个 token）转化为 G 步块级累加（每步处理 1 个块），串行步数减少 C 倍（如 T=8K, C=256 时，步数从 8192→32）。
### 内存与计算资源的节省
缓存$K_h^{\mathrm{lin}^\top}V_h\in\mathbb{R}^{s\times e}$(低维矩阵，$s=128$)，内存占用远低于全注意力的 $T \times T$ 矩阵
## 门控输出
### 门控机制
将局部和全局输出相加后，与 $GAU$ 的门控向量 $U_g$ 逐元素相乘（类似 $GLU$ 的门控机制）：$O_g=\left[U_g\odot\left(\hat{V}_g^\mathrm{quad}+\hat{V}_g^\mathrm{lin}\right)\right]W_o$ ，这里的$W_o\in\mathbb{R}^{e\times d}$，为投影矩阵，将维度恢复为 $d$



