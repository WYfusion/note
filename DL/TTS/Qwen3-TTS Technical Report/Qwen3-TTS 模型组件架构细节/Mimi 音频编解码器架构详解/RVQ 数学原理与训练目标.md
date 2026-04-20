## 前置知识

> [!important]
> 
> 阅读本页前建议先读：[[DL/TTS/Qwen3-TTS Technical Report/Qwen3-TTS 模型组件架构细节/Mimi 音频编解码器架构详解/Mimi 音频编解码器架构详解|Mimi 音频编解码器架构详解]]。

---

## 0. 定位

> 本页在 Mimi 架构的基础上，**从向量量化的第一性原理**开始，系统推导 RVQ（Residual Vector Quantization）的数学形式化、训练目标、梯度估计方法，并给出最小可运行的 PyTorch 实现。

---

## 1. 变量定义表

|**符号**|**含义**|**维度**|$\mathbf{x} \in \mathbb{R}^d$|输入连续特征（一帧）|$d$ 维|
|---|---|---|---|---|---|
|$K$|RVQ 总层数|Mimi：8。Qwen3-TTS 12Hz：16|$\mathcal{C}_k = \{\mathbf{e}_k^{(i)}\}_{i=1}^V$|第 $k$ 层码本|包含 $V$ 个码字|
|$V$|每层码本大小|Mimi/Qwen：2048|$c_k \in \{0, \dots, V-1\}$|第 $k$ 层选中的码字索引|标量|
|$\hat{\mathbf{x}}_k$|第 $k$ 层量化输出|$d$ 维|$\mathbf{r}_k$|第 $k$ 层残差|$d$ 维|

---

## 2. 从 VQ 到 RVQ

### 2.1 标准向量量化（VQ）

给定输入 $\mathbf{x}$ 和码本 $\mathcal{C} = \{\mathbf{e}^{(i)}\}_{i=1}^V$，最近邻量化为：

$$c^\star = \arg\min_{i \in \{1, \dots, V\}} \|\mathbf{x} - \mathbf{e}^{(i)}\|_2^2, \quad \hat{\mathbf{x}} = \mathbf{e}^{(c^\star)}$$

**信息容量**：$\log_2 V$ 比特/帧。对于 $V = 2048$，每帧仅 **11 bit**。

### 2.2 VQ 的容量瓶颈

要达到等效 176 bit/帧的容量（12.5Hz × 16 层 × 11 bit = 2200 bit/s = 2.2 kbps），单层 VQ 需要码本大小达到：

$$V_{\text{single}} = 2^{176} \approx 10^{53}$$

这在实践上**不可行**（存储、检索、训练都崩溃）。RVQ 通过**等效笛卡尔乘积**故事解决：

$$\underbrace{2048^{16}}_{\text{16 层分别 11 bit}} = 2^{176} \approx 10^{53}$$

但将存储从 $O(V_{\text{single}} \cdot d)$ 降到 $O(K \cdot V \cdot d)$，**指数级压缩**。

---

## 3. RVQ 递归定义

> 统一使用 **0-indexed** 层号（$k = 0, 1, \dots, K-1$）与统一符号：$\mathbf{r}_k$ 表示「进入第 $k$ 层之前的残差」；$\hat{\mathbf{q}}_k$ 表示「第 $k$ 层量化器输出的码字」。

### 3.1 逐层拆分解释（以 Mimi K=8 为例）

每一层量化器只做一件事：**把当前残差投影到本层码本的最近邻码字上**，然后用「原残差 − 量化后码字」交给下一层继续处理。

> [!important]
> 
> **第 0 层**：量化原始特征 $\mathbf{x}$
> 
> $\hat{\mathbf{q}}_0 = \mathbf{e}_0^{(c_0)},\quad c_0 = \arg\min_i \|\mathbf{x} - \mathbf{e}_0^{(i)}\|_2^2,\quad \mathbf{r}_1 = \mathbf{x} - \hat{\mathbf{q}}_0$
> 
> **第 1 层**：量化残差 $\mathbf{r}_1$
> 
> $\hat{\mathbf{q}}_1 = \mathbf{e}_1^{(c_1)},\quad c_1 = \arg\min_i \|\mathbf{r}_1 - \mathbf{e}_1^{(i)}\|_2^2,\quad \mathbf{r}_2 = \mathbf{r}_1 - \hat{\mathbf{q}}_1$
> 
> **第 2 层**：量化残差 $\mathbf{r}_2$
> 
> $\hat{\mathbf{q}}_2 = \mathbf{e}_2^{(c_2)},\quad c_2 = \arg\min_i \|\mathbf{r}_2 - \mathbf{e}_2^{(i)}\|_2^2,\quad \mathbf{r}_3 = \mathbf{r}_2 - \hat{\mathbf{q}}_2$
> 
> $\vdots$
> 
> **第** $k$ **层**（一般式）：量化残差 $\mathbf{r}_k$
> 
> $\hat{\mathbf{q}}_k = \mathbf{e}_k^{(c_k)},\quad c_k = \arg\min_i \|\mathbf{r}_k - \mathbf{e}_k^{(i)}\|_2^2,\quad \mathbf{r}_{k+1} = \mathbf{r}_k - \hat{\mathbf{q}}_k$
> 
> $\vdots$
> 
> **第 7 层**（最后一层）：量化残差 $\mathbf{r}_7$
> 
> $\hat{\mathbf{q}}_7 = \mathbf{e}_7^{(c_7)},\quad \mathbf{r}_8 = \mathbf{r}_7 - \hat{\mathbf{q}}_7 \ \text{（丢弃，作为最终重建误差）}$
> 
> **最终重建**：
> 
> $\hat{\mathbf{x}} = \hat{\mathbf{q}}_0 + \hat{\mathbf{q}}_1 + \hat{\mathbf{q}}_2 + \cdots + \hat{\mathbf{q}}_7 = \sum_{k=0}^{7} \hat{\mathbf{q}}_k$

> [!important]
> 
> **一句话记住**：上一层的残差 = 下一层的输入。层和层之间沿「残差流」串联，码字和码字之间沿「重建流」并联叠加。

### 3.2 统一的递归形式

将上面的逐层展开压缩为数学递归式（初值 + 递推）：

$$\begin{aligned}  
\textbf{初值:}\quad & \mathbf{r}_0 = \mathbf{x} \\  
\textbf{量化:}\quad & c_k = \arg\min_{i \in \{0,\dots,V-1\}} \|\mathbf{r}_k - \mathbf{e}_k^{(i)}\|_2^2 && \text{// 第 \(k\) 层最近邻索引} \\  
& \hat{\mathbf{q}}_k = \mathbf{e}_k^{(c_k)} && \text{// 第 \(k\) 层码字} \\  
\textbf{残差更新:}\quad & \mathbf{r}_{k+1} = \mathbf{r}_k - \hat{\mathbf{q}}_k && \text{// 传给下一层} \\  
\textbf{范围:}\quad & k = 0, 1, \dots, K-1 && \text{// Mimi \(K\!=\!8\)，Qwen3-TTS 12Hz \(K\!=\!16\)}  
\end{aligned}$$

### 3.3 步骤级关系表（K=8）

|**步骤**|**输入**|**操作**|**码字输出**|**残差输出**|**语义角色**|
|---|---|---|---|---|---|
|Layer 0|$\mathbf{r}_0 = \mathbf{x}$|在 $\mathcal{C}_0$ 找最近邻|$\hat{\mathbf{q}}_0$|$\mathbf{r}_1 = \mathbf{x} - \hat{\mathbf{q}}_0$|语义主骨（WavLM 蒸馏）|
|Layer 1|$\mathbf{r}_1$|在 $\mathcal{C}_1$ 找最近邻|$\hat{\mathbf{q}}_1$|$\mathbf{r}_2 = \mathbf{r}_1 - \hat{\mathbf{q}}_1$|主要声学（F0、共振峰）|
|Layer 2|$\mathbf{r}_2$|在 $\mathcal{C}_2$ 找最近邻|$\hat{\mathbf{q}}_2$|$\mathbf{r}_3 = \mathbf{r}_2 - \hat{\mathbf{q}}_2$|能量包络|
|Layer 3|$\mathbf{r}_3$|在 $\mathcal{C}_3$ 找最近邻|$\hat{\mathbf{q}}_3$|$\mathbf{r}_4 = \mathbf{r}_3 - \hat{\mathbf{q}}_3$|韵律、节奏残差|
|Layer 4|$\mathbf{r}_4$|在 $\mathcal{C}_4$ 找最近邻|$\hat{\mathbf{q}}_4$|$\mathbf{r}_5 = \mathbf{r}_4 - \hat{\mathbf{q}}_4$|高频谐波|
|Layer 5|$\mathbf{r}_5$|在 $\mathcal{C}_5$ 找最近邻|$\hat{\mathbf{q}}_5$|$\mathbf{r}_6 = \mathbf{r}_5 - \hat{\mathbf{q}}_5$|气口、摩擦音|
|Layer 6|$\mathbf{r}_6$|在 $\mathcal{C}_6$ 找最近邻|$\hat{\mathbf{q}}_6$|$\mathbf{r}_7 = \mathbf{r}_6 - \hat{\mathbf{q}}_6$|微细声学细节|
|Layer 7|$\mathbf{r}_7$|在 $\mathcal{C}_7$ 找最近邻|$\hat{\mathbf{q}}_7$|$\mathbf{r}_8$（丢弃）|环境/噪声尾部|
|**Output**|—|**求和**|$\hat{\mathbf{x}} = \sum_{k=0}^{7} \hat{\mathbf{q}}_k$|$\\|\mathbf{x}-\hat{\mathbf{x}}\\|_2 = \\|\mathbf{r}_8\\|_2$|重建输出|

### 3.4 完整数据流图

![[2026-04-18 10.04.12RVQ完整数据流图.excalidraw|800]]

横向红箭头（残差流）代表 $\mathbf{r}_k \to \mathbf{r}_{k+1}$ 的传递；纵向蓝箭头（重建流）代表每层 $\hat{\mathbf{q}}_k$ 向总和节点汇聚。两条流正交就构成了 RVQ 的全部计算。

### 3.5 残差的伸缩与重建收敛

定义第 $k$ 层的前 $k$ 项部分和：

$$\mathbf{s}_k \triangleq \sum_{j=0}^{k} \hat{\mathbf{q}}_j,\qquad k = 0, 1, \dots, K-1$$

由残差递推 $\mathbf{r}_{k+1} = \mathbf{r}_k - \hat{\mathbf{q}}_k$ 逐项累加可得：

$$\mathbf{r}_{k+1} = \mathbf{x} - \mathbf{s}_k$$

所以**第** $k$ **步之后的残差就是「真值减去目前重建」**。理想情况下，每层码本能把 $\|\mathbf{r}_k\|_2$ 压一半数量级，因此残差范数随层数近似**指数衰减**：

$$\|\mathbf{r}_k\|_2 \ \lesssim\ \alpha^k \cdot \|\mathbf{x}\|_2,\qquad \alpha \in (0, 1)$$

![[2026-04-18 10.05.56RQV残差范数衰减.excalidraw|1000]]

> [!important]
> 
> **结论**：初时层承担了主要能量，开发者只要让 LM 预测 $\hat{\mathbf{q}}_0$ 就能拿到**绝大部分信息**；后续层用 MTP 预测小残差即可。这也是 Qwen3-TTS 「单 LM 预测语义 + MTP 并行补开声学」设计的理论依据。

### 3.6 重建等价性的一行证明

从递推式 $\mathbf{r}_{k+1} = \mathbf{r}_k - \hat{\mathbf{q}}_k$ 依次代入：

$$\mathbf{r}_K \;=\; \mathbf{r}_{K-1} - \hat{\mathbf{q}}_{K-1} \;=\; \mathbf{r}_{K-2} - \hat{\mathbf{q}}_{K-2} - \hat{\mathbf{q}}_{K-1} \;=\; \cdots \;=\; \mathbf{x} - \sum_{k=0}^{K-1} \hat{\mathbf{q}}_k$$

整理即得 RVQ 重建恒等式：

$$\boxed{\ \hat{\mathbf{x}} \;=\; \sum_{k=0}^{K-1} \hat{\mathbf{q}}_k \;=\; \mathbf{x} - \mathbf{r}_K\ }$$

**重建误差**精确等于最后一层未被量化的残差：$\|\mathbf{x} - \hat{\mathbf{x}}\|_2 = \|\mathbf{r}_K\|_2$。这也解释了**为什么增加层数** $K$ **可以单调降低重建误差**。

### 3.7 正交性直觉

![[2026-04-18 10.07.27RQV正交性直觉.excalidraw]]

> [!important]
> 
> **核心直觉**：每一层都在**正交分解**输入信号的残差空间。第 0 层捕获最大方差方向（语义），后续层逐次捕获更细微的残差方向（声学细节）。这与 PCA 的主成分分解在目标上类似，但量化替代了线性投影。

---

## 4. 训练目标

### 4.1 重建损失

最基础的重建损失为：

$$\mathcal{L}_{\text{rec}} = \|\mathbf{x} - \hat{\mathbf{x}}\|_2^2 = \left\| \mathbf{x} - \sum_{k=1}^{K} \mathbf{e}_k^{(c_k)} \right\|_2^2$$

### 4.2 Codebook 损失与 Commitment 损失（VQ-VAE 传承）

类似 VQ-VAE（van den Oord et al., 2017），每层要训练两个辅助损失：

$$\begin{aligned}  
\mathcal{L}_{\text{codebook}}^{(k)} &= \| \text{sg}[\mathbf{r}_{k-1}] - \mathbf{e}_k^{(c_k)} \|_2^2 \quad (\text{更新码本}) \\  
\mathcal{L}_{\text{commit}}^{(k)} &= \beta \| \mathbf{r}_{k-1} - \text{sg}[\mathbf{e}_k^{(c_k)}] \|_2^2 \quad (\text{约束编码器})  
\end{aligned}$$

其中 $\text{sg}[\cdot]$ 表示 stop-gradient（停止梯度传播），$\beta \approx 0.25$。

> [!important]
> 
> **为什么需要 stop-gradient？**
> 
> 码本损失仅更新码本，不要回传梯度到编码器；承诺损失仅更新编码器，不要更新码本。否则码本会退化到输入分布的均值，输入也会崩塌到码本点。

### 4.3 Straight-Through Estimator（STE）

#### 4.3.1 问题的起源：离散操作阻断梯度

VQ / RVQ 的核心算子是 $\arg\min$：

$$c^\star = \arg\min_{i \in \{1,\dots,V\}} \|\mathbf{x} - \mathbf{e}^{(i)}\|_2^2, \qquad \hat{\mathbf{x}} = \mathbf{e}^{(c^\star)}$$

从 $\mathbf{x}$ 到 $\hat{\mathbf{x}}$ 是一个**分段常数函数**：在码本 Voronoi 胞腔内部，输出完全恒定；在胞腔边界上发生跳变。因此：

$$\frac{\partial \hat{\mathbf{x}}}{\partial \mathbf{x}} =  
\begin{cases}  
\mathbf{0}, & \mathbf{x} \text{ 在 Voronoi 胞腔内部（几乎处处）} \\  
\text{未定义}, & \mathbf{x} \text{ 在胞腔边界上（零测度）}  
\end{cases}$$

> [!important]
> 
> 如果直接用链式法则反向传播，编码器 $E_\theta$ 收到的梯度**几乎处处为零**，训练无法进行。这和 Sign / Step / Threshold / One-hot 采样等所有离散算子遇到的是**同一个问题**，最早由 Hinton 在 2012 Coursera 课程上提出「直通」启发式作为应对。

#### 4.3.2 STE 的定义：identity surrogate for backward

STE（Bengio et al., 2013）的核心思想只有一句话：

> **前向照常量化，反向时把量化当作恒等映射。**

形式上引入 stop-gradient 算子 $\text{sg}[\cdot]$（PyTorch 里就是 `.detach()`），定义：

$$\boxed{\ \hat{\mathbf{x}}_{\text{STE}} \;=\; \mathbf{x} + \text{sg}\!\left[\,\hat{\mathbf{x}} - \mathbf{x}\,\right]\ }$$

- **前向**：$\text{sg}$ 不改数值，$\hat{\mathbf{x}}_{\text{STE}} = \mathbf{x} + (\hat{\mathbf{x}} - \mathbf{x}) = \hat{\mathbf{x}}$，和不加 STE 的结果完全一致。

- **反向**：$\text{sg}$ 把括号里那一项的梯度置零，于是

$$\frac{\partial \hat{\mathbf{x}}_{\text{STE}}}{\partial \mathbf{x}} \;=\; \frac{\partial \mathbf{x}}{\partial \mathbf{x}} + \underbrace{\frac{\partial\, \text{sg}[\cdot]}{\partial \mathbf{x}}}_{=\ \mathbf{0}} \;=\; \mathbf{I}$$

也就是说，来自上游（如重建损失）的梯度 $\nabla_{\hat{\mathbf{x}}} \mathcal{L}$ 被**原封不动**地传给编码器：$\nabla_{\mathbf{x}} \mathcal{L} = \nabla_{\hat{\mathbf{x}}} \mathcal{L}$。

#### 4.3.3 图解：前向与反向两个计算图

![[2026-04-18 10.08.06直通估计器STE-前后向计算图.excalidraw|1200]]

**一个关键观察**：STE 让**前向图**和**反向图**不再是同一张图。这是一种「有偏但有用」的梯度估计，牺牲严格正确性来换取可训练性。

#### 4.3.4 为什么「有偏但有用」？

数学上，STE 给出的是 $\nabla_\mathbf{x} \mathcal{L}$ 的**有偏估计量**。但它有一个非常友好的几何解释：

> [!important]
> 
> **一阶近似视角**：在 Voronoi 胞腔内部，$\hat{\mathbf{x}} \approx \mathbf{x} + \boldsymbol{\epsilon}$，其中 $\boldsymbol{\epsilon} = \mathbf{e}^{(c^\star)} - \mathbf{x}$ 是量化误差。如果把量化看作「加一个与 $\mathbf{x}$ 弱相关的小扰动」，则 $\partial \hat{\mathbf{x}} / \partial \mathbf{x} \approx \mathbf{I}$ 就是这个扰动视角下的一阶近似。只要量化误差足够小（码本足够稠密），STE 的偏差就足够可控。

理论上，Yin et al.（ICLR 2019）的 _Understanding Straight-Through Estimator in Training Activation Quantized Neural Nets_ 证明了：在一定假设下，STE 方向与真实下降方向的夹角小于 90°，因此**仍是一个下降方向**，这保证了 SGD 的收敛性。

#### 4.3.5 STE 与 Commitment Loss 的分工

STE 只解决了「编码器收不到梯度」的问题，**没有**保证 $\mathbf{x}$ 靠近码本。设想一种退化情形：编码器把 $\mathbf{x}$ 输出到离所有码字都很远的区域，重建损失依然能通过 STE 给编码器回传梯度，但量化误差 $\|\mathbf{x} - \hat{\mathbf{x}}\|$ 会持续变大。承诺损失正好弥补这个缺陷：

$$\mathcal{L}_{\text{commit}} = \beta\,\|\mathbf{x} - \text{sg}[\hat{\mathbf{x}}]\|_2^2$$

它显式惩罚 $\mathbf{x}$ 远离最近码字的情况，迫使编码器输出「承诺」留在码本附近。二者结合后的总梯度方向是：

$$\nabla_{\mathbf{x}} \mathcal{L}_{\text{total}} = \underbrace{\nabla_{\hat{\mathbf{x}}} \mathcal{L}_{\text{rec}}}_{\text{STE 传递}} + \underbrace{2\beta(\mathbf{x} - \hat{\mathbf{x}})}_{\text{commitment 拉向码本}}$$

#### 4.3.6 PyTorch 一行实现

```python
# x:     编码器输出，requires_grad=True
# x_hat: 最近邻码字（gather 得到），不带梯度
x_hat_ste = x + (x_hat - x).detach()
# 前向数值 = x_hat；反向时 ∂x_hat_ste/∂x = I
```

等价写法（语义完全一致，有些实现更喜欢第二种）：

```python
x_hat_ste = x_hat.detach() + x - x.detach()
```

#### 4.3.7 STE 家族的扩展

|**变体**|**反向替代函数** $g'(\mathbf{x})$|**典型场景**|
|---|---|---|
|Vanilla STE|$\mathbf{I}$（恒等）|VQ-VAE、RVQ、BinaryNet|
|Clipped STE|$\mathbf{I} \cdot \mathbb{1}[\|\mathbf{x}\| \le 1]$|1-bit 量化（BinaryConnect）|
|Soft STE|$\text{tanh}'(\mathbf{x})$ 或 sigmoid 导数|激活值量化|
|Gumbel-Softmax|温度退火的可微近似|离散采样（非 argmin）|
|Rotation Trick|用旋转矩阵替代 identity，保持范数|Fifty et al. 2024，改进 VQ-VAE|

#### 4.3.8 在 Qwen3-TTS 12Hz Tokenizer 里的具体位置

![[2026-04-18 10.10.02STE在 Qwen3-TTS 12Hz Tokenize.excalidraw|800]]

- 第 0 层（语义）：STE 将 WavLM 蒸馏损失的梯度回传到 Encoder。

- 第 1–15 层（声学）：STE 将 Mel / GAN / FM 损失的梯度回传到 Encoder。

- 所有 16 层码本本身用 **EMA 更新**，不走 STE 通路。

> [!important]
> 
> **实现 checklist**
> 
> 1. 前向取 `x_hat = codebook[idx]`，务必配合 `x_hat_st = x + (x_hat - x).detach()` 使用。
> 
> 1. commitment loss 必须加，否则编码器发散。
> 
> 1. 多层 RVQ 时，对每一层的残差 `r = r - x_hat_k.detach()`，**切断层间梯度耦合**。
> 
> 1. 码本用 EMA buffer，不放入 `nn.Parameter`，避免优化器二次更新。

### 4.4 RVQ 完整损失

$$\mathcal{L}_{\text{RVQ}} = \mathcal{L}_{\text{rec}} + \sum_{k=1}^{K} \left( \mathcal{L}_{\text{codebook}}^{(k)} + \mathcal{L}_{\text{commit}}^{(k)} \right)$$

### 4.5 码本更新的 EMA 变体

实践中更常用**指数滑动平均（EMA）**更新码本代替 $\mathcal{L}_{\text{codebook}}$，更稳定：

$$\begin{aligned}  
N_k^{(i)}(t) &= \gamma N_k^{(i)}(t-1) + (1-\gamma) \mathbb{1}[c_k = i] \\  
\mathbf{m}_k^{(i)}(t) &= \gamma \mathbf{m}_k^{(i)}(t-1) + (1-\gamma) \mathbb{1}[c_k = i] \cdot \mathbf{r}_{k-1} \\  
\mathbf{e}_k^{(i)} &\leftarrow \frac{\mathbf{m}_k^{(i)}(t)}{N_k^{(i)}(t) + \varepsilon}  
\end{aligned}$$

其中 $\gamma \approx 0.99$。这个更新规则本质是**流式 K-means**。

---

## 5. 语义蒸馏损失（Qwen3-TTS 12Hz 第 0 层专属）

第 0 层要编码语义，用 WavLM 作为教师模型添加蒸馏损失：

$$\mathcal{L}_{\text{sem}} = 1 - \cos\left( \phi_{\text{WavLM}}(\mathbf{x}), \ \psi(\hat{\mathbf{x}}_0) \right)$$

- $\phi_{\text{WavLM}}$：WavLM 特征提取器（冻结）

- $\psi$：码本特征到 WavLM 空间的投影头（可学习）

- 强迫第 0 层量化结果的语义特征与 WavLM 对齐

---

## 6. 完整训练损失

Qwen3-TTS 12Hz Tokenizer 的全部损失：

$$\mathcal{L} = \underbrace{\lambda_1 \mathcal{L}_{\text{mel}}}_{\text{多尺度 Mel}} + \underbrace{\lambda_2 \mathcal{L}_{\text{adv}}}_{\text{GAN}} + \underbrace{\lambda_3 \mathcal{L}_{\text{fm}}}_{\text{特征匹配}} + \underbrace{\lambda_4 \mathcal{L}_{\text{RVQ}}}_{\text{量化}} + \underbrace{\lambda_5 \mathcal{L}_{\text{sem}}}_{\text{语义蒸馏}}$$

|**损失项**|**作用**|**典型权重**|$\mathcal{L}_{\text{mel}}$|时频一致性|45|
|---|---|---|---|---|---|
|$\mathcal{L}_{\text{adv}}$|自然度|1|$\mathcal{L}_{\text{fm}}$|判别器特征匹配|2|
|$\mathcal{L}_{\text{RVQ}}$|量化器优化|0.25|$\mathcal{L}_{\text{sem}}$|语义解耦|1|

---

## 7. RVQ 的最小 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """单层 VQ，训练时使用 EMA 更新码本 + STE。"""
    def __init__(self, codebook_size: int, dim: int, ema_decay: float = 0.99):
        super().__init__()
        self.V, self.d, self.gamma = codebook_size, dim, ema_decay
        # 码本以 buffer 存储（不走常规梯度，由 EMA 更新）
        self.register_buffer("codebook", torch.randn(codebook_size, dim))
        self.register_buffer("N", torch.zeros(codebook_size))
        self.register_buffer("m", torch.zeros(codebook_size, dim))

    def forward(self, x: torch.Tensor):
        # x: (B, d)
        # 计算距离并找最近邻
        dist = (x.pow(2).sum(-1, keepdim=True)
                - 2 * x @ self.codebook.t()
                + self.codebook.pow(2).sum(-1))
        idx = dist.argmin(-1)                        # (B,)
        x_hat = self.codebook[idx]                   # (B, d)

        if self.training:
            # EMA 更新码本
            onehot = F.one_hot(idx, self.V).float()  # (B, V)
            with torch.no_grad():
                self.N.mul_(self.gamma).add_(onehot.sum(0), alpha=1 - self.gamma)
                self.m.mul_(self.gamma).add_(onehot.t() @ x, alpha=1 - self.gamma)
                self.codebook.copy_(self.m / (self.N.unsqueeze(-1) + 1e-5))

        # STE：前向用 x_hat，反向梯度直接给 x
        x_hat_st = x + (x_hat - x).detach()
        commit_loss = F.mse_loss(x, x_hat.detach())
        return x_hat_st, idx, commit_loss


class ResidualVQ(nn.Module):
    def __init__(self, K: int, codebook_size: int, dim: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantizer(codebook_size, dim) for _ in range(K)]
        )

    def forward(self, x: torch.Tensor):
        # x: (B, d)
        r = x
        x_hat_total = torch.zeros_like(x)
        idxs, losses = [], 0.0
        for vq in self.layers:
            x_hat_k, idx_k, loss_k = vq(r)
            x_hat_total = x_hat_total + x_hat_k
            r = r - x_hat_k.detach()                 # 残差切断梯度避免层间耦合
            idxs.append(idx_k)
            losses = losses + loss_k
        return x_hat_total, torch.stack(idxs, -1), losses
```

> [!important]
> 
> **关键实现细节**：残差更新时对 `x_hat_k` 调用 `.detach()` 非常重要。不这样做的话，低层的梯度会通过残差通路反向影响高层码本，造成层间训练耦合和崩塌。

---

## 8. 容量与失真权衡

### 8.1 比特率计算

$$\text{bitrate} = K \cdot \log_2 V \cdot \text{FPS}$$

- Qwen3-TTS 12Hz: $16 \times 11 \times 12.5 = 2200$ bit/s = **2.2 kbps**

- Mimi: $8 \times 11 \times 12.5 = 1100$ bit/s = **1.1 kbps**

### 8.2 层数与重建质量的经验规律

$$\text{PESQ}(K) \approx \text{PESQ}_\infty - \alpha \cdot e^{-\beta K}$$

重建质量随层数**指数收敛**。Qwen3-TTS 选 16 层是在边际收益变小之前的好权衡点：

- 8 层 → 16 层 的 PESQ 提升明显（Mimi 2.88 → Qwen 3.21）

- 16 层 → 32 层 收益极小，但比特率翻倍，不划算

---

## 9. 与其他量化方法的对比

|**方法**|**原理**|**优势**|**劣势**|单层 VQ|最近邻查表|简单|容量受码本大小限|
|---|---|---|---|---|---|---|---|
|**RVQ**|**层级残差量化**|**指数级容量、结构层级化**|**训练需层间协调**|FSQ|有限标量量化|无码本塌缩|容量低|
|GQ|分组量化|平行化友好|层级化不明显|LFQ|Lookup-Free Quantization|无查表、极高容量|实践不成熟|

---

## 10. Qwen3-TTS 12Hz 的层级角色分配

![[2026-04-18 10.12.52Qwen3-TTS 12Hz 的RQV层级角色分配.excalidraw|200]]

> [!important]
> 
> **层级领域的工程启示**：LM 预测第 0 层就抓住了说啥的关键；MTP 预测第 1–15 层属于「漆油工」。两者分工合理、训练友好、推理快速。

---

## 延伸阅读

- [[DL/TTS/Qwen3-TTS Technical Report/Qwen3-TTS 模型组件架构细节/Mimi 音频编解码器架构详解/Mimi 音频编解码器架构详解|Mimi 音频编解码器架构详解]]

- [[论文库/Qwen3-TTS Technical Report/Qwen3-TTS 模型组件架构细节/MTP 模块（Multi-Token Prediction）详解|MTP 模块（Multi-Token Prediction）详解]]：RVQ 分层预测的下游消费者

---

## 参考文献

1. van den Oord et al. _Neural Discrete Representation Learning_ (VQ-VAE). NeurIPS 2017.

1. Zeghidour et al. _SoundStream: An End-to-End Neural Audio Codec_. IEEE/ACM TASLP, 2022—RVQ 在音频上的开创性应用。

1. Défossez et al. _High Fidelity Neural Audio Compression_ (EnCodec). 2022.

1. Mentzer et al. _Finite Scalar Quantization: VQ-VAE Made Simple_ (FSQ). 2023.

1. Yu et al. _Language Model Beats Diffusion — Tokenizer is Key to Visual Generation_ (LFQ). 2023.