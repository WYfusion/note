长短期记忆网络的设计灵感来自于计算机的逻辑门。 长短期记忆网络引入了记忆元（memory cell），或简称为单元（cell）。
### RNN的基本结构
普通RNN的结构非常简单：
- 一个单一的神经网络层
- 一个简单的tanh或sigmoid激活函数
- 一个状态传递机制
![[循环神经网络的结构.excalidraw|600]]
### LSTM的改进结构
LSTM在RNN的基础上增加了以下关键结构：
1. 记忆单元（Memory Cell）
- 这是LSTM最重要的创新，相当于一条"高速公路"
- 信息可以在其中长期保存
- 解决了RNN中的梯度消失问题
 ![lstm-0.svg](https://zh.d2l.ai/_images/lstm-1.svg)
1. 三个门控机制：
假设有$h$个隐藏单元，批量大小为$n$，输入数为$d$。 因此，输入为$\mathbf{X}_t\in\mathbb{R}^{n\times d}$， 前一时间步的隐状态为$\mathbf{H}_{t-1}\in\mathbb{R}^{n\times h}$。 相应地，时间步$t$的门被定义如下：
输入门是$\mathbf{I}_{t}\in\mathbb{R}^{n\times h}$， 
遗忘门是$\mathbf{F}_t\in\mathbb{R}^{n\times h}$， 
输出门是$\mathbf{O}_t\in\mathbb{R}^{n\times h}$,
候选记忆元是$\tilde{\mathbf{C}}_t\in\mathbb{R}^{n\times h}$，
记忆元是$\mathbf{C}_{t-1}\in\mathbb{R}^{n\times h}$
隐状态是$\mathbf{H}_t\in\mathbb{R}^{n\times h}$
它们的计算方法如下：
$$\begin{gathered}\mathbf{I}_{t}=\sigma(\mathbf{X}_t\mathbf{W}_{xi}+\mathbf{H}_{t-1}\mathbf{W}_{hi}+\mathbf{b}_i),\\\mathbf{F}_t=\sigma(\mathbf{X}_t\mathbf{W}_{xf}+\mathbf{H}_{t-1}\mathbf{W}_{hf}+\mathbf{b}_f),\\O_{t}=\sigma(\mathbf{X}_t\mathbf{W}_{xo}+\mathbf{H}_{t-1}\mathbf{W}_{ho}+\mathbf{b}_o),\\\tilde{\mathbf{C}}_t=\tanh(\mathbf{X}_t\mathbf{W}_{xc}+\mathbf{H}_{t-1}\mathbf{W}_{hc}+\mathbf{b}_c),\\\mathbf{C}_t=\mathbf{F}_t\odot\mathbf{C}_{t-1}+\mathbf{I}_t\odot\tilde{\mathbf{C}}_t,\\\mathbf{H}_t=\mathbf{O}_t\odot\tanh(\mathbf{C}_t)\end{gathered}$$
其中$\mathbf{W}_{xi},\mathbf{W}_{xf},\mathbf{W}_{xo},\mathbf{W}_{xc}\in\mathbb{R}^{d\times h}$是输入权重矩阵;$\mathbf{W}_{hi},\mathbf{W}_{hf},\mathbf{W}_{ho},\mathbf{W}_{hc}\in\mathbb{R}^{h\times h}$是隐层权重矩阵，$\mathbf{b}_i,\mathbf{b}_f,\mathbf{b}_o,\mathbf{b}_c\in\mathbb{R}^{1\times h}$是偏置参数。
- **输入门权重**：$\mathbf{W}_{xi}, \mathbf{W}_{hi}$
- **遗忘门权重**：$\mathbf{W}_{xf}, \mathbf{W}_{hf}$
- **输出门权重**：$\mathbf{W}_{xo}, \mathbf{W}_{ho}$
- **候选记忆元权重**：$\mathbf{W}_{xc}, \mathbf{W}_{hc}$
- **输出层权重**（若存在）：$\mathbf{W}_{hy}$​![lstm-1.svg](https://zh.d2l.ai/_images/lstm-3.svg)

### 特点
**遗忘门始终为1**且**输入门始终为0**， 则过去的记忆元$C_{t−1}$ 将随时间被保存并传递到当前时间步。
    引入这种设计是为了缓解梯度消失问题， 并更好地捕获序列中的长距离依赖关系。
    $C_t=f_t\odot C_{t-1}+i_t\odot\tilde{C_t}=1\cdot C_{t-1}+0\cdot\tilde{C_t}=C_{t-1}$
    这将使得记忆单元梯度可以无损地传递到任意远处的时间步，**彻底避免了梯度消失**。
    $\frac{\partial C_t}{\partial C_k}=\prod\limits_{j=k+1}^{t}\frac{\partial C_{j}}{\partial C_{j-1}}=1$
    遗忘门和输入门的设计使得模型可以在一定程度上缓解梯度消失。
**输出门接近1**，我们就能够有效地将所有记忆信息传递给预测部分；
**输出门接近0**，我们只保留记忆元内的所有信息，而不需要更新隐状态。
**输出门将隐藏状态的梯度解耦**：隐藏状态的梯度与细胞记忆状态的传播路径分离，提升稳定性。
### 注意
LSTM **不能完全解决梯度爆炸**，但相比传统 RNN，其爆炸风险更低。梯度爆炸的发生取决于权重矩阵的谱范数（最大奇异值）。


### **是否存在“输出权重矩阵”？**

- **严格来说**，**LSTM没有独立的输出权重矩阵**（例如，类似RNN中 $\mathbf{W}_{hh}$ 或 $\mathbf{W}_{xh}$​ 的显式输出权重）。
- **实际实现中**：
    - 若需将输出 $\mathbf{H}_t$ 映射到任务特定维度（如分类任务的类别数），通常会**添加额外的全连接层**（即输出权重矩阵）。例如：
        $\mathbf{Y}_t = \mathbf{H}_t \mathbf{W}_{hy} + \mathbf{b}_y$
        其中$\mathbf{W}_{hy} \in \mathbb{R}^{h \times m}$ 是任务相关的输出权重矩阵（$m$ 为输出维度）。

---
1. 解决长期依赖问题
- RNN：随着序列长度增加，早期信息会逐渐丢失
- LSTM：通过记忆单元可以长期保存重要信息

2. 梯度问题的改善
- RNN：存在严重的梯度消失/爆炸问题
- LSTM：记忆单元提供了梯度的"快速通道"，大大缓解了这个问题

3. 信息流控制
- RNN：信息强制性全部通过
- LSTM：可以选择性地保留或遗忘信息

4. 状态分离
- RNN：只有一个混合状态
- LSTM：分离了记忆单元（Ct）和隐藏状态（ht），使信息处理更有针对性