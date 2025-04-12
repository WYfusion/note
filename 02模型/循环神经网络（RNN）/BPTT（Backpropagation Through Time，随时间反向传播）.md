BPTT 是专门用于训练**循环神经网络**（RNN）的梯度计算算法，核心思想是将RNN在时间轴上展开为前馈网络，通过反向传播计算梯度。其本质是传统反向传播（BP）在时序数据上的扩展，但需处理时间依赖性和序列长度带来的计算挑战。

---
[BPTT学习视频](https://www.bilibili.com/video/BV1fF411P72y/?spm_id_from=333.337.search-card.all.click&vd_source=1574bd9421ca96a2f458f57838315ec6)
所有单独的时刻下，各个节点共享权重矩阵（U、W、V）
- 误差项沿时间步反向传播，梯度计算需累加所有时间步的影响
#### **1. 算法原理**
**a. 前向传播阶段**
- RNN在每个时间步 $t$ 的隐藏状态 $h_t$​ 和输出 $y_t$​ 由以下公式决定：![[RNN#^aadbf5]]![[RNN#^4210c7]]
    其中$W_{hh}, W_{xh},W = [W_{hh}, W_{xh}]$为权重矩阵，$b_h$ 为偏置项。
- **时间展开**：将RNN视为一个展开的前馈网络，每个时间步对应一个“虚拟层”

**b. 损失计算**
- 总损失 $L$ 为各时间步损失 $L_t$ 的累加（如交叉熵、均方误差）：

$$L=\sum_{t=1}^TL_t(y_t,\hat{y}_t)$$
其中的$L_t$是损失函数，$\hat{y}_t$是期望输出。 ^ad153a
- **c. 反向传播阶段**
    - 从时间步 $T$ 到 $1$ **逆序**计算梯度，传播路径包括：
    1. **输出层梯度**：$\frac{\partial L}{\partial y_t}$
    2. **隐藏层梯度**：
        - 当前时间步的梯度来自两部分：$$\frac{\partial L}{\partial h_t}=\frac{\partial L_t}{\partial h_t}+\frac{\partial L}{\partial h_{t+1}}\frac{\partial h_{t+1}}{\partial h_t}$$
            - 第一项：当前步输出损失对 $h_t$​ 的直接影响（通过 $y_t​$）。
            - 第二项：未来时间步的梯度通过 $h_{t+1}$​ 反向传播到 $h_t$，体现时间依赖性。
        
![[循环神经网络的结构.excalidraw|80%]]
$$\begin{cases}h_{1}=g(U{x_1}+W{h_0})\\\hat{y}_{1}=g(V{h_1})&\end{cases}\begin{cases}h_{2}=g(U{x_2}+W{h_1})\\\hat{y}_{2}=g(V{h_2})&\end{cases}\begin{cases}h_{3}=g(U{x_3}+W{h_2})\\\hat{y}_{3}=g(V{h_3})&\end{cases}$$
注意,$h$隐层的输入含有之前时刻所有隐层，这将直接影响后面计算权重梯度时的链式求导过程。
对于$t=3$时刻，损失函数$L$对三个权重矩阵的偏导：
$$①\frac{\partial L_{3}}{\partial V}=\frac{\partial L_{3}}{\partial\widehat{y}_{3}}\times\frac{\partial\widehat{y}_{3}}{\partial V}$$
$$②\frac{\partial L_{3}}{\partial W}=\frac{\partial L_{3}}{\partial\widehat{y}_{3}}\times\frac{\partial\widehat{y}_{3}}{\partial h_{3}}\times\frac{\partial h_{3}}{\partial W}+\frac{\partial L_{3}}{\partial\widehat{y}_{3}}\times\frac{\partial\widehat{y}_{3}}{\partial h_{3}}\times\frac{\partial h_{3}}{\partial h_{2}}\times\frac{\partial h_{2}}{\partial W}+\frac{\partial L_{3}}{\partial\widehat{y}_{3}}\times\frac{\partial \widehat{y}_3}{\partial h_{3}}\times\frac{\partial h_{3}}{\partial h_{2}}\times\frac{\partial h_{2}}{\partial h_{1}}\times\frac{\partial h_{1}}{\partial W}$$
$$③\frac{\partial L_3}{\partial U}=\frac{\partial L_3}{\partial\hat{y}_{3}}\times\frac{\partial \hat{y}_{3}}{\partial h_{3}}\times\frac{\partial h_{3}}{\partial U}+\frac{\partial L_{3}}{\partial\hat{y}_{3}}\times\frac{\partial\hat{y}_{3}}{\partial h_{3}}\times\frac{\partial h_{3}}{\partial h_{2}}\times\frac{\partial h_{2}}{\partial U}+\frac{\partial L_{3}}{\partial\hat{y}_{3}}\times\frac{\partial\hat{y}_{3}}{\partial h_{3}}\times\frac{\partial h_{3}}{\partial h_{2}}\times\frac{\partial h_{2}}{\partial h_{1}}\times\frac{\partial h_{1}}{\partial u}$$
综上有任意t时刻，损失函数$L$对三个权重矩阵的偏导：
$$①\frac{\partial L_t}{\partial V}=\frac{\partial L_t}{\partial\hat{y}_t}\times\frac{\partial\hat{y}_t}{\partial V}$$
$$②\frac{\partial L_t}{\partial W}=\sum_{k=1}^{t}\frac{\partial L_t}{\partial\hat{y}_t}\times\frac{\partial\hat{y}_t}{\partial h_t}\times\frac{\partial h_t}{\partial h_k}\times\frac{\partial h_{k}}{\partial W}$$
这里的$\frac{\partial h_t}{\partial h_k}=\frac{\partial h_t}{\partial h_{t-1}}\times\frac{\partial h_{t-1}}{\partial h_{t-2}}\times\cdots\frac{\partial h_{k+1}}{\partial h_{k}}=\prod\limits_{j=k+1}^{t}\frac{\partial h_{j}}{\partial h_{j-1}}$
故②式可写为：$$②\frac{\partial L_t}{\partial W}=\sum_{k=1}^{t}\frac{\partial L_t}{\partial\hat{y}_t}\times\frac{\partial\hat{y}_t}{\partial h_t}(\prod\limits_{j=k+1}^{t}\frac{\partial h_{j}}{\partial h_{j-1}})\times\frac{\partial h_{k}}{\partial w}$$
同理有
$$③\frac{\partial L_t}{\partial U}=\sum_{k=1}^{t}\frac{\partial L_t}{\partial y_t}\times\frac{\partial\widehat{y}_t}{\partial h_t}(\prod\limits_{j=k+1}^{t}\frac{\partial h_{j}}
{\partial h_{j-1}})\times\frac{\partial h_{k}}{\partial U}$$
计算总损失函数![[BPTT（Backpropagation Through Time，随时间反向传播）#^ad153a]]
并结合①②③可得总的损失函数$L$对于三个权重矩阵的偏导是：

$$\begin{aligned}&\frac{\partial L}{\partial V}=\sum_{i=1}^{T}\frac{\partial L_t}{\partial V}=\sum_{i=1}^{T}\frac{\partial L_t}{\partial \hat{y}_t}\times\frac{\partial\hat{y}_t}{\partial V}\\&\frac{\partial L}{\partial W}=\sum_{i=1}^{T}\frac{\partial L_t}{\partial W}=\sum_{i=1}^{T}\frac{\partial L_t}{\partial W}=\sum_{k=1}^{t}\frac{\partial L_t}{\partial\hat{y}_t}\times\frac{\partial\hat{y}_t}{\partial h_t}(\prod\limits_{j=k+1}^{t}\frac{\partial h_{j}}{\partial h_{j-1}})\times\frac{\partial h_{k}}{\partial w}\\&\frac{\partial L}{\partial U}=\sum_{i=1}^{T}\frac{\partial L_t}{\partial W}=\sum_{i=1}^{T}\sum_{k=1}^{t}\frac{\partial L_t}{\partial y_t}\times\frac{\partial\widehat{y}_t}{\partial h_t}(\prod\limits_{j=k+1}^{t}\frac{\partial h_{j}}
{\partial h_{j-1}})\times\frac{\partial h_{k}}{\partial U}\end{aligned}$$
##### 累乘部分暴露的**关键问题**
由于隐层$h_t$是由![[RNN#^aadbf5]]
可知$\frac{\partial h_{j}}{\partial h_{j-1}}$进行求导运算时，先对非线性激活函数$g$求导，再对内部$h_{t-1}$求导后将会暴露出权重W或U的累乘，若$g^` \times W$或$g^` \times U$这一指数级运算的底数**大于1或小于1**(也即，$\frac{\partial h_{j}}{\partial h_{j-1}}>1或<1$)，则会分别随着时间序列长度T的变长而导致梯度爆炸和梯度消失。

---
#### **3. 算法变体与优化**
**a. 截断BPTT（Truncated BPTT）**
- **动机**：长序列训练时内存和计算成本过高。
- **方法**：将长序列分割为多个子序列（如长度 $k$），在子序列内部执行BPTT，忽略跨子序列的梯度传播。  
    ![截断BPTT](https://pfst.cf2.poecdn.net/base/image/beaf31990ae0e4e4543ebd74a0c17f878f43600e1d210b5258d2088fb5d9e2c4?pmaid=312498326)
**影响**：牺牲部分长期依赖建模能力，但显著降低计算复杂度。

**b. 梯度裁剪（Gradient Clipping）**
- 对梯度进行逐元素或全局范数限制，防止梯度爆炸。

**c. 结构改进**
- **LSTM/GRU**：通过门控机制（遗忘门、输入门）选择性传递梯度，缓解梯度消失。

---
BPTT 是RNN训练的基石算法，通过时间展开和反向传播实现时序依赖建模，但其计算复杂度和梯度问题促使了LSTM、Transformer等结构的演进。理解BPTT的机制有助于设计更高效的序列模型和优化策略。