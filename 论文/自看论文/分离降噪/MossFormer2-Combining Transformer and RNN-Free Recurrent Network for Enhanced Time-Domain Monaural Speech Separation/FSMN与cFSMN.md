FSMN（Feedforward Sequential Memory Networks）是一种用于处理序列数据的**前馈神经网络结构**，它通过引入记忆块来有效建模序列中的长期依赖关系，原理基于数字信号处理中滤波器的近似思想，公式用于描述**记忆块**的计算以及**隐藏层**的激活。[[1803.05030] Deep-FSMN 用于大词汇量连续语音识别 --- [1803.05030] Deep-FSMN for Large Vocabulary Continuous Speech Recognition](https://ar5iv.labs.arxiv.org/html/1803.05030?_immersive_translate_auto_translate=1)


# FSMN
1. **FSMN 的原理**：FSMN 的设计灵感来源于数字信号处理知识，即任何无限脉冲响应（IIR）滤波器可以用高阶有限脉冲响应（FIR）滤波器很好地近似。循环神经网络（RNN）中的循环层在概念上可被视为一阶 IIR 滤波器，因此可以用高阶 FIR 滤波器精确近似。FSMN 通过在标准前馈全连接神经网络的隐藏层中添加记忆块来扩展网络结构，这些记忆块采用类似于 FIR 滤波器的抽头延迟线结构，能够编码长上下文信息，帮助模型捕捉长期依赖关系。在语音识别任务中，语音信号具有时间序列特性，前后帧之间存在依赖关系，FSMN 的记忆块可以整合这些信息，提升对语音内容的理解和识别能力。
2. **FSMN 的公式**
    - **记忆块输出公式（以 vFSMN 为例）**：$$\mathbf{\tilde{h}}_t^\ell=\sum_{i=0}^{N_1}\mathbf{a}_i^\ell\odot\mathbf{h}_{t-i}^\ell+\sum_{j=1}^{N_2}\mathbf{c}_j^\ell\odot\mathbf{h}_{t+j}^\ell$$这个公式用于计算记忆块的输出。其中，$\odot$表示两个等大小向量的逐元素乘法；$N_{1}$是回溯阶数，代表向过去回溯的历史项数量，$N_{2}$是前瞻阶数，表示向未来前瞻的窗口大小；$h_{t}^{\ell}$是$\ell$层在时刻$t$的隐藏层输出，$a_{i}^{\ell}$和$c_{j}^{\ell}$是可学习的参数；$\tilde{h}_{t}^{\ell}$是记忆块在时刻$t$的输出，可看作是时刻$t$周围长上下文的固定大小表示。
    - **下一层隐藏层激活公式**：$$\mathbf{h}_t^{\ell+1}=f(\mathbf{W}^\ell\mathbf{h}_t^\ell+\mathbf{\tilde{W}}^\ell\mathbf{\tilde{h}}_t^\ell+\mathbf{b}^\ell)$$此公式用于计算下一层隐藏层的激活值。$W^{\ell}$和$\tilde{W}^{\ell}$是权重矩阵，$b^{\ell}$是偏置项，$f(\cdot)$是激活函数，如 ReLU 等。该公式表明下一层隐藏层的激活值是由当前层的隐藏层输出$h_{t}^{\ell}$ 、记忆块输出$\tilde{h}_{t}^{\ell}$经过线性变换和激活函数处理后得到的。


# cFSMN（FSMN 的变体）
1. **cFSMN（Compact FSMN）的优势**：
    - **参数效率提升**：cFSMN 通过引入瓶颈层（Bottleneck Layer）减少了记忆块处理的特征维度。在传统 FSMN 中，记忆块需要处理完整维度的隐藏层输出，而 cFSMN 先通过线性投影层$p_{t}^{\ell}=V^{\ell}h_{t}^{\ell}$将高维特征映射到低维空间，再对低维特征应用记忆块，显著降低了参数量。
    - **计算复杂度降低**：由于记忆块处理的是低维特征，计算量大幅减少。例如，在语音识别任务中，cFSMN 的计算效率比传统 FSMN 提高约 30%，同时保持相近的模型性能。
    - **上下文建模优化**：cFSMN 通过门控机制（如 GCU 层）增强了对关键上下文信息的选择性关注，避免了传统 FSMN 中可能存在的信息冗余问题。这种结构在长序列任务中表现更优，能够更精确地捕捉时序依赖关系。
2. **cFSMN的公式**
    - **cFSMN 记忆块编码公式**：$$\mathbf{\tilde{p}}_t^\ell=\mathbf{p}_t^\ell+\sum_{i=0}^{N_1}\mathbf{a}_i^\ell\odot\mathbf{p}_{t-i}^\ell+\sum_{j=1}^{N_2}\mathbf{c}_j^\ell\odot\mathbf{p}_{t+j}^\ell$$cFSMN 将记忆块添加到线性投影层，此公式描述了 cFSMN 中记忆块的编码方式。其中$p_{t}^{\ell}=V^{\ell}h_{t}^{\ell}+b^{\ell}$是$\ell$层线性投影层的输出 ，与 FSMN 的记忆块公式相比，形式上有一定变化，但同样用于捕捉上下文信息。
    - **cFSMN 下一层隐藏层激活公式**：$$\mathbf{h}_t^{\ell+1}=f(\mathbf{U}^\ell\mathbf{\tilde{p}}_t^\ell+\mathbf{b}^{\ell+1})$$该公式用于计算 cFSMN 中下一层隐藏层的激活值，$U^{\ell}$是权重矩阵，$b^{\ell+1}$是偏置项，$f(\cdot)$是激活函数。它表明下一层隐藏层的激活基于记忆块的输出$\tilde{p}_{t}^{\ell}$经过线性变换和激活函数处理得到。
