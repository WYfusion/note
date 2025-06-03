### PReLU（参数化 ReLU，Parametric ReLU）

**提出时间**：2015 年由何凯明等人在论文《Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification》中提出。
和[[Leaky ReLU]]激活函数很像，但是这里将固定的$\alpha$变成了可学习的参数了$$f(x)=\begin{cases}x,&\mathrm{if~}x\geq0\\\alpha x,&\mathrm{if~}x<0&&\end{cases}$$
其中 **α 是一个可学习的参数**（通常初始化为 0，通过反向传播优化）。
- **特点**：
    - α 可以是**全局参数**（所有神经元共享一个 α），也可以是**按通道 / 按神经元的局部参数**（如 ResNet 中每个通道有独立的 α）。
    - 自适应学习负区间斜率，理论上能更好拟合数据分布，避免固定斜率的经验性偏差。
    - 增加了少量计算量（需优化 α 参数），但在深层网络中效果显著（如 ImageNet 分类任务中性能提升）。
- **变种**：
    - **CPReLU**：对称的 PReLU（α 在正负区间对称，较少使用）。
    - **APReLU**：自适应 PReLU，α 在训练过程中动态调整。

