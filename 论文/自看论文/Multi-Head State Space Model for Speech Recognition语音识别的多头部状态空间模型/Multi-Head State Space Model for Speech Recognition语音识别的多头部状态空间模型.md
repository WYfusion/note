---
类型: 语音增强
---
2023年发表于INTERSPEECH会议上

---
[[状态空间模型（State Space Models, SSMs）|State space models (SSMs)]]近年来在在小规模序列和语言建模任务上，可以与许多基于注意力的方法相媲美并表现出色。本文作者实现了配置了特殊门控机制的**多头状态空间**(MH-RSM)架构。其中***并行头***作用是动态地学习关于序列数据的**局部**和**全局**时间信息。
- 可作为Transformer编码器中[[Multi-Head Self-Attention#^61f90e|多头注意力]]的替代品。
- 用MH-SSMs层（称为Stateformer）来增强Transformer块
- 可以使用MH-SSMs取代模型的前置卷积

### Gate & Cat & Project结构
![[Stacked & Multi-Head Extension.png|300]]

下式**定义了双向 S4 的**外部框架（正向 + 反向处理 + 拼接）：
$$\begin{aligned}&\boldsymbol{y}\leftarrow\mathsf{Cat}([\mathsf{S}4(\boldsymbol{u}),\mathsf{Rev}(\mathsf{S}4(\mathsf{Rev}(\boldsymbol{u})))])\\&\boldsymbol{y}\leftarrow\mathsf{Linear}(\text{Activation}(\boldsymbol{y}))\end{aligned}$$






