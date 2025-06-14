残差结构在**ResNet**提出

当模型深度增加时，模型的效果并不会变好（网络退化）

由于非线性激活函数ReLU的存在，导致输入输出不可逆，造成了模型的信息损失，更深层次的网络使用了更多的ReLU函数，导致了更多的信息损失，这使得浅层特征随着前项传播难以得到保存。

使用两个分支，一个捷径分支保留原输入的特征，一个主分支进行**降维、卷积、升维**

对于主分支：

- 通过1×1卷积核进行通道压缩减少通道数，降维操作
- 通过卷积核进行卷积处理
- 再次通过1×1卷积核进行通道扩展恢复原始通道数，升维操作

对于捷径分支：

- 一般不做处理，直接与主分支的输出进行相加操作，需要保障通道数、特征图尺寸大小均一致。
- 但是面对需要缩小特征图尺寸、加倍通道数时，也需要在此分支中添加合适卷积核

也就是说形成一个两头小中间大的瓶颈结构


## 注意1×1 卷积可以实现用卷积代替线性的效果
1. **线性单元的本质** 线性单元（Linear Unit）本质是对输入进行线性变换，即 $y = Wx + b$，其中 $W$ 是权重矩阵，$b$ 是偏置。在序列处理中，传统线性单元（如全连接层）会对每个时间步的特征独立进行线性变换，权重矩阵的形状为 **(输入维度，输出维度)**。
    
2. **1×1 卷积的等价性** ^37c3dd
    - **输入信号**为$X_{in}\in\mathbb{R}^{B \times N \times S}$，这里的 $B$ 是独立的音频序列数， $N$ 是每个时间步的特征向量长度（原嵌入维度），$S$ 是每个序列包含 $S$ 个时间步（如音频的采样点数）。
    - 在**一维序列处理**中（输出序列 $X \in \mathbb{R}^{B \times N' \times S}$，$B$ 是批大小，$N'$ 为嵌入维度，$S$ 为序列长度），1×1 卷积的核大小为 1，相当于对每个时间步的 N' 维特征进行线性变换。
    - 一维 1×1 卷积的权重矩阵形状为 **(输出维度(卷积核个数)，输入维度，1)**，卷积层尺寸 $N' \times N \times 1$（输出通道数 × 输入通道数 × 核大小）。但由于核大小为 1，其计算等价于线性变换 $y = Wx + b$，即**每个时间步的特征通过相同的权重矩阵进行线性变换**（权重共享于所有时间步）。
    - 因此，1×1 卷积在功能上完全等价于线性单元，但属于卷积操作的范畴，可无缝融入卷积神经网络（CNN）架构。
### 线性单元与全连接层的区别
| **特性**                | **线性单元（Linear Unit）**          | **全连接层（Fully Connected Layer）**  |
| --------------------- | ------------------------------ | -------------------------------- |
| **定义**                | 单个线性变换单元，实现 $y = Wx + b$。      | 由多个线性单元组成，每个输出维度对应一个线性单元。        |
| **输入 / 输出**           | 输入为向量，输出为标量（单个维度的线性变换）。        | 输入为向量，输出为向量（多个维度的线性变换）。          |
| **参数规模**              | 权重矩阵大小为 **(1, 输入维度)**（单个输出维度）。 | 权重矩阵大小为 **(输出维度，输入维度)**（多个输出维度）。 |
| **应用场景**              | 通常作为全连接层的组成部分，或用于逐元素线性变换。      | 用于将高维特征映射到目标维度（如分类任务的类别数）。       |

