$BN(\text{Batch Normalization})$原理，**纵向规范化**

主要用于**加速训练**和**提高模型**的稳定性。

减小当前卷积层**输出特征的波动**进而减少后面卷积层的输入波动，使得模型在训练过程中更加稳定。

由于输入的稳定性提高，后续层的参数更新更加一致，**减少了梯度更新过程中的不确定性**，从而加速了收敛过程。

#### 基本知识

- **作用范围**：一个batch批次

- **作用位置**：卷积层后、激活函数之前

**$BN$作用于**：每批次每个通道的数据

**长度**：对**每个通道**（channel）分别在同一个batch内**跨所有样本**（examples）和**空间位置**（features，如高度H和宽度W）计算均值和方差。

对于每一个$\text{batch}$批次的所有输入$\text{feature}\ \text{map}$的每一个维度通道分别进行标准化处理。
$$
\begin{aligned}
&\mu_{\mathcal B}\leftarrow\frac{1}{m}\sum_{i=1}^{m}x_{i}  & \text{// mini-batch mean} \\
&\sigma_{\mathcal B}^{2}\leftarrow\frac{1}{m}\sum_{i=1}^{m}(x_{i}-\mu_{\mathcal{B}})^{2} &\quad\text{// mini-batch variance} \\
&\widehat{x}_{i}\leftarrow\frac{x_{i}-\mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^{2}+\epsilon}} & \text{// normalize} \\
&y_{i}\leftarrow\gamma\widehat{x}_{i}+\beta\equiv\mathrm{BN}_{\gamma,\beta}(x_{i})& //scale\ and\ shift \\
\end{aligned}
$$
其中的 $\epsilon$是一个极小量，避免分母为$0$。

1.$x_i$是同一维度下所有$\text{feature}\ \text{map}$的元素值
2.$m$是同一维度下所有元素的个数
3.$\mu_{\mathcal B}$和$\sigma_{\mathcal B}^{2}$是在正向传播中统计得到的
4.$\gamma$和$\beta$是在反向传播中训练得到的。$\gamma$和$\beta$是$BN$层的精髓

$\widehat{x}_{i}$的获得目的是：
- 数据就被移到中心区域，对于大多数**激活函数**而言，这个区域的**梯度都是最大**的或者是**有梯度**的这可以看做是一种对抗梯度消失的有效手段。 
- ​对于每一层数据都那么做的话，数据的分布总是在随着变化敏感的区域(梯度大的地方)，相当于没考虑数据分布变化，这样训练起来更有效率。 

$\widehat{x}_{i}$的引入缺陷是：

- 在中心区域内，在激活函数中数据的范围被限制住了，将导致不同特征下的参数难以根据梯度进行有效的更新，泛化能力可能变弱，减弱网络性能。

$y_{i}$的获得目的是：

- BN的本质就是利用优化变一下$\mu_{\mathcal B}$和$\sigma_{\mathcal B}^{2}$方差大小和均值位置，数据的范围可以根据损失值更低的所需特征数据的分布进行动态自适应的调整(也即保障还能有机会再次降低损失值)。同时此步骤为线性变化，保证了之前模型的非线性表达能力。 

由于归一化后的$\widehat{x}_{i}$基本会被限制在正态分布下，使得网络的表达能力可能会下降。所以引入$\gamma$和$\beta$两个参数进行平移和放缩，而不被限制于均值为0和方差为1的分布。在训练开始时，$\gamma$和$\beta$会被随机初始化（通常为$\gamma$初始化为1，$\beta$初始化为0），$\gamma$和$\beta$是可学习的参数，和权重一样在反向传播中不断更新。

##### 使用BN时需要注意的问题

(1)训练时要将$\text{training}$参数设置为$\text{True}$，在验证时将$\text{trainning}$参数设置为$\text{False}$。在$\text{pytorch}$中可通过创建模型的$\text{model}.\text{train}()$和$\text{model}.\text{eval}()$方法控制。
(2)$\text{batch}\ \text{size}$尽可能设置大点，设置小后表现总体数据的能力可能很糟糕。设置的越大，求得的均值和方差越接近整个训练集的均值和方差。
(3)建议将bn层放在卷积层$(\text{Conv})$和激活层(例如$\text{Relu}$)之间，且卷积层不要使用偏置$\text{bias}$，因为没有作用，即使使用了偏置$\text{bias}$求出的结果也是一样的$y_i$，还徒增加参数量。