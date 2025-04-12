层归一化
$$y=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}\cdot\gamma+\beta$$
其中：
- $x$ 是输入数据
- $\mu$ 是在特征维度上计算的均值
- $\sigma^2$ 是在特征维度上计算的方差
- $\epsilon$ 是为了数值稳定性添加的小常数（通常取 1e-5）
- $\gamma$ 和 $\beta$ 是可学习的缩放和偏移参数

**作用位置**：卷积层后、激活函数之前



**作用范围**：对每个样本（example），跨所有通道（channels）和空间位置（features）计算均值和方差。每一个样本整个特征（所有通道）共享一个均值和方差。消除样本内所有特征的分布差异，包括通道间的差异。
- 对于图像数据 (N,C,H,W)，归一化维度是 (C,H,W)
- 均值和方差对每个样本计算一次。
### 不同维度输入的计算公式

1. 二维输入 (N, D)（全连接层）：
$\mu_n = \frac{1}{D}\sum_\limits{i=1}^{D}x_{n,i}$
$\sigma_n^2 = \frac{1}{D}\sum_\limits{i=1}^{D}(x_{n,i} - \mu_n)^2$
2. 三维输入 (N, L, D)（序列数据）：
$\mu_n = \frac{1}{L \cdot D}\sum_\limits{l=1}^{L}\sum_\limits{d=1}^{D}x_{n,l,d}$
$\sigma_n^2 = \frac{1}{L \cdot D}\sum_\limits{l=1}^{L}\sum_\limits{d=1}^{D}(x_{n,l,d} - \mu_n)^2$

3. 四维输入 (N, C, H, W)（图像数据）：
$\mu_n = \frac{1}{C \cdot H \cdot W}\sum_\limits{c=1}^{C}\sum_\limits{h=1}^{H}\sum_\limits{w=1}^{W}x_{n,c,h,w}$
$\sigma_n^2 = \frac{1}{C \cdot H \cdot W}\sum_\limits{c=1}^{C}\sum_\limits{h=1}^{H}\sum_\limits{w=1}^{W}(x_{n,c,h,w} - \mu_n)^2$



## 层归一化特别适用于：

- Transformer 模型中的归一化层
- RNN 和 LSTM 网络
- 需要批次独立性的任务
- 可变长度序列的处理（如NLP中的文本序列），不受批次大小影响，适合变长序列