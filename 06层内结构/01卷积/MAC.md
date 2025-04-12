内容访问时间成本(memory access cost(MAC))

相同的FLOPs条件下，当卷积层的输入特征矩阵与输出特征矩阵的Channel数目相等时，MAC最小

以下为1×1卷积的条件下MAC的情况
$$
MAC=hw(c_{1}+c_{2})+c_{1}c_{2}
$$

$$
\begin{aligned}&\text{算数平均数}\quad\text{几何平均数}\\&\frac{c_1+c_2}2\geq\sqrt{c_1c_2}\quad\text{均值不等式}\end{aligned}
$$

由以上两式得到下面：
$$
\begin{aligned}MAC&\geq2hw\sqrt{c_{1}c_{2}}+c_{1}c_{2}\\&\geq2\sqrt{hwB}+\frac{B}{hw}\quad B=hwc_{1}c_{2}\end{aligned}
$$

^6b8f85

$c_1$是输入特征矩阵通道、$c_2$是输出特征矩阵的通道数、$h$和$w$是输入特征矩阵的形状

对于$MA$C右式第一项的处理，$c_1=c_2$取最小值

此外，当GConv的groups增大时(保持FLOPs不变时)， MAC也会增大，所以分组数$g$应该尽可能的小
$$
\begin{aligned}MAC&=hw(c_{1}+c_{2})+\frac{c_{1}c_{2}}{g}\\&=hwc_{1}+\frac{Bg}{c_{1}}+\frac{B}{hw}\quad B=hwc_{1}c_{2}/g\quad\text{(FLOPs)}\end{aligned}
$$