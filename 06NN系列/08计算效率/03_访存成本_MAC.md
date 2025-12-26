# 内容访问成本 (Memory Access Cost, MAC)

在评估模型效率时，仅看 FLOPs 是不够的。**MAC (Memory Access Cost)** 衡量了模型在推理过程中需要进行多少次内存读写操作。在带宽受限的设备（如移动端）上，MAC 往往比 FLOPs 更能决定实际推理速度。

## 1. MAC 的定义与计算

对于一个标准的 $1 \times 1$ 卷积层（常用于 ShuffleNet, MobileNet 等轻量级网络），其 MAC 主要由三部分组成：
1.  **读取输入特征图**: $H \times W \times C_{in}$
2.  **读取权重参数**: $C_{in} \times C_{out}$ (忽略 $1 \times 1$ 的尺寸系数)
3.  **写入输出特征图**: $H \times W \times C_{out}$

### 总公式
$$ 
\text{MAC} = H W (C_{in} + C_{out}) + C_{in} C_{out} 
$$

---

## 2. 最小化 MAC 的条件 (ShuffleNet G1 准则)

我们希望在保持计算量 (FLOPs) 固定的情况下，最小化 MAC。

对于 $1 \times 1$ 卷积，其 FLOPs (记为 $B$) 为：
$$ B = H W C_{in} C_{out} 
$$
由此可得 $H W = \frac{B}{C_{in} C_{out}}$。

将 $MAC$ 公式改写并利用**均值不等式** ($a+b \ge 2\sqrt{ab}$)：

$$ 
\begin{aligned}
\text{MAC} &= H W (C_{in} + C_{out}) + C_{in} C_{out} \\
&\ge 2 H W \sqrt{C_{in} C_{out}} + C_{in} C_{out} \\
\end{aligned}
$$ 

或者更直观地，利用 $C_{in} C_{out} = \frac{B}{H W}$ 代入：
$$ 
\begin{aligned}
\text{MAC} &\ge 2 \sqrt{H W B} + \frac{B}{H W}
\end{aligned}
$$ 

### 结论
**当且仅当 $C_{in} = C_{out}$ 时，MAC 取最小值。**

这便是 **ShuffleNet V2** 提出的第一条设计准则：
> **G1: 当卷积层的输入特征矩阵与输出特征矩阵的 Channel 数目相等时，MAC 最小。**

这意味着在设计轻量级网络时，应尽量避免通道数剧烈变化的卷积层（如 bottleneck 结构中通道数先降后升），而倾向于等宽结构。

---

## 3. 分组卷积对 MAC 的影响 (ShuffleNet G2 准则)

虽然分组卷积能降低 FLOPs，但它会增加 MAC。
带分组数 $g$ 的 $1 \times 1$ 卷积，其 FLOPs 为 $B = \frac{h w c_1 c_2}{g}$。
此时 MAC 为：
$$ 
\begin{aligned}
\text{MAC} &= h w (c_1 + c_2) + \frac{c_1 c_2}{g} \\
&= h w c_1 + \frac{B g}{c_1} + \frac{B}{h w}
\end{aligned}
$$ 

可以看到，MAC 与分组数 $g$ 成正比（在 $c_1$ 固定时）。

### 结论
> **G2: 当分组数 $g$ 增大时（保持 FLOPs 不变），MAC 也会增大。**

因此，分组数 $g$ 不宜过大。过大的分组数虽然降低了理论计算量，但增加了内存访问开销，可能导致实测速度变慢。
