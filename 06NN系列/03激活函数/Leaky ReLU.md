**产生背景**
Leaky ReLU 是为解决传统 **ReLU** 的“神经元死亡”问题而设计。ReLU 在输入为负时梯度为零，导致神经元无法更新参数。Leaky ReLU 通过引入负区间的微小斜率，允许梯度“泄露”，从而保持神经元活性。
**定义与数学表达式**
Leaky ReLU 的公式为：
$$
{\Large \displaystyle f(x)=
\begin{cases}
x, & x\geq0 \\
\alpha x, & x<0 & 
\end{cases}}
$$
Leaky ReLU的导函数公式：
$$\frac{d}{dx}LeakyReLU(x)=\{\begin{array}{cc}1&\mathrm{if~}x\geq0\\\alpha&\mathrm{if~}x<0\end{array}$$
其中，$α$是超参数（通常设为 $0.01$），控制负区间的斜率。![[LeakyReLU.png|600]]![[LeakyReLU_derivative.png|600]]
下图是$α=0.1$时的图像
![[LeakyReLU_0.1.png|600]]
![[LeakyReLU_0.1_derivative.png|600]]
**核心优势**
1. **缓解神经元死亡**：负输入时仍有梯度更新，避免神经元永久失活。
2. **训练稳定性**：在深层网络中保持梯度流动，加速收敛。
3. **计算高效**：与 ReLU 类似，仅需简单阈值操作，适合大规模数据。