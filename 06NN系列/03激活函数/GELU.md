 GELU（ (Gaussian Error Linear Units) 高斯误差激活函数）

相较于 ReLU 等激活函数，GELU 更加平滑，有助于提高训练过程的收敛速度和性能。

```python
# GELU激活函数的定义
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
```


$$
{\Large \displaystyle \operatorname{GELU}(x)=x\Phi(x)\approx0.5 x\left(1+\tanh \left(\sqrt{\frac{2}{\pi}}\left(x+0.044715 x^3\right)\right)\right)}
$$
$\sqrt{\frac{2}{\pi}}$和$0.044715$是 GELU 函数的两个调整系数。

![[Figure_1.png|500]]
其导数是：$\frac{d}{dx}GELU(x)=\Phi(x)+x\cdot\phi(x)$
![[GELU_derivative.png|500]]

### GELU优点

-  GELU引入了**非线性变换**，使得神经网络能够学习更复杂的映射，有助于提高模型的表达能力。

- 具有连续的导数，无梯度截断，并且GELU 在 0 附近比 ReLU 更加平滑，因此在训练过程中**更容易收敛**。

- 不同于ReLU在负数范围内完全置零，GELU在负数范围内引入了一个**平滑的非线性**，有助于防止神经元“死亡”问题。
- 输出在输入接近于 0 时接近于高斯分布，有助于**提高神经网络的泛化能力**，使得模型更容易适应不同的数据分布。

### GELU缺点

- 涉及到指数、平方根和双曲正切等运算，因此在计算资源有限的情况下可能会带来较大的计算开销。
- 非线性范围较小，对于较大的输入值，GELU函数的输出**趋向于线性**，可能会导致一些非线性特征的丢失。
