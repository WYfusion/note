函数：$SELU(x)=\lambda\cdot\{\begin{array}{cc}x&\mathrm{if}x\geq0\\\alpha(e^x-1)&\mathrm{if}x<0\end{array}$
通常 $λ=1.0507$, $α=1.67326$

- **函数范围**：$(−λα,+∞)$，通过自归一化机制稳定训练。
![[SELU.png|600]]
导函数：$\frac{d}{dx}SELU(x)=\{\begin{array}{cc}\lambda &\mathrm{if}x\geq0\\\lambda\alpha e^x&\mathrm{if}x<0\end{array}$
![[SELU_derivative.png|600]]
### 特点
SELU在负区间引入非零梯度，避免神经元“死亡”，且通过缩放因子提升稳定性
SELU是ELU的缩放版本，通过λ调节输出分布，实现自归一化，而ELU需要手动归一化