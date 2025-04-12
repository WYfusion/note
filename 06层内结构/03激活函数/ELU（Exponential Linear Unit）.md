数：$ELU(x)=\{\begin{array}{cc}x&\mathrm{if}x\geq0\\\alpha(e^x-1)&\mathrm{if}x<0\end{array}$
**函数范围**：$(−α,+∞)$（当 $x→−∞$ 时输出趋近于$−α$）
![[ELU.png|600]]
导函数：$\frac{d}{dx}ELU(x)=\{\begin{array}{cc}1&\mathrm{if~}x\geq0\\ELU(x)+\alpha&\mathrm{if~}x<0\end{array}$
![[ELU_derivative.png|600]]

**导数范围**：$(α,1]$，当 $α$ 固定时（如 $α=1$）