![[Pasted image 20250608104822.png]]
![[Pasted image 20250608104829.png]]

设 $\|\cdot\|$ 为范数，其共轭函数为： $\|\cdot\|^*(y) = \begin{cases} 0 & \|y\|_* \leq 1 \\ +\infty & \text{否则} \end{cases}$ 其中 $\|\cdot\|_*$ 是对偶范数（如 $\ell_p$ 与 $\ell_q$ 对偶，$1/p + 1/q = 1$）。


对于：$L = -\frac{1}{2}\|\lambda\|^2 + \mu\|x\|_1 - (A^T\lambda)^T x + b^T\lambda$
对 x 求 inf： 项 $-(A^T\lambda)^T x + \mu\|x\|_1$ 的下确界。设 $c = A^T\lambda$，利用 $\ell_1$ 范数的共轭性质（$\|x\|_1$ 的共轭为 $\|\cdot\|_\infty$ 指示函数），当且仅当 $\|c\|_\infty \leq \mu$ 时，inf 为 0（否则无界负）。代入得： $g(\lambda) = \begin{cases} b^T\lambda - \frac{1}{2}\|\lambda\|^2 & \|A^T\lambda\|_\infty \leq \mu \\ -\infty & \text{其他} \end{cases}$

对于形如 $\mu\|x\|_1+\beta x$ 样式的式子，求取 $inf$，直接将 $\mu\|x\|_1+\beta x$ 部分剔除，并设置范围为 $\|\beta\|_\infty \leq \mu$ 即可
注意这里的 $\|*\|_\infty$ 部分取值是$\ell_p$是1，所以为了满足$1/p + 1/q = 1$，所以 $\ell_q$ 是$\|*\|_\infty$
