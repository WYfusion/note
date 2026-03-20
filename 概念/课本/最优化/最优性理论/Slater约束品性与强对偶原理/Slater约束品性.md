### Slater条件
![[Pasted image 20250608204826.png]]
注意Slater约束品性只需要**存在一个实例**符合无等式约束即可！！
[[相对内点]]，[[非负加权和与仿射函数的复合]]
![[Pasted image 20250608204906.png]]
![[Pasted image 20250608204924.png]]
![[Pasted image 20250608212442.png]]
![[Pasted image 20250608212737.png]]

1. **原始与对偶问题** 原始凸优化问题：
    $\begin{cases} \min\ f(x) \\ \text{s.t.}\ c_i(x) \leq 0\ (i=1,\dots,m),\ Ax = b,\ x \in D, \end{cases}$
    其中 $f, c_i$ 凸，D 凸集，最优值 $p^* = \min f(x)$（有限）。对偶问题通过拉格朗日函数 $L(x,\lambda,\nu) = f(x) + \lambda^T c(x) + \nu^T(Ax - b)$（$\lambda \geq 0$），对偶函数 $g(\lambda,\nu) = \inf_{x \in D} L(x,\lambda,\nu)$，最优值 $d^* = \max g(\lambda,\nu)$。
2. **集合定义与分离性**
    - $A = \{(u,v,t) \mid \exists x \in D,\ c_i(x) \leq u_i,\ Ax - b = v,\ f(x) \leq t\}$（原始可行解的 “松弛” 表示，凸集，因凸函数与线性约束保凸）。
    - $B = \{(0,0,s) \mid s < p^*\}$（射线，凸集）。
    - $A \cap B = \emptyset$（否则 $f(x) \leq t < p^*$ 与 $p^*$ 最优性矛盾）。
3. **超平面分离定理** 存在 $(\lambda,\nu,\mu) \neq 0$，$\alpha$，使：
    $\lambda^T u + \nu^T v + \mu t \geq \alpha\ (\forall (u,v,t) \in A),\quad \lambda^T u + \nu^T v + \mu t \leq \alpha\ (\forall (u,v,t) \in B).$
    - **符号分析**：$\lambda \geq 0$（否则 $u_i \to +\infty$ 时左边无下界），$\mu \geq 0$（同理 $t \to +\infty$ 时无下界）。
    - 对 B，令 $u=0,v=0,t \to p^-$，得 $\mu p^* \leq \alpha$。
    - 对 A 中任意 x，代入 $(u,v,t)=(c(x),Ax-b,f(x))$，得：$\lambda^T c(x) + \nu^T(Ax - b) + \mu f(x) \geq \alpha \geq \mu p^*.$
4. **排除 $\mu = 0$（Slater 条件的关键作用）**
    - 若 $\mu = 0$，则 $\lambda^T c(x) + \nu^T(Ax - b) \geq 0\ (\forall x \in D)$。
    - 由 Slater 条件，存在 $x_S \in \text{int} D$ 满足 $c_i(x_S) < 0$（严格不等式可行），且 $Ax_S = b$（等式约束可行）。代入得 $\lambda^T c(x_S) \geq 0$，但 $\lambda \geq 0$ 且 $c_i(x_S) < 0$，故 $\lambda = 0$。
    - 此时 $\nu^T(Ax - b) \geq 0\ (\forall x \in D)$。因 A 行满秩，$x_S \in \text{int} D$，存在 e 使 $x_S + e \in D$ 且 $\nu^T Ae < 0$（否则 $\nu = 0$，与 $(\lambda,\nu,\mu) \neq 0$ 矛盾），但 $\nu^T(A(x_S + e) - b) = \nu^T Ae \geq 0$，矛盾。故 $\mu > 0$。
5. **强对偶性证明（$\mu > 0$ 时）** 令 $\lambda^* = \lambda/\mu$，$\nu^* = \nu/\mu$（$\lambda^* \geq 0$），则：
    $L(x,\lambda^*,\nu^*) = \frac{1}{\mu}(\lambda^T c(x) + \nu^T(Ax - b) + \mu f(x)) \geq p^*\ (\forall x \in D),$
    故 $g(\lambda^*,\nu^*) = \inf L(x,\lambda^*,\nu^*) \geq p^*$。由弱对偶性 $g(\lambda^*,\nu^*) \leq p^*$，得 $g(\lambda^*,\nu^*) = p^* = d^*$，即对偶最优解存在，强对偶成立。
### 通俗解释
- **集合分离**：把原始问题的可行 “松弛”（A）和比最优值还小的区域（B）用超平面分开，超平面的系数 $(\lambda,\nu,\mu)$ 对应对偶变量和目标权重。
- **z**：保证存在一个 “内部点”（严格满足不等式约束），使得分离超平面的目标权重 $\mu$ 不能为零（否则会矛盾）。
- **强对偶的本质**：$\mu > 0$ 时，对偶函数的下确界恰好等于原始最优值，且能取到（因为超平面分离的几何意义保证了对偶解存在）。
简言之，Slater 条件提供了一个 “严格可行点”，让对偶问题的最优解 “落地”（存在且等于原始解），避免了对偶间隙（弱对偶到强对偶的关键跨越）。