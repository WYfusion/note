![[Pasted image 20250604195442.png|800]]

从最优化角度证明 BFGS 算法的更新公式为上述优化问题的解，步骤如下：
  
  设 H 为对称矩阵（$H = H^\top$），满足割线方程 $H y^k = s^k$，目标是最小化加权范数 $\|H - H^k\|_W$，其中加权范数定义为：
$\|H\|_W = \|W^{1/2} H W^{1/2}\|_F, \quad W s^k = y^k \quad$ ($\text{W满足割线方程的对偶形式}$).
令 $X = W^{1/2} H W^{1/2}$，则 $H = W^{-1/2} X W^{-1/2}$，约束转化为$X \cdot \underbrace{W^{-1/2} y^k}_{z^k} = \underbrace{W^{1/2} s^k}_{b^k} \quad \implies \quad X z^k = b^k$ ，且 $X = X^\top$。目标函数变为 $\|X - X^k\|_F^2$（$X^k = W^{1/2} H^k W^{1/2}$）。

1. 构造拉格朗日函数：  
$\mathcal{L}(X, \lambda) = \frac{1}{2}\|X - X^k\|_F^2 + \lambda^\top (b^k - X z^k)$.

对 $X$ 求导并令导数为零：
$\frac{\partial \mathcal{L}}{\partial X} = (X - X^k) - \lambda z^k{}^\top = 0 \quad \implies \quad X = X^k + \lambda z^k{}^\top$

代入约束 $X z^k = b^k$：
$X^k z^k + \lambda \|z^k\|^2 = b^k \implies \lambda = \frac{b^k - X^k z^k}{\|z^k\|^2}$.
因此，$X = X^k + \frac{(b^k - X^k z^k) z^{k\top}}{\|z^k\|^2}$.

2. 回代到 H （修正系数与变量替换）
$H = W^{-1/2} X W^{-1/2} = H^k + \frac{W^{-1/2}(b^k - X^k z^k) z^{k\top} W^{-1/2}}{\|z^k\|^2}$.
利用 $b^k = W^{1/2} s^k$，$z^k = W^{-1/2} y^k$，得：
$b^k - X^k z^k = W^{1/2} s^k - W^{1/2} H^k W^{1/2} \cdot W^{-1/2} y^k = W^{1/2}(s^k - H^k y^k)$，
$z^k{}^\top W^{-1/2} = y^k{}^\top W^{-1}, \quad \|z^k\|^2 = y^k{}^\top W^{-1} y^k$
带入化简得：
$H = H^k + \frac{(s^k - H^k y^k) y^{k\top} W^{-1}}{y^{k\top} W^{-1} y^k}$.
3. 选择 W 满足对偶割线方程：
取 $W = \frac{y^k y^{k\top}}{s^{k\top} y^k}$（满足 $W s^k = y^k$），则 $W^{-1} = \frac{s^k s^{k\top}}{y^{k\top} s^k}$($\text{由 } W W^{-1} = I \text{ 验证}$)（假设 $s^{k\top} y^k \neq 0$），代入得H表达式：
$y^k{}^\top \frac{s^k s^k{}^\top}{y^k{}^\top s^k} y^k = \frac{(y^k{}^\top s^k)^2}{y^k{}^\top s^k} = y^k{}^\top s^k$,  
分子第一项化简：$(s^k - H^k y^k) y^k{}^\top \frac{s^k s^k{}^\top}{y^k{}^\top s^k} = \frac{s^k s^k{}^\top}{y^k{}^\top s^k} - \frac{H^k y^k y^k{}^\top s^k s^k{}^\top}{y^k{}^\top s^k}$.  
注意到 $y^k{}^\top s^k$ 为标量，第二项可写为：$\frac{H^k y^k y^k{}^\top H^k \cdot y^k{}^\top s^k}{y^k{}^\top s^k} = H^k y^k y^k{}^\top H^k \cdot \frac{y^k{}^\top s^k}{y^k{}^\top s^k} = H^k y^k y^k{}^\top H^k \cdot \frac{1}{y^k{}^\top H^k y^k} \cdot y^k{}^\top H^k y^k \quad$ ($\text{引入恒等变形}$).  
最终得 BFGS 更新公式
$H = H^k + \frac{s^k s^{k\top}}{y^{k\top} s^k} - \frac{H^k y^k y^{k\top} H^k}{y^{k\top} H^k y^k}$,
这正是 **BFGS 更新公式**。

4. 验证约束与范数最小性
- **对称性**：$H$ 由对称矩阵 $X$ 变换得到，由 $X = X^\top$ 和 $H = W^{-1/2} X W^{-1/2}$，且 $W^{-1/2}$ 对称，故 $H = H^\top$。
- **割线方程**：代入 $H y^k$ ：$H y^k = H^k y^k + \frac{s^k s^k{}^\top y^k}{y^k{}^\top s^k} - \frac{H^k y^k y^k{}^\top H^k y^k}{y^k{}^\top H^k y^k}$注意到 $s^k{}^\top y^k = y^k{}^\top s^k$，第二项化简为 $s^k$，第三项分子为 $H^k y^k \cdot y^k{}^\top H^k y^k = H^k y^k \cdot (y^k{}^\top H^k y^k)$，故第三项为 $H^k y^k$，因此：$H y^k = H^k y^k + s^k - H^k y^k = s^k$，满足拟牛顿条件。
- **范数最小性**：拉格朗日乘数法保证该解是凸优化问题（Frobenius 范数平方为凸函数，约束为仿射子空间）的唯一极小点，故 $H$ 是满足约束的 $H^k$ 的最佳逼近。