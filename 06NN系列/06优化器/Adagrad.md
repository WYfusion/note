$Adagrad$优化器(自适应学习率)

$$
{\Large \displaystyle\begin{aligned}&s_t=s_{t-1}+g(w_t)\cdot g(w_t)\\&w_{t+1}=w_t-\frac\alpha{\sqrt{s_t+\varepsilon}}\cdot g(w_t)\end{aligned}}
$$

$\varepsilon$ 是$10^{-7}$作用是避免分母为0。

##### 缺点：学习率下降的太快，可能还未收敛就停止训练了