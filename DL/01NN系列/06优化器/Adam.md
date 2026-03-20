$Adam$优化器(自适应学习率)

$$
{\Large \displaystyle \begin{aligned}
&m_t=\beta_1\cdot m_{t-1}+(1-\beta_1)\cdot g(w_t)\quad\text{一阶动量} \\
&\nu_t=\beta_2\cdot\nu_{t-1}+(1-\beta_2)\cdot g(w_t)\cdot g(w_t)\quad\text{二阶动量} \\
&\hat{m}_t=\frac{m_t}{1-\beta_1^t}\quad\hat{\nu}_t=\frac{\nu_t}{1-\beta_2^t} \\
&w_{t+1}=w_t-\frac{\alpha}{\sqrt{\hat{\nu}_t+\varepsilon}}\hat{m}_t
\end{aligned}}
$$

$\beta_1$与$\beta_2$控制衰减速度的通常取$0.9$和$0.999$