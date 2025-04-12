 $RMSProp$优化器(自适应学习率)

$$
{\Large \displaystyle \begin{aligned}&s_t=\eta\cdot s_{t-1}+(1-\eta)\cdot g(w_t)\cdot g(w_t)\\&w_{t+1}=w_t-\frac\alpha{\sqrt{s_t+\varepsilon}}\cdot g(w_t)\end{aligned}}
$$
