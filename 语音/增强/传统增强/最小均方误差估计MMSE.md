### 最小均方误差估计定义（Minimum Mean Square Error Estimation）
假设有两个随机变量 $\mathrm{X}$ 和 $\mathrm{Y}$， 他们之间存在联合分布，其中 $\mathrm{Y}$ 为观测信号，利用观测信号 $\mathrm{Y}$ 对 $\mathrm{X}$ 进行估计得到 $\hat{\mathrm{X}}$ (为固定的数据了，不再像 $\mathrm{X}$ 是一个随机变量了)，令 $\mathrm{E}\left(\left|\mathrm{X-\hat{X}}\right|^2|\mathrm{Y}\right)$ 最小，即为最小均方误差估计。
因此对 $\mathrm{E}\left(\left|\mathrm{X-\hat{X}}\right|^2|\mathrm{Y}\right)$ 这个求取极值：
$${\frac{\partial E\left(\left|\mathbf{X}-{\hat{\mathbf{X}}}\right|^{2}\mid\mathbf{Y}\right)}{\partial{\hat{\mathbf{X}}}}}=-2E\left(\left({\hat{\mathbf{X}}}-\mathbf{X}\right)\mid\mathbf{Y}\right)=-2\left({\hat{\mathbf{X}}}-E(\mathbf{X}\mid\mathbf{Y})\right)=0$$
也即 $\hat{\mathrm{X}}=\mathrm{E}(\mathrm{X}|\mathrm{Y})$ 
具体的语音降噪任务中的应用：
![[Pasted image 20250914141555.png|700]]
也即仅先考虑第k个频率bin下的，作用是简化理解，实际上是一个向量的形式。
$$\hat{X}_k=\operatorname{E}(X_k\left|Y(\omega_k)\right)=\int X_kP(X_k\mid Y(\omega_k))dX_k$$
### 因此需要先计算出相关的概率密度函数：
关于高斯分布看[[高斯、复高斯分布]]
![[Pasted image 20250914161609.png]]
1. 根据幅度最小均方误差估计的原则可以得到：
![[Pasted image 20250914163038.png|700]]
其中第二行采用的是[[贝叶斯公式]]的通用形式可以得出。
    $P(Y(\omega_k)\mid x_k)P(x_k)=\int_0^{2\pi}P(Y(\omega_k)\mid x_k,\theta_x)P(x_k,\theta_x)d\theta_x$ 这一部分的引入是由于 $Y(\omega_k)$ 为复数，而 $x_k$ 是实数，不容易构建的，因此引入相角(可以积分掉)
带入即可有$$\hat{X}_{k}=\frac{\int_{0}^{\infty}\int_{0}^{2\pi}x_{k}P(Y(\omega_{k})\mid x_{k},\theta_{x})P(x_{k},\theta_{x})d\theta_{x}dx_{k}}{\int_{0}^{\infty}\int_{0}^{2\pi}P(Y(\omega_{k})\mid x_{k},\theta_{x})P(x_{k},\theta_{x})d\theta_{x}dx_{k}}$$
2. 已知 $Y(\omega_k)=X(\omega_k)+D(\omega_k)$ 其中的 $D(\omega_k)$ 是噪声谱，其均值为0，是高斯分布
所以有 $p(Y(\omega_{k})\mid x_{k},\theta_{x})=\frac{1}{\pi\lambda_{d}(k)}\exp\left\{-\frac{1}{\lambda_{d}(k)}|Y(\omega_{k})-X(\omega_{k})|^{2}\right\}$ 
假设 $x_k$ 和 $\theta_k$ 是相互独立的，则两者的概率密度函数可以直接相乘，有 $p(x_k,\theta_x)=\frac{x_k}{\pi\lambda_x(k)}\exp\left\{-\frac{x_k^2}{\lambda_x(k)}\right\}$ 
3. 带入 $\hat{X}_{k}$ 中有 $$\hat{X}_k=\sqrt{\lambda_k}\Gamma(1.5)\Phi(-0.5,1;-\nu_k)$$
其中 $\lambda_k=\frac{\lambda_x(k)\lambda_d(k)}{\lambda_x(k)+\lambda_d(k)}=\frac{\lambda_x(k)}{1+\xi_k}$ 、$v_k=\frac{\xi_k}{1+\xi_k}\gamma_k$ 、 后验信噪比 $\gamma_k=\frac{Y_k^2}{\lambda_d(k)}$ 、 先验信噪比 $\xi_k=\frac{\lambda_x(k)}{\lambda_d(k)}$ 、 $\Gamma$ 是Gamma函数 、 $\Phi$ 是合流超几何函数
4. 进一步化简可得：$$\hat{X}_k=\frac{\sqrt{\pi}}{2}\frac{\sqrt{\nu_k}}{\gamma_k}\exp\left(-\frac{\nu_k}{2}\right)\left[(1+\nu_k)I_o\left(\frac{\nu_k}{2}\right)+\nu_kI_1\left(\frac{\nu_k}{2}\right)\right]Y_k$$
也即系统函数是一个关于先验($\xi_k$)/后验($\gamma_k$)信噪比的增益函数
其中先验($\xi_k$)对噪声抑制起主要的作用，并且MMSE的方法对先验($\xi_k$)的波动比较敏感，因此需要对先验($\xi_k$)需要进行准确且较为平滑的估计
5. 判决导引法（Decision-Directed）
$\xi_k(m)=\frac{E\{X_k^2(m)\}}{\lambda_d(k,m)}$ 第m帧的先验信噪比，$\begin{aligned}\xi_{k}(m)&=\frac{E\left\{Y_{k}^{2}(m)-D_{k}^{2}(m)\right\}}{\lambda_{d}(k,m)}\\&=\frac{E\left\{Y_{k}^{2}(m)\right\}}{\lambda_{d}(k,m)}-\frac{E\left\{D_{k}^{2}(m)\right\}}{\lambda_{d}(k,m)}\\&=E\{\gamma_{k}(m)\}-1\end{aligned}$，也就是说先验($\xi_k$)可以从先验信噪比公式来推导出来，也可以使用后验信噪比推导出来。
所以可以使用加权的形式计算出当前帧的**估计先验($\xi_k$)**，$\hat{\xi}_k(m)=\alpha\frac{\hat{X}_k^2(m-1)}{\lambda_d(k,m-1)}+(1-\alpha)\max[\gamma_k(m)-1,0]$，其中的 $\mathrm{0<\alpha<1}$ ，初始值 $\hat{\xi}_{k}(0)=\alpha+(1-\alpha)\max[\gamma_{k}(0)-1,0]$
之后引入限制参数 $\xi_{\min}$ 保障最小先验信噪比，有 $\hat{\xi}_{k}(m)=\max\left[\alpha\frac{\hat{X}_{k}^{2}(m-1)}{\lambda_{d}(k,m-1)}+(1-\alpha)\max[\gamma_{k}(m)-1,0],\xi_{\min}\right]$

### 基于VAD的噪声估计
先验知识：能量比较小的语音帧，通常是噪声帧
#### 流程
1. 设定一个SNR阈值 $\theta$
2. 计算语音前M帧的平均能量作为噪声能量 $E_n=\sum_\limits k\lambda_d(k)$ 
3. for t=1:N 对每一帧进行遍历，计算每帧的能量 $E_s$ 并计算信噪比 $\mathrm{SNR=E_s/E_n}$ ，$E_{\mathrm{s}}=\sum_\limits{k}\lambda_{\mathrm{Y}}(k)$ ，如若 $\mathrm{SNR}<\theta$ 则进行更新 $\lambda_{\mathrm{d}}(k)=\mu\lambda_{\mathrm{d}}(k)+(1-\mu)\lambda_Y(k)$ 噪声


### MMSE 与 维纳滤波之间的区别
![[Pasted image 20250914200547.png|1600]]
$\gamma_k - 1 = \frac{|X_k|^2 - \sigma_N^2}{\sigma_N^2}$（$|X_k|^2$是带噪语音功率，$\sigma_N^2$是噪声功率），反映 **“当前带噪语音中，超出噪声的能量占噪声的比例”**：
- $\gamma_k - 1$越小 → 当前带噪语音中 “语音能量” 越弱，甚至被噪声完全淹没；
- $\gamma_k - 1$越大 → 当前带噪语音中 “语音能量” 越强，噪声占比低。
##### “保语音” 比 “极致噪声抑制” 更重要
语音增强的目标是 **“让增强后的语音更‘可用’”**（比如能被识别、能流畅通话），而非 “把噪声压到极致但语音残缺”。如果为了 “干净” 而丢失语音细节，得不偿失：
- 噪声抑制到极致，但语音失真→ 识别系统可能完全无法工作，通话对方也听不清内容。
- 容忍一点噪声残留，但语音完整流畅→ 识别和通话体验都会更好。

### Log-MMSE
$E\{(\log X_k-\log\hat{X}_k)^2\}$ 根据最小均方误差估计：$\begin{aligned}&\log\hat{X}_{k}=E\{\log X_{k}|Y(\omega_{k})\}\\&\hat{X}_{k}=\exp(E\{\log X_{k}\mid Y(\omega_{k})\})\end{aligned}$，但是直接计算比较困难，且当前形式与矩母函数形式很贴近，因此引入矩母函数
![[Pasted image 20250914221405.png|600]]
通过泰勒展开后再求导后引入的 $t=0$ 仅保留 $E(x)$ 
1. 令 $Z_k=\log X_k$ ，矩母函数 $\exp[\mu Z_k]$ 于 $\Phi_{Z_k|Y(\omega_k)}(\mu)=E\{\exp[\mu Z_k]|Y(\omega_k)\}=E\left\{X_{k}^{\mu}\mid Y(\omega_{k})\right\}$ 中(其中的 $\mu$ 就是上图的 $t$，$Z_k=\log X_k$ 就是上图的 $x$)，于是有 $E\{\log X_k\mid Y(\omega_k)\}=\frac{d}{d\mu}\Phi_{Z_k\mid Y(\omega_k)}(\mu)|_{\mu=0}$ 
2. 对 $\Phi_{Z_{k}|Y(\omega_{k})}(\mu)$ 进行推导计算，利用贝叶斯公式：$$\begin{aligned}\Phi_{Z_{k}|Y(\omega_{k})}(\mu)&=E\left\{X_{k}^{\mu}\mid Y(\omega_{k})\right\}\\&=\frac{\int_{0}^{\infty}\int_{0}^{2\pi}x_{k}^{\mu}p(Y(\omega_{k})\mid x_{k},\theta_{x})p(x_{k},\theta_{x})d\theta_{x}dx_{k}}{\int_{0}^{\infty}\int_{0}^{2\pi}p(Y(\omega_{k})\mid x_{k},\theta_{x})p(x_{k},\theta_{x})d\theta_{x}dx_{k}}\\&=\lambda_k^{\mu/2}\Gamma\left(\frac{\mu}{2}+1\right)\Phi\left(-\frac{\mu}{2},1;-\nu_k\right)\end{aligned}$$
3. 对 $\mu$ 求导并置 $\mu=0$ 得：$$E\{\log X_k\mid Y(\omega_k)\}=\frac{1}{2}\log\lambda_k+\frac{1}{2}\log\nu_k+\frac{1}{2}\int_{\nu_k}^\infty\frac{e^{-t}}{t}dt$$
4. 带入 $\hat{X}_{k}=\exp(E\{\log X_{k}\mid Y(\omega_{k})\})$  有$$\begin{aligned}\hat{X}_{k}&=\frac{\xi_{k}}{\xi_{k}+1}\exp\left\{\frac{1}{2}\int_{\nu_{k}}^{\infty}\frac{e^{-t}}{t}dt\right\}Y_{k}\\&\triangleq G_{LSA}(\xi_{k},\nu_{k})Y_{k}\end{aligned}$$
这里的 $G_{LSA}(\xi_{k},\nu_{k})$ 就是设计的系统函数。
公式 $Ei(x)=\int_x^\infty\frac{e^{-x}}{x}dx\approx\frac{e^x}{x}\sum_\limits k\frac{k!}{x^k}$ 可以展开
近似计算：$$\left.\int_{\nu(n,t)}^\infty\frac{e^{-t}}{t}\mathrm{d}t\approx\left\{\begin{array}{cc}\nu(n,t)<0.1&-2.3*log_{10}(\nu(n,t))-0.6\\0.1\leq\nu(n,t)<1&-1.544*log_{10}(\nu(n,t))+0.166\\\nu(n,t)>1&10^{-0.52*v(n,t)-0.26}\end{array}\right.\right.$$

### 幅度平方估计
可以由矩母函数推导出$$\begin{aligned}\hat{X}_{k}^{2}&=E[X_{k}^{2}|Y_{k}]\\& \hat{X}_{k}^{2}=\frac{\xi_{k}}{1+\xi_{k}}(\frac{1+v_{k}}{\gamma_{k}})Y_{k}^{2}\\&H_{k}=\sqrt{\frac{\xi_{k}}{1+\xi_{k}}(\frac{1+v_{k}}{\gamma_{k}})}\end{aligned}$$`