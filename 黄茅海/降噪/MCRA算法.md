## 噪声谱估计
令 $x(n)$ 和 $d(n)$ 分别表示语音和不相关的加性噪声信号，其中 $n$ 是离散时间指标。观测信号 $y(n)$ （由 $y(n)=x(n)+d(n)$ 给出）通过应用窗口函数分为重叠帧，并使用短时傅里叶变换 ($\text{STFT}$) 进行分析。具体而言:$$Y(k,\ell)=\sum _{n=0}^{N-1} y(n+\ell M) h(n)e^{-j(2\pi/N)nk} \ \ \ \ \ \ \ \ \ \ \ \ \hbox{(1)}$$
其中 $k$ 是频点指标， $ℓ$ 是时间帧指标， $h$ 是大小为 $N$ 的分析窗口， $M$ 是时间上的帧更新步长。给定两个假设 $H_0(k,ℓ)$ 和 $H_1(k,ℓ)$ ，分别表示在第 $k$ 个子带的第 $ℓ$ 个帧中**不存在语音和存在语音**(这里是仅是描述一个事实)，我们得到:$$\eqalignno{H_{0}(k,\ell):Y(k,\ell)=& D(k,\ell) \cr H_{1}(k,\ell):Y(k,\ell)=& X(k,\ell)+D(k,\ell)&\mathrm{(2)}}$$
其中 $X(k,ℓ)$ 和 $D(k,ℓ)$ 分别表示干净信号和噪声信号的 $\text{STFT}$。令 $λ_d(k,ℓ)=E[|D(k,ℓ)|^2]$ 表示第 $k$ 个子带中噪声的方差。然后，一种常用的获取其估计值的技术是在语音缺失期间对含噪测量应用时间递归平滑。具体而言:$$\begin{aligned}&H_{0}^{\prime}(k,\ell):\hat{\lambda}_{d}(k,\ell+1)=\alpha_{d}\hat{\lambda}_{d}(k,\ell)+(1-\alpha_{d})\left|Y(k,\ell)\right|^{2}\\&H_{1}^{\prime}(k,\ell):\hat{\lambda}_{d}(k,\ell+1)=\hat{\lambda}_{d}(k,\ell)&\mathrm{(3)}\end{aligned}$$
其中 $α_d(0<α_d>1)$ 是平滑参数， $H^′_0$ 和 $H^′_1$ 分别表示假设的语音缺失和存在。这里，我们区分了用于**估计干净语音**的假设和**控制噪声频谱的自适应**的假设。显然，在存在语音（ $H_1$ ）时判定语音缺失（ $H_0$ ）在估计信号时比在估计噪声时更具破坏性。
因此，采用不同的决策规则，通常我们倾向于以比 $H^′_1$ 更高的置信度判定 $H_1$ ，即 $P(H_1|Y)≥P(H^′_1|Y)$令 $p^′(k,ℓ)≜P(H^′_1(k,ℓ)|Y(k,ℓ))$ 表示条件信号存在概率，则$1-p^{\prime}(k,\ell)$表示条件信号不存在概率。将(3)式上下相加则蕴含:$$\begin{aligned}\hat{\lambda_d}(k,\ell+1)&=\left[\alpha_d\hat{\lambda}_d(k,\ell)+(1-\alpha_d)\left|Y(k,\ell)\right|^2\right]\times(1-p^{\prime}(k,\ell)) \\&+\hat{\lambda}_d(k,\ell)p^{\prime}(k,\ell)\\&=\tilde{\alpha}_d(k,\ell)\hat{\lambda}_d(k,\ell)\\&+\left[1-\tilde{\alpha}_d(k,\ell)\right]\left|Y(k,\ell)\right|^2\ \ \ \ \ \ \ &\mathrm{(4)}\end{aligned}$$
其中$$\tilde{\alpha}_{d}(k,\ell)\triangleq\alpha_{d}+\left(1-\alpha_{d}\right)p^{\prime}(k,\ell)\quad(5)$$
是随时间变化的平滑参数。因此，可以通过对过去的谱功率值取平均值来估计噪声谱，并使用由信号存在概率调整的平滑参数$α_d$。
## 信号存在概率
子带中给定帧中的语音存在性取决于带噪语音的局部能量与其在指定时间窗口内的最小值之间的比率。设带噪语音的局部能量可以通过对其时间和频率上的 $\text{STFT}$ 幅值平方进行平滑处理来获得。在频率方面，我们使用一个窗口函数 $b$ ，其长度为 $2w+1$
$$S_{f}(k,\ell)=\sum _{i=-w}^{w} {b(i)\left \vert {Y(k-i,\ell)} \right \vert ^{2}} \quad  \quad  \quad   \quad{(6)}$$
时间上的平滑由一阶递归平均执行，由$$S(k,\ell)=\alpha _{s} S(k,\ell -1)+\left(1-\alpha _{s}\right) S_{f}(k,\ell)  \quad  \quad  \quad \quad{(7)}$$
给出，其中 $α_s(0<α_s<1)$ 是参数。搜索局部能量的最小值 $S_{min}(k,ℓ)$ 。首先，通过 $S_{min}(k,0)=S(k,0)$ 和 $S_{tmp}(k,0)=S(k,0)$ **初始化**最小值和临时变量 $S_{tmp}(k,ℓ)$ 。然后，**逐个样本比较**`局部能量`和`前一帧的最小值`，得出当前帧的最小值
$$\eqalignno{S_{\min }(k,\ell)=& \min \left \{ {S_{\min }(k,\ell-1),S(k,\ell)} \right \}& \hbox{(8)}\cr S_{tmp}(k,\ell)=&\min \left \{ {S_{tmp}(k,\ell-1),S(k,\ell)} \right \}& \hbox{(9)}}$$
每当读取了 $L$ 帧时，即 $ℓ$ 能被 $L$ 整除，就使用临时变量并由$$\eqalignno{S_{\min}(k,\ell)=&\min \left \{ {S_{tmp}(k,\ell-1),S(k,\ell)} \right \} & \hbox{(10)} \cr S_{tmp}(k,\ell)=& S(k,\ell)&\hbox{(11)}}$$
注：前 $L$ 帧时 $S_{min}(k,ℓ)$ 完全等同于 $S_{tmp}(k,ℓ)$ ，但是到达了第一个 $L$ 时有 $S_{tmp}(k,ℓ)≥S_{min}(k,ℓ)$ ，也就是说 $S_{tmp}(k,ℓ)$ 展示的是当前 $L$ 帧的最小值，而 $S_{min}(k,ℓ)$ 展示的是上一个 $L$ 窗的所有帧与当前 $L$ 窗前  $L\%ℓ$ 帧的最小值，因为 (10) 式。
(10) 和 (11)**再次初始化**，然后继续使用 (8) 和 (9) 搜索最小值。参数 $L$ 决定局部最小值搜索的分辨率。局部最小值基于至少 $L$ 帧但不超过 $2L$ 帧的窗口。窗口的长度控制“连续”语音期间的向上偏差和噪声水平增加时的向下偏差。根据前人及作者cohen针对不同说话者和环境条件的实验，合适的窗口通常为 $\text{0.5-1.5}$ 秒。
令 $S_r(k,ℓ)≜S(k,ℓ)/S_{min}(k,ℓ)$ 表示含噪语音的局部能量与其导出的最小值之间的比率。贝叶斯最小成本决策规则由(12)式给出:
$${{p\left(S_{r}\vert H_{1}\right)} \over {p\left(S_{r}\vert H_{0}\right)}} \quad \mathop{\gtrless}^{H^{\prime}_1}_{H^{\prime}_0} \quad {{c_{10} P\left(H_{0}\right)} \over {c_{01} P\left(H_{1}\right)}}   \quad  \quad  \quad  \quad  \quad{(12)}$$
其中 $P(H_0)$ 和 $P(H_1)$ 分别是语音缺失和存在的*先验*概率， $c_{ij}$ 是在 $H^′_j$ 时判定 $H^′_i$ 的成本。 Fig.1 显示了在 $−5 dB$ 分段信噪比下，针对高斯白噪声(a)和 F16 驾驶舱噪声(b)通过实验获得的条件概率密度函数的代表性示例 $p(S_r|H_0)$ 和 $p(S_r|H_1)$ 。由于似然比 $p(S_r|H_1)/p(S_r|H_0)$ 是单调函数**语音存在时 $S_r$ 更大概率取较大值，而语音不存在时 $S_r$ 更大概率取较小值**，Fig.1也展示了这一点
- 分子 $p(S_r | H_1)$ 随 $S_r$ 增大而增大；
- 分母 $p(S_r | H_0)$ 随 $S_r$ 增大而减小；
- 整体似然比 $\frac{p(S_r | H_1)}{p(S_r | H_0)}$ 随 $S_r$ 增大而严格递增。
因此 (12) 的叶斯决策规则中对似然比的阈值判断（$\frac{p(S_r | H_1)}{p(S_r | H_0)} \gtrless \text{阈值}$）可等价转化为对 $S_r$ 的阈值判断（$S_r \gtrless \delta$），其中 $\delta$ 是与原阈值对应的常数。可以表示为
$$S_{r}(k,\ell)\quad \mathop{\gtrless}^{H^{\prime}_1}_{H^{\prime}_0}\quad \delta \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad {(13)}$$
![[Pasted image 20250808170443.png|750]]


我们为 $p^′(k,ℓ)$ 提出以下估计器：$$\hat{p^{\prime}}(k,\ell)=\alpha_{p}\hat{p^{\prime}}(k,\ell-1)+(1-\alpha_{p})I(k,\ell) \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad {(14)}$$
其中 $α_p(0<α_p<1)$ 是平滑参数， $I(k,ℓ)$ 表示 (13) 中结果的指示函数，即，如果 $S_r(k,ℓ)>δ$ ，则为 $I(k,ℓ)=1$ ，否则为 $I(k,ℓ)=0$ 。该估计的优点有三方面。
- 第一， $δ$ 对环境噪声的类型和强度不敏感。
- 第二，当 $S_r<δ$ 时， $|Y|^2≫λ_d$ 的概率非常小。因此，在 $H^′_1$ 时错误地判断 $H^′_0$ 会导致估计噪声的增加的现象并不显著。
- 第三，可以利用连续帧中语音存在的强相关性（通过 $α_p$ ）。

