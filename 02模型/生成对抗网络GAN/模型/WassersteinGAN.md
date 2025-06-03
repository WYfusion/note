2017 年提出的 GAN 改进版本WGAN，其核心贡献在于通过 **Wasserstein 距离（推土机距离，Wasserstein Distance）** 替代传统 GAN 使用的 *JS* 散度（*Jensen-Shannon Divergence*），从根本上解决了传统 GAN 训练不稳定、模式崩溃（Mode Collapse）、评估指标不可靠等问题。

---
$JS$ 散度和 $Wasserstein$ (推土机)距离之间的对比
![[散度(divergence)#^baf974]]
#k-Lipsschitz 
评估 $p_{G}$​ 和 $p_{data}$​ 之间的$Wasserstein$距离$\mathop {\max }\limits_{D\in1-Lipschitz}\left\{E_{y\thicksim P_{data}}[D(y)]-E_{y\thicksim P_G}[D(y)]\right\}$，这里的$D\in1-Lipschitz$说明$D$这个函数是一个高度平滑的函数$K-Lipsschitz$函数定义为对于$K>0,||f(x1) – f(x2)||≤K||x1 – x2||$,作用是避免$E_{x\thicksim P_{data}}[D(y)]-E_{z\thicksim P_G}[D(z)]$数值过大，导致判别器错误的计算出来了一个很大的距离，难以收敛。只有加了限制，才算叫做WGAN
![[Pasted image 20250316225504.png|300]]
以欧氏距离为例的WassersteinGAN的损失函数
$$\begin{aligned}\min_{D}V_{\mathrm{LSGAN}}(D)&=\frac{1}{2}\mathbb{E}_{\mathbf{x},\mathbf{x}_{c}\sim p_{\mathrm{data}}(\mathbf{x},\mathbf{x}_{c})}[(D(\mathbf{x},\mathbf{x}_{c})-1)^{2}]+&\text{D(真实)->1}\\&+\frac{1}{2}\mathbb{E}_{\mathbf{z}\sim p_{\mathbf{z}}(\mathbf{z}),\mathbf{x}_{c}\sim p_{\mathrm{data}}(\mathbf{x}_{c})}[D(G(\mathbf{z},\mathbf{x}_{c}),\mathbf{x}_{c})^{2}]&\text{D(虚假)->0}\end{aligned}$$
$$\min_{G}V_{\mathrm{LSGAN}}(G)=\frac{1}{2}\mathbb{E}_{\mathbf{z}\sim p_{\mathbf{z}}(\mathbf{z}),\mathbf{x}_{c}\sim p_{\mathrm{data}}(\mathbf{x}_{c})}[(D(G(\mathbf{z},\mathbf{x}_{c}),\mathbf{x}_{c})-1)^{2}].\quad\text{D(虚假)->1}$$对生成器部分再次更新改进
$$\min_{G}V_{\mathrm{LSGAN}}(G)=\frac{1}{2}\mathbb{E}_{\mathbf{z}\sim p_{\mathbf{z}}(\mathbf{z}),\tilde{\mathbf{x}}\sim p_{\mathrm{data}}(\tilde{\mathbf{x}})}[(D(G(\mathbf{z},\tilde{\mathbf{x}}),\tilde{\mathbf{x}})-1)^{2}]+\lambda\left\|G(\mathbf{z},\tilde{\mathbf{x}})-\mathbf{x}\right\|_{1}$$
$\tilde{\mathbf{x}}$是输入的条件噪声，$x$是干净语音，$z$是设定好的分布输入。