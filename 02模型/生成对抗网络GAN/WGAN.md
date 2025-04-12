![[散度(divergence)#^baf974]]
#k-Lipsschitz 
评估 $p_{G}$​ 和 $p_{data}$​ 之间的Wasserstein距离$\mathop {\max }\limits_{D\in1-Lipschitz}\left\{E_{y\thicksim P_{data}}[D(y)]-E_{y\thicksim P_G}[D(y)]\right\}$，这里的$D\in1-Lipschitz$说明$D$这个函数是一个高度平滑的函数$K-Lipsschitz$函数定义为对于$K>0,||f(x1) – f(x2)||≤K||x1 – x2||$,作用是避免$E_{x\thicksim P_{data}}[D(y)]-E_{z\thicksim P_G}[D(z)]$数值过大，导致判别器错误的计算出来了一个很大的距离，难以收敛。只有加了限制，才算叫做WGAN
![[Pasted image 20250316225504.png|300]]
