在生成对抗网络（GAN）中，**divergence（散度）** 是衡量生成器（Generator）***产生的数据分布***  $p_{G}$​ 与***真实数据分布*** $p_{data}$​ ​ 之间**差异**的数学工具。GAN的核心目标是让生成器最小化这种散度，从而使生成的数据分布尽可能接近真实数据分布。以下是关于GAN中散度的详细解释： ^fbbd83

![[Pasted image 20250316203433.png|600]]
图中原始随机分布经过生成器后得到的$p_{G}$分布与真实的$p_{data}$的分布需要尽可能的一致。 ^553f97

---
### **1. 散度的基本概念**
- **定义**：散度是一种衡量两个概率分布差异的函数。它不是严格意义上的“距离”（因为不满足对称性和三角不等式），但能反映分布间的相似性。
- **GAN中的散度**：通过对抗训练，生成器试图最小化 $p_{G}$​ 和 $p_{data}$​  之间的散度，而判别器则间接帮助估计这种差异。

---
### **2. GAN中的散度形式**
#### 2.1 **JS散度**（Jensen-Shannon Divergence）
在原始GAN的论文中，生成器和判别器的对抗训练被证明等价于最小化 **Jensen-Shannon散度（JS Divergence）**：$$\mathrm{JS}(P_{\mathrm{data}}\parallel P_G)=\frac{1}{2}\mathrm{KL}\left(P_{\mathrm{data}}\parallel\frac{P_{\mathrm{data}}+P_G}{2}\right)+\frac{1}{2}\mathrm{KL}\left(P_G\parallel\frac{P_{\mathrm{data}}+P_G}{2}\right)$$其中 $KL$ 是 **Kullback-Leibler散度**。 ^28d1be
- 使用JS散度中**GAN中的问题**：
    - $P_{G}$和$P_{data}$这两个分布之间的重合很少
    - $P_{G}$和$P_{data}$是二维低维特征，当在高维空间中展示时的重叠往往少到可以忽略。
    - Sample的个数太少了，哪怕有重合区域，从二维角度上也可以用一条线将重合区域两部分给区分出来。
    ![[Pasted image 20250316214047.png|300]]
    ![[Pasted image 20250316214635.png|600]]
    $P_{G_{1}}$明显比$P_{G_{0}}$更好，但是无法从$P_{G_{0}}$迭代到$P_{G_{1}}$
- JS散度本身的性质造成的危险：
    - 当 $p_{G}$​ 和 $p_{data}$​  无重叠时，JS散度恒为 $log⁡2$，导致梯度消失（判别器"太好了"，生成器无法有效更新）。
    - 这是原始GAN训练不稳定的重要原因之一。

#### 2.2 **KL散度**（Kullback-Leibler Divergence）
$$\mathrm{KL}(P\parallel Q)=\sum P(x)\log\frac{P(x)}{Q(x)}$$
**问题**：不对称性$\mathrm{KL}(P\parallel Q)\neq\mathrm{KL}(Q\parallel P)$，且当$Q(x)=0$而$P(x)>0$时发散（趋于无穷大），导致训练不稳定。

#### 2.3 **Wasserstein距离**（Earth-Mover Distance）
针对于JS散度无法考量当 $p_{G}$​ 和 $p_{data}$​  无重叠时的梯度消失问题(数值不再变化了)，使用Wasserstein距离可以通过距离的形式体现分布的相似度。

下图中的$d_0$明显比$d_1$更大，也更差。可以反应若两个分布没有重合时的差异距离，也就是可以慢慢迭代了。
![[Pasted image 20250316223632.png|600]] ^baf974