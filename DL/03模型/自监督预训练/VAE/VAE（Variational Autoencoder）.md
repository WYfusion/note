# 变分自编码器（VAE）笔记

## 0. 先给一句话直觉

VAE想学一个**能生成数据的模型**：不是把输入硬压成一个点，而是学一个“潜在分布”，让生成和表示都可控。

---

## 1. 记号与参数（先把名字认清）

- $x$：观测数据（输入样本），维度 $D$
- $z$：潜变量（隐空间表示），维度 $d$
- $p(z)$：潜变量先验，一般取 $\mathcal{N}(0,I)$
- $p_\theta(x|z)$：解码器定义的**生成分布**
- $q_\phi(z|x)$：编码器定义的**近似后验**
- $\theta$：解码器参数
- $\phi$：编码器参数
- $f_\phi(\cdot)$ / $g_\theta(\cdot)$：编码器/解码器网络

这些记号贯穿全文，后面每个公式都会用到。

---

## 2. VAE要解决什么问题

普通自编码器（AE）是确定性映射：

$$
z = f_\phi(x), \quad \hat{x} = g_\theta(z)
$$

它能重构，但**隐空间没有概率结构**，随机采样 $z$ 往往解码失败。
VAE引入**概率分布**，把“生成”作为第一目标，同时仍能学习表示。

---

## 3. 生成模型：先采样 $z$ 再生成 $x$

VAE假设数据是这样产生的：

1. 从先验采样潜变量：
$$
z \sim p(z)=\mathcal{N}(0,I)
$$
2. 用解码器生成数据：
$$
x \sim p_\theta(x|z)
$$

**解码器输出的不是一个确定的 $x$，而是一个分布的参数。**

常见两种情况：

- $x$ 是连续值：$p_\theta(x|z)=\mathcal{N}(g_\theta(z), \sigma_x^2 I)$
  负对数似然对应MSE（平方误差）
- $x$ 是二值：$p_\theta(x|z)=\text{Bernoulli}(g_\theta(z))$
  负对数似然对应BCE（交叉熵）

---

## 4. 编码器和解码器到底在做什么

### 4.1 编码器（Encoder）

- **输入**：$x$（比如一张图）
- **输出**：$z$ 的分布参数（通常是高斯）
  - 均值 $\mu_\phi(x)$
  - 标准差 $\sigma_\phi(x)$（实现时常输出 $\log\sigma^2$）

**直观理解**：
编码器不是说“这张图就是某个点”，而是说“这张图应该落在这个概率团里”。

### 4.2 采样（重参数化）

直接从 $q_\phi(z|x)$ 采样无法反向传播，所以做：

$$
z = \mu_\phi(x) + \sigma_\phi(x)\odot\epsilon,\quad \epsilon\sim\mathcal{N}(0,I)
$$

这样随机性在 $\epsilon$，梯度可通过 $\mu,\sigma$ 回传。

### 4.3 解码器（Decoder）

- **输入**：$z$
- **输出**：$p_\theta(x|z)$ 的参数（比如均值图像）

**直观理解**：
解码器不是“吐出一张图”，而是“定义一张图该长什么样的概率分布”。

---

## 5. 为什么要最大化对数似然

模型希望“已观测数据在模型下越可能越好”。
定义似然：$L(\theta)=p_\theta(x)$，对数似然为 $\log p_\theta(x)$。

对于数据集 $\{x_i\}_{i=1}^N$：

$$
L(\theta)=\prod_{i=1}^N p_\theta(x_i),
\quad \log L(\theta)=\sum_{i=1}^N \log p_\theta(x_i)
$$

取对数的好处：

- 乘积变求和，计算稳定
- 便于求导优化

所以训练时通常最大化对数似然。

---

## 6. 真实后验难算：引入变分推断

真实后验：

$$
p_\theta(z|x)=\frac{p_\theta(x,z)}{p_\theta(x)}
$$

其中 $p_\theta(x)$ 是积分，难算。
因此用可计算的 $q_\phi(z|x)$ 来近似它。

---

## 7. ELBO推导：把目标变成可优化

从KL散度出发：

$$
\text{KL}(q_\phi(z|x)\|p_\theta(z|x)) \ge 0
$$

代入 $p_\theta(z|x)=\frac{p_\theta(x,z)}{p_\theta(x)}$ 并整理：

$$
\log p_\theta(x)
=
\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x,z) - \log q_\phi(z|x)]
 + \text{KL}(q_\phi(z|x)\|p_\theta(z|x))
$$

定义证据下界（ELBO）：

$$
\text{ELBO}=
\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x,z) - \log q_\phi(z|x)]
$$

所以：

$$
\log p_\theta(x)\ge \text{ELBO}
$$

最大化ELBO就是“在可计算的范围内最大化对数似然”。

---

## 8. ELBO拆解：重构项 + 正则项

利用 $p_\theta(x,z)=p(z)p_\theta(x|z)$：

$$
\text{ELBO}
=
\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]
- \text{KL}(q_\phi(z|x)\|p(z))
$$

含义非常清楚：

- **重构项**：让解码器输出的分布能“解释”输入 $x$
- **KL正则项**：让后验接近先验，隐空间结构统一、可采样

这两个目标同时推动训练。

---

## 9. 高斯参数化与KL闭式解

常见设定：

$$
q_\phi(z|x)=\mathcal{N}(\mu_\phi(x), \text{diag}(\sigma_\phi^2(x)))
$$

当 $p(z)=\mathcal{N}(0,I)$ 时：

$$
\text{KL}(q_\phi(z|x)\|p(z))
= \frac{1}{2}\sum_{i=1}^{d}(\mu_i^2 + \sigma_i^2 - \log\sigma_i^2 - 1)
$$

这个闭式公式使训练高效且稳定。

---

## 10. 训练目标与完整流程（一步一步）

训练目标（最大化ELBO，等价最小化负ELBO）：

$$
\mathcal{L}_{\text{VAE}}
=
\underbrace{-\mathbb{E}_{q_\phi}[\log p_\theta(x|z)]}_{\text{重构损失}}
\;+\;
\underbrace{\text{KL}(q_\phi(z|x)\|p(z))}_{\text{正则项}}
$$

训练流程：

1. 输入 $x$，编码器输出 $\mu,\log\sigma^2$
2. 重参数采样 $z=\mu+\sigma\odot\epsilon$
3. 解码器输出 $p_\theta(x|z)$ 的参数
4. 计算重构损失与KL损失
5. 反向传播更新 $\theta,\phi$

---

## 11. 推断与生成：两条使用路径

- **推断/重构**：$x \rightarrow q_\phi(z|x)\rightarrow z \rightarrow p_\theta(x|z)$
  用来重构输入、做表示学习
- **生成**：$z \sim p(z)\rightarrow p_\theta(x|z)$
  用来生成新样本

---

## 12. 应用与常见问题（简要）

**应用**

1. 生成建模（图像、语音、文本）
2. 异常检测（异常样本重构误差更大）
3. 表示学习（低维语义特征）
4. 缺失值补全

**常见问题**

- 生成模糊：高斯解码器偏平均
  解决：更强解码器、VAE-GAN、自回归解码器
- 后验塌陷：$q_\phi(z|x)$ 过度接近 $p(z)$
  解决：KL退火、Free Bits、改结构

---

## 13. 潜变量是连续还是离散？怎么判断、怎么选

隐变量（潜变量）是连续还是离散，不是数据本身决定的，而是**在建模时对 $z$ 的设定**决定的：
可以自行让 $z$ 取什么样的取值范围（支持集），它就是什么类型的随机变量。

### 13.1 连续潜变量（Continuous latent）

**定义**：$z$ 取值在连续空间（通常是 $\mathbb{R}^d$），常见假设是高斯：

$$
z \in \mathbb{R}^d,\quad p(z)=\mathcal{N}(0,I),\quad q_\phi(z|x)=\mathcal{N}(\mu_\phi(x),\mathrm{diag}(\sigma_\phi^2(x)))
$$

**特点（你能感受到的）**

- $z$ 是“实数向量”，可以做平滑插值（interpolation）
- 常用重参数化：$z=\mu+\sigma\odot\epsilon$（可直接反向传播）
- 更像“连续因素”（比如亮度、风格强度、姿态角度）

**典型方法**：本笔记介绍的标准VAE（连续高斯潜变量）。

### 13.2 离散潜变量（Discrete latent）

**定义**：$z$ 只能取离散集合中的值，例如类别索引：

$$
z \in \{1,\dots,K\},\quad q_\phi(z=k|x)=\pi_k(x)
$$

或 one-hot 向量 $y\in\{0,1\}^K$（等价表达）。

**特点**

- 表示更像“符号/词表/聚类ID”（例如对象类别、音素、码本索引）
- 生成往往更“利落”（不容易被连续高斯的平均化拖糊），但训练会遇到“采样不可导”

**离散潜变量不一定只能用 Gumbel-Softmax**，常见路线有三类：

1. **Gumbel-Softmax（连续松弛）**：把离散采样近似为可导的 soft one-hot
   适合端到端训练、类别数不太夸张的场景
   参考：[[Gumbel-Softmax（Gumbel Softmax）]]
2. **VQ-VAE（向量量化 + 直通估计）**：用码本把连续特征量化成离散索引
   特别常见于高质量生成/压缩（图像、音频），并且往往更稳定
   参考：[[VQ-VAE（Vector Quantized Variational Autoencoder，向量量化变分自编码器）]]
3. **REINFORCE/Score Function 等无偏梯度估计**：理论无偏，但方差大、训练难（工程上相对少用）

### 13.3 “如何判定”最实用的标准

从实现角度，你可以看编码器输出什么：

- 如果编码器输出的是 **$\mu(x),\sigma(x)$**，并用 $z=\mu+\sigma\epsilon$ 采样：这就是连续潜变量（VAE那一套）
- 如果编码器输出的是 **类别概率 $\pi(x)$ / logits**，并且你想“抽一个类别/one-hot”：这就是离散潜变量（需要 Gumbel-Softmax / VQ-VAE / 其他离散估计）

从建模意图角度：

- 你希望表示是“连续可插值的因素” → 更倾向连续潜变量
- 你希望表示是“离散符号/码本/可枚举选择” → 更倾向离散潜变量

---

## 14. 总结

1. 假设生成过程：$p(z)$ 先验 + $p_\theta(x|z)$ 解码器
2. 后验难算，用 $q_\phi(z|x)$ 近似
3. 最大化ELBO = 重构好 + 隐空间规整
4. 用重参数化解决采样不可导
5. 训练后可重构也可生成

