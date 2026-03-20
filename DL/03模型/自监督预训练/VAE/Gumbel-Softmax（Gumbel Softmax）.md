---
tags:
  - 机器学习
  - 深度学习
  - Gumbel-Softmax
  - 离散采样
  - 可微采样
  - VAE
created: 2025-01-18
modified: 2025-01-18
difficulty: 中高
related:
  - [[自监督预训练/Self-supervised Learning Framework]]
  - [[生成对抗网络GAN/生成式神经网络]]
  - [[生成对抗网络GAN/散度(divergence)]]
---

# Gumbel-Softmax（又名 Concrete）笔记

## 0. 一句话直觉

当你想从**离散类别**里“抽一个类”（比如 one-hot 选择），但又想像训练神经网络那样**反向传播**
#Gumbel-Softmax 提供了一条路：  
先用Gumbel噪声把“采样”写成 $\arg\max$，再用一个带温度的 softmax 把 $\arg\max$ **平滑成可导近似**。

---

## 1. 背景：离散采样为什么难训练？

假设你有 $K$ 个类别，概率为 $\pi_1,\dots,\pi_K$（$\sum_k \pi_k=1,\pi_k\ge 0$）。  
你想采样一个 one-hot 向量 $y\in\{0,1\}^K$：

$$
y \sim \mathrm{Categorical}(\pi), \quad y_k=1 \text{ 表示选中第 }k\text{ 类}
$$

问题在于：采样/取最大值（如 $\arg\max$）是离散操作，**对参数不可导**。  
例如你用网络输出 logits $a\in\mathbb{R}^K$，再得到 $\pi=\mathrm{softmax}(a)$，此时

$$
\text{sample}(y) = \text{离散随机变量} \Rightarrow \nabla_a \,\mathbb{E}[f(y)] \text{ 很难直接算}
$$

Gumbel-Softmax就是为了解决“离散采样不可导”这个核心痛点。

---

## 2. Gumbel分布：噪声从哪里来

标准Gumbel(0,1)分布的分布函数CDF为：

$$
F(g)=\exp\left(-e^{-g}\right)
$$

一个非常常用的采样方式：

$$
u \sim \mathrm{Uniform}(0,1),\quad g=-\log(-\log u)
$$

**为什么这样能采到Gumbel？（思路）**  
令 $g=-\log(-\log u)$，则
$$
\Pr(g\le t)=\Pr(-\log(-\log u)\le t)=\Pr(u\le e^{-e^{-t}})=e^{-e^{-t}}=F(t)
$$
所以 $g$ 确实服从标准Gumbel分布。

---

## 3. Gumbel-Max Trick：把“按概率采样类别”写成 $\arg\max$

### 3.1 结论（最重要的式子）

给定类别概率（或未归一化权重）$\pi_1,\dots,\pi_K$，采样 $K$ 个独立Gumbel噪声 $g_k\sim \mathrm{Gumbel}(0,1)$，定义：

$$
k^*=\arg\max_{k}\left(\log \pi_k + g_k\right)
$$

那么 $k^*$ 的分布满足：

$$
\Pr(k^*=k)=\frac{\pi_k}{\sum_\limits{j=1}^{K}\pi_j}
$$

若 $\pi$ 已经是概率（和为1），那就是 $\Pr(k^*=k)=\pi_k$。  
这叫 **Gumbel-Max**：用“加噪声再取最大”实现按指定概率采样。

### 3.2 为什么成立

记 $s_k=\log \pi_k$。我们关心事件：

$$
k^*=k \iff s_k+g_k \ge s_j+g_j,\ \forall j\ne k
$$

标准结论（可查任何Gumbel-Max推导）是：对独立Gumbel噪声，$\max_j(s_j+g_j)$ 的分布可以被解析，最终得到

$$
\Pr(k^*=k)=\frac{e^{s_k}}{\sum_j e^{s_j}}=\frac{\pi_k}{\sum_j \pi_j}
$$

直观上可以这样记：  
Gumbel噪声把“取最大”变成“**随机取最大**”，而这个随机性**刚好**对应softmax概率。

> 你只需要牢牢记住：  
> “Categorical采样” = “logits + Gumbel噪声 后 argmax”。

### 3.3 更通俗的原理：把采样想成“赛跑”（指数赛跑 $\rightarrow$ Gumbel-Max）

如果你觉得“加噪声再取最大”为啥能采样很抽象，可以换个更直观但严格的视角：**指数赛跑（exponential race）**。

#### 第一步：先理解“指数赛跑为什么等价于按权重抽样”

对每个类别 $k$，生成一个“到达时间”：

$$
T_k \sim \mathrm{Exp}(\text{rate}=\pi_k)
$$

然后选最先到达的那个类别：

$$
k^*=\arg\min_k T_k
$$

这时有一个非常经典的结论：

$$
\Pr(k^*=k)=\frac{\pi_k}{\sum_j \pi_j}
$$

**直觉**：$\pi_k$ 越大，指数分布的“平均等待时间”越短，所以越容易最先到达。  
**简单推导（给你一个能看懂的证明）**：指数分布满足

$$
f_{T_k}(t)=\pi_k e^{-\pi_k t},\quad \Pr(T_j>t)=e^{-\pi_j t}
$$

“$k$ 赢”,即k对应的是最短用时的概率等于：

$$
\Pr(k\text{赢})
=\int_{0}^{\infty} f_{T_k}(t)\prod_{j\ne k}\Pr(T_j>t)\,dt
=\int_{0}^{\infty}\pi_k e^{-\pi_k t}\prod_{j\ne k}e^{-\pi_j t}\,dt
$$

把指数项合并：

$$
\Pr(k\text{赢})
=\int_{0}^{\infty}\pi_k e^{-(\sum_j\pi_j)t}\,dt
=\frac{\pi_k}{\sum_j\pi_j}
$$

##### 补充：为什么会出现 $\prod_{j\ne k}\Pr(T_j>t)$（把每一步拆开）

你看到的关键一步是：

$$
\Pr(k\text{赢})
=\int_{0}^{\infty} f_{T_k}(t)\prod_{j\ne k}\Pr(T_j>t)\,dt
$$

这里的 $\prod_{j\ne k}\Pr(T_j>t)$ 其实是在表达一件很直观的事：

> 如果第 $k$ 个选手在时刻 $t$ 到达，那么他想赢，必须要求所有其他选手都“还没到”（也就是都比 $t$ 慢）。

把它写成事件更清晰。令

$$
A_k=\{k\text{赢}\}=\{T_k < T_j,\ \forall j\ne k\}
$$

对任意固定的 $t$，在条件 $T_k=t$ 下，“$k$ 赢”等价于：

$$
A_k \mid (T_k=t)\quad \Longleftrightarrow\quad \bigcap_{j\ne k}\{T_j>t\}
$$

因此（连续型随机变量版本的全概率公式）：

$$
\Pr(A_k)
=\int_{0}^{\infty}\Pr(A_k\mid T_k=t)\, f_{T_k}(t)\,dt
=\int_{0}^{\infty}\Pr\Big(\bigcap_{j\ne k}\{T_j>t\}\ \Big|\ T_k=t\Big)\, f_{T_k}(t)\,dt
$$

接下来两步都依赖于“独立”：

1) **给定 $T_k=t$ 不影响其他人**（独立性）  
对任意 $j\ne k$，
$$
\Pr(T_j>t\mid T_k=t)=\Pr(T_j>t)
$$

2) **“所有人都还没到”的联合概率 = 各自概率的乘积**（独立事件相乘）  
因为不同 $j$ 的 $T_j$ 相互独立，所以事件 $\{T_j>t\}$ 也相互独立：
$$
\Pr\Big(\bigcap_{j\ne k}\{T_j>t\}\Big)=\prod_{j\ne k}\Pr(T_j>t)
$$

把两步合起来，就得到：

$$
\Pr\Big(\bigcap_{j\ne k}\{T_j>t\}\ \Big|\ T_k=t\Big)=\prod_{j\ne k}\Pr(T_j>t)
$$

这就是你疑惑的那一项 $\prod_{j\ne k}\Pr(T_j>t)$ 的来源。

最后，如果 $T_j\sim \mathrm{Exp}(\text{rate}=\pi_j)$，它的“还没到达（生存函数）”为：

$$
\Pr(T_j>t)=e^{-\pi_j t}
$$

于是：

$$
\prod_{j\ne k}\Pr(T_j>t)=\prod_{j\ne k}e^{-\pi_j t}
$$

到这里你已经知道：**只要能构造这样的“赛跑时间”并取最小，就等价于按 $\pi$ 采样。**

#### 第二步：把“取最小时间”改写成“加噪取最大”

指数分布可以由均匀分布生成：

$$
u\sim \mathrm{Uniform}(0,1),\quad E=-\log u \sim \mathrm{Exp}(1)
$$

若令

$$
T_k=\frac{E_k}{\pi_k},\quad E_k\sim\mathrm{Exp}(1)
$$

则 $T_k\sim\mathrm{Exp}(\text{rate}=\pi_k)$，因此 $\arg\min_k T_k$ 就是按 $\pi$ 采样。

现在把 $\arg\min$ 变成 $\arg\max$：因为 $\log(\cdot)$ 单调，

$$
\arg\min_k T_k
=\arg\min_k \frac{E_k}{\pi_k}
=\arg\min_k \left(\log E_k - \log \pi_k\right)
=\arg\max_k \left(\log \pi_k - \log E_k\right)
$$

而 $-\log E_k$（当 $E_k\sim \mathrm{Exp}(1)$）的分布就是标准Gumbel噪声（等价写法是 $g_k=-\log(-\log u_k)$）。  
于是得到：

$$
k^*=\arg\max_k(\log\pi_k+g_k)
$$

这就是 **Gumbel-Max** 的“为什么”。

### 3.4 例子推算（手算一遍你就会了）

下面用一个具体例子把“从 $u$ 生成 $g$，再取最大”完整算一遍。

假设有 3 个类别，概率：

$$
\pi=[0.10,\ 0.30,\ 0.60]
$$

先算 $\log\pi$（自然对数）：

$$
\log\pi\approx[-2.3026,\ -1.2040,\ -0.5108]
$$

#### 例子1：给定一组随机数，算出最终选哪个类

从均匀分布采样（这里直接给出一组具体值）：

$$
u=[0.20,\ 0.70,\ 0.40]
$$

用公式 $g=-\log(-\log u)$ 得到Gumbel噪声：

- $u_1=0.20$：$\log u_1=-1.6094$，$-\log u_1=1.6094$，$\log(1.6094)=0.4759$，所以 $g_1=-0.4759$
- $u_2=0.70$：$\log u_2=-0.3567$，$-\log u_2=0.3567$，$\log(0.3567)=-1.0310$，所以 $g_2=1.0310$
- $u_3=0.40$：$\log u_3=-0.9163$，$-\log u_3=0.9163$，$\log(0.9163)=-0.0870$，所以 $g_3=0.0870$

把分数 $s_k=\log\pi_k+g_k$ 算出来：

$$
s\approx[
-2.3026-0.4759,\ 
-1.2040+1.0310,\ 
-0.5108+0.0870
]
=[-2.7785,\ -0.1730,\ -0.4238]
$$

取最大分数对应的类别是第2类（因为 $-0.1730$ 最大），所以这次采样结果：

$$
k^* = 2
$$

注意：虽然第3类概率最大（0.60），但随机性可能让第2类赢一次；这正是“随机采样”的含义。

#### 例子2：再来一组随机数，看是否更偏向大概率类

再取一组：

$$
u=[0.90,\ 0.20,\ 0.30]
$$

快速计算（同样用 $g=-\log(-\log u)$，这里给出近似值）：

$$
g\approx[2.2513,\ -0.4759,\ -0.1856]
$$

分数：

$$
s=\log\pi+g\approx[
-2.3026+2.2513,\ 
-1.2040-0.4759,\ 
-0.5108-0.1856
]=[-0.0513,\ -1.6799,\ -0.6964]
$$

最大的是第1类，于是这次采样结果：

$$
k^* = 1
$$

你会发现：单次结果可能选到小概率类，但**重复很多次**，统计频率会趋近于 $[0.10,0.30,0.60]$。

---

## 4. 从 Gumbel-Max 到 Gumbel-Softmax：把 $\arg\max$ 变可导

### 4.1 问题：$\arg\max$ 还是不可导

Gumbel-Max虽然能采样，但 $\arg\max$ 依旧是离散操作，反向传播仍然断掉。

### 4.2 软化：带温度的softmax近似

将 $\arg\max$ 用 softmax 近似，并加入温度 $\tau>0$：

$$
y = \mathrm{softmax}\left(\frac{\log \pi + g}{\tau}\right)
$$

更常见实现是用 logits $a$（未归一化）：

$$
y = \mathrm{softmax}\left(\frac{a + g}{\tau}\right)
$$

其中：
- $a\in\mathbb{R}^K$：网络输出（logits）
- $g\in\mathbb{R}^K$：独立Gumbel噪声
- $\tau$：温度，控制“接近one-hot的程度”

softmax展开写就是：

$$
y_k=\frac{\exp((a_k+g_k)/\tau)}{\sum_\limits{j=1}^{K}\exp((a_j+g_j)/\tau)}
$$

### 4.3 温度 $\tau$ 的含义（非常关键）

- $\tau \to 0$：$y$ 趋近 one-hot（非常尖锐），更像真实离散采样，但梯度更不稳定  
- $\tau$ 较大：$y$ 更平滑（更“软”），梯度更稳定，但更偏离离散变量

因此训练时常用 **退火（annealing）**：先大后小。

---

## 5. “硬采样但能训练”：Straight-Through（直通）技巧

有时你希望前向过程是真正的离散 one-hot（比如送进下游模块），但反向仍需要梯度。常用做法：

1. 先算软样本 $y$（可导）
2. 前向用硬样本：
$$
y^{\text{hard}}=\mathrm{one\_hot}(\arg\max_k y_k)
$$
3. 反向把梯度当作来自软样本（直通估计）：

$$
y^{\text{st}} = y^{\text{hard}} + \text{sg}[y - y^{\text{hard}}]
$$

其中 $\text{sg}[\cdot]$ 表示停止梯度：前向值保留，反向梯度为0。  
这样前向是离散的，反向仍沿着 $y$ 的梯度走（是近似/有偏，但很常用）。

---

## 6. 用在模型里时，到底替代了什么？

### 6.1 替代“离散潜变量的采样”

如果你有离散潜变量 $z\in\{1,\dots,K\}$，其后验或策略分布由网络给出：

$$
q_\phi(z=k|x)=\pi_k(x)
$$

原本你要采样 $z \sim \mathrm{Categorical}(\pi(x))$，不可导。  
现在你用 Gumbel-Softmax 得到一个近似 one-hot 向量 $y$，把它当作“离散选择”的连续替身。

### 6.2 与“变分（VAE）”的关系

VAE里困难点是后验 $p_\theta(z|x)$ 难算，用 $q_\phi(z|x)$ 近似并做可导采样（连续高斯用重参数化）。  
当 $z$ 是**离散**的（Categorical），就不能用高斯重参数化；Gumbel-Softmax就成了“离散版重参数化”的常用选择。

一句话：

- 连续隐变量：$z=\mu+\sigma\epsilon$（高斯重参数化）VAE
- 离散隐变量：$y=\mathrm{softmax}((a+g)/\tau)$（Gumbel-Softmax）

---

## 7. 实操小抄：怎么用（算法步骤）

给定 logits $a$（来自网络）：

1. 采样 $u_k\sim\mathrm{Uniform}(0,1)$
2. 得到Gumbel噪声 $g_k=-\log(-\log u_k)$
3. 计算软样本：
$$
y=\mathrm{softmax}\left(\frac{a+g}{\tau}\right)
$$
4. （可选）用直通得到硬样本参与前向：
$$
y^{\text{hard}}=\mathrm{one\_hot}(\arg\max y),\quad y^{\text{st}}=y^{\text{hard}}+\text{sg}[y-y^{\text{hard}}]
$$

---

## 8. 常见坑与注意事项

- **温度太小**：梯度不稳定、训练抖动；可先用较大 $\tau$，再逐步退火
- **温度太大**：一直很“软”，模型学到的是连续混合而不是离散选择
- **直通估计是有偏的**：好用但不是严格无偏梯度估计；用于工程实践更常见
- **类别很多时的数值稳定**：softmax用logits实现、注意减去最大值（框架通常已处理）

---

## 9. 连续潜变量 vs 离散潜变量：什么时候需要 Gumbel-Softmax？

你之前的理解方向基本对：  
**连续潜变量**通常可以用VAE的“高斯重参数化”直接训练；  
但当潜变量是**离散的**，采样不可导，就需要额外技巧——Gumbel-Softmax是最常见的一种，但不是唯一一种。

下面用“怎么判断 + 怎么选方法”的方式讲清楚。

### 9.1 怎么判定潜变量是连续还是离散？

判定标准不是“数据长什么样”，而是你对 $z$ 的数学设定（支持集）：

- **连续**：$z\in\mathbb{R}^d$（能取任意实数），常见高斯  
  编码器通常输出 $\mu(x),\sigma(x)$
- **离散**：$z\in\{1,\dots,K\}$ 或 $y\in\{0,1\}^K$（只能取有限/可数集合）  
  编码器通常输出 logits/概率 $\pi(x)$，你想“抽一个类别/one-hot”

从实现角度一句话判断：

- 你有没有在前向过程中做“抽类别/argmax/one-hot/查表索引”？  
  有 → 离散；没有（只是实数运算）→ 连续。

### 9.2 连续潜变量：VAE为什么好训练？

连续高斯潜变量可以写成可导的采样形式（重参数化）：

$$
z=\mu_\phi(x)+\sigma_\phi(x)\odot \epsilon,\quad \epsilon\sim \mathcal{N}(0,I)
$$

随机性在 $\epsilon$，而 $\mu,\sigma$ 可导，所以梯度能正常回传（见：`VAE（Variational Autoencoder）.md`）。

### 9.3 离散潜变量：为什么会卡住？

离散采样通常长这样：

$$
z \sim \mathrm{Categorical}(\pi(x)),\quad z\in\{1,\dots,K\}
$$

或者等价的 one-hot：

$$
y=\mathrm{one\_hot}(\arg\max(\cdot))
$$

这里的“采样/argmax/one-hot”是离散的，梯度无法直接对 logits 回传。

### 9.4 离散潜变量的三条常见路线：Gumbel-Softmax / VQ-VAE / REINFORCE

1) **Gumbel-Softmax（连续松弛）**  
把离散 one-hot 用一个“接近 one-hot 的连续向量”近似：
$$
y=\mathrm{softmax}\left(\frac{a+g}{\tau}\right)
$$
优点：端到端、梯度稳定、实现简单；  
代价：它是近似（$\tau$ 不是 0），梯度带偏。

2) **VQ-VAE（向量量化 + 直通估计）**  
不是“按概率抽类别”，而是“把连续特征映射到最近的码本向量”（离散索引），并用直通估计训练。  
优点：在图像/音频压缩与高质量生成里非常常见，离散表征像“码本词表”；  
代价：需要码本与额外损失（commitment/codebook），生成时常配合离散先验。  

3) **REINFORCE / Score Function（无偏，但高方差）**  
理论上能给出无偏梯度，但方差大，工程训练更难，因此在深度生成模型里相对少用。

### 9.5 什么时候“必须用” Gumbel-Softmax？

更准确的说法是：当你满足下面两点时，Gumbel-Softmax很合适：

- 你确实需要离散选择（类别/符号），并且希望端到端训练
- 你希望“离散采样”对 logits 可微（用连续松弛来近似）

如果你的离散表示更像“码本/压缩/词表”，并且你更在意生成质量与稳定性，那么 VQ-VAE 往往更合适。

---

## 10. 一页总结（主线关系）

1. 离散采样不可导  
2. Gumbel-Max：$\arg\max(\log\pi + g)$ 实现按 $\pi$ 采样  
3. Gumbel-Softmax：用温度softmax把 $\arg\max$ 平滑成可导  
4. 可选直通：前向硬、反向软  
5. 常用于“离散VAE/离散潜变量”的可导训练
