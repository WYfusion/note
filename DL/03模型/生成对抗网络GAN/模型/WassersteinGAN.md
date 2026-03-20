---
tags:
  - 机器学习
  - 深度学习
  - GAN
  - WGAN
  - Wasserstein距离
created: 2025-01-18
modified: 2025-01-18
difficulty: 中高
related:
  - [[生成对抗网络GAN/生成式神经网络]]
  - [[生成对抗网络GAN/生成对抗的训练算法]]
  - [[生成对抗网络GAN/散度(divergence)]]
---

> [!summary] 核心思想
> WGAN 通过 Wasserstein 距离替代传统 GAN 的 JS 散度，从根本上解决了训练不稳定、模式崩溃和评估指标不可靠等问题。

# Wasserstein GAN（WGAN）

## 概述

2017 年提出的 GAN 改进版本 WGAN，其核心贡献在于通过 **Wasserstein 距离（推土机距离）** 替代传统 GAN 使用的 *JS* 散度，从根本上解决了传统 GAN 训练不稳定、模式崩溃、评估指标不可靠等问题。

---

## JS 散度和 Wasserstein 距离对比

参见 [[生成对抗网络GAN/散度(divergence)]] 中的详细对比。

---

## 1. K-Lipschitz 连续性约束

### 评估 Wasserstein 距离

评估 $p_G$ 和 $p_{data}$ 之间的 Wasserstein 距离：

$$
W(p_G, p_{data}) = \max_{D \in 1\text{-Lipschitz}} \left\{ \mathbb{E}_{y \sim P_{data}} [D(y)] - \mathbb{E}_{y \sim P_G} [D(y)] \right\}
$$

### 1-Lipschitz 函数定义

这里的 $D \in 1\text{-Lipschitz}$ 说明 $D$ 这个函数是一个高度平滑的函数。

$K$-Lipschitz 函数定义为：对于 $K > 0$，

$$
||f(x_1) - f(x_2)|| \le K ||x_1 - x_2||
$$

### 作用

- 避免 $\mathbb{E}_{x \sim P_{data}} [D(y)] - \mathbb{E}_{z \sim P_G} [D(z)]$ 数值过大
- 导致判别器错误的计算出来了一个很大的距离，难以收敛
- 只有加了限制，才算叫做 WGAN

![[Pasted image 20250316225504.png|300]]

---

## 2. WGAN 的损失函数

### 以欧氏距离为例的 WGAN 损失函数

#### 判别器损失

$$
\min_D V_{\text{WGAN}}(D) = -\mathbb{E}_{x \sim p_{data}} [D(x)] + \mathbb{E}_{z \sim p_z} [D(G(z))]
$$

#### 生成器损失

$$
\min_G V_{\text{WGAN}}(G) = -\mathbb{E}_{z \sim p_z} [D(G(z))]
$$

### LSGAN 版本（最小二乘 GAN）

$$
\min_D V_{\text{LSGAN}}(D) = \frac{1}{2}\mathbb{E}_{\mathbf{x},\mathbf{x}_c \sim p_{\text{data}}(\mathbf{x},\mathbf{x}_c)}[(D(\mathbf{x},\mathbf{x}_c)-1)^2] + \frac{1}{2}\mathbb{E}_{\mathbf{z} \sim p_{\mathbf{z}}(\mathbf{z}),\mathbf{x}_c \sim p_{\text{data}}(\mathbf{x}_c)}[D(G(\mathbf{z},\mathbf{x}_c),\mathbf{x}_c)^2]
$$

$$
\min_G V_{\text{LSGAN}}(G) = \frac{1}{2}\mathbb{E}_{\mathbf{z} \sim p_{\mathbf{z}}(\mathbf{z}),\mathbf{x}_c \sim p_{\text{data}}(\mathbf{x}_c)}[(D(G(\mathbf{z},\mathbf{x}_c),\mathbf{x}_c)-1)^2]
$$

其中：
- $D(\text{真实}) \to 1$
- $D(\text{虚假}) \to 0$

### 生成器改进版本

$$
\min_G V_{\text{LSGAN}}(G) = \frac{1}{2}\mathbb{E}_{\mathbf{z} \sim p_{\mathbf{z}}(\mathbf{z}),\tilde{\mathbf{x}} \sim p_{\text{data}}(\tilde{\mathbf{x}})}[(D(G(\mathbf{z},\tilde{\mathbf{x}}),\tilde{\mathbf{x}})-1)^2] + \lambda ||G(\mathbf{z},\tilde{\mathbf{x}})-\mathbf{x}||_1
$$

其中：
- $\tilde{\mathbf{x}}$ 是输入的条件噪声
- $\mathbf{x}$ 是干净语音
- $\mathbf{z}$ 是设定好的分布输入

---

## 3. WGAN 的核心优势

### 1. 解决梯度消失

- 即使生成分布和真实分布完全不重叠
- Wasserstein 距离仍然能提供有意义的梯度
- 生成器可以持续改进

### 2. 训练稳定性提升

- 不再需要精心平衡生成器和判别器
- 可以使用更高的学习率

### 3. 有意义的损失值

- Wasserstein 距离的值直接反映生成质量
- 可以作为训练进度的指标

### 4. 缓解模式崩溃

- 通过平滑的梯度引导
- 生成器更容易覆盖整个真实分布

---

## 4. 实现关键

### 梯度惩罚（Gradient Penalty）

WGAN-GP 通过添加梯度惩罚项来实现 1-Lipschitz 约束：

$$
L = \text{WGAN Loss} + \lambda \mathbb{E}_{\hat{x} \sim \mathbb{P}_{\hat{x}}} [(||\nabla_{\hat{x}} D(\hat{x})||_2 - 1)^2]
$$

其中：
- $\hat{x}$ 是真实样本和生成样本的线性插值
- $\lambda$ 是惩罚系数（通常设为 10）

### 权重裁剪（Weight Clipping）

原始 WGAN 使用权重裁剪来实现 Lipschitz 约束：

$$
w \leftarrow \text{clip}(w, -c, c)
$$

但这种方法可能导致梯度消失或爆炸。

---

## 相关链接

- [[生成对抗网络GAN/生成式神经网络]] - GAN 基本框架
- [[生成对抗网络GAN/生成对抗的训练算法]] - 训练算法详解
- [[生成对抗网络GAN/散度(divergence)]] - 散度度量详解

## 参考资料

- Arjovsky, M., Chintala, S., & Bottou, L. (2017). Wasserstein GAN. *ICML*
- Gulrajani, I., et al. (2017). Improved Training of Wasserstein GANs. *NeurIPS* (WGAN-GP)
