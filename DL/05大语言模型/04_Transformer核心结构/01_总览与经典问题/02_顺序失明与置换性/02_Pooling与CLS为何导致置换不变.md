---
tags:
  - LLM/架构
created: 2026-03-29
updated: 2026-03-29
---

# Pooling 与 CLS 为何导致置换不变

## 问题定义

为什么无位置 self-attention 只是“跟着重排”，但一旦加上 pooling 或集合式聚合，顺序信息就会彻底丢失？

## 直觉解释

置换等变意味着：你把 token 顺序打乱，输出向量也只是跟着打乱。此时顺序还没有被彻底抹掉，因为“哪一行对应哪个位置”这件事仍然存在。可一旦你把所有行再压成一个向量，那个“哪一行是哪一行”的索引信息就没了。

## 形式化推导

设 self-attention 输出为 $Y \in \mathbb{R}^{L \times d}$，且对任意置换矩阵 $P$ 满足

$$
Y(PX) = PY(X)
$$

如果后续聚合器是平均池化：

$$
z(X) = \frac{1}{L}\sum_{i=1}^{L} Y_i(X)
$$

则对置换后的输入有

$$
z(PX)
= \frac{1}{L}\sum_{i=1}^{L} (PY(X))_i
= \frac{1}{L}\sum_{i=1}^{L} Y_{\pi(i)}(X)
= \frac{1}{L}\sum_{i=1}^{L} Y_i(X)
= z(X)
$$

其中 $P$ 对应某个置换 $\pi$，而 $(PY(X))_i = Y_{\pi(i)}(X)$ 只是在重排行。因为求和或求平均对重排不敏感，所以 pooling 会把“置换等变”进一步变成“置换不变”。

## 什么是集合式 CLS 聚合器

> [!note] 术语说明
> 这里的“集合式 CLS 聚合器”不是固定论文术语，而是一个描述性说法：当 `[CLS]` token 在**没有显式位置锚点**的情况下读取整段内容时，它更像是在对一个 token 集合做 readout，而不是在对一个有顺序的序列做 readout。

先看标准做法。设我们在序列前面加入一个可学习的 `CLS` 向量 $c$，输入写成：

$$
\tilde{X} = [c; x_1; x_2; \dots; x_L]
$$

做完一层 self-attention 后，`CLS` 位置的输出可以写成：

$$
y_{\mathrm{cls}} = \alpha_0 v_{\mathrm{cls}} + \sum_{i=1}^{L} \alpha_i v_i
$$

其中

$$
\alpha_i = \operatorname{softmax}\left(\frac{q_{\mathrm{cls}}^\top k_i}{\sqrt{d_k}}\right)
$$

也就是说，`CLS` 会用自己的 query 去读取所有 token 的 key/value，把整段内容压成一个全局表示。

如果我们**只重排内容 token**，而不改变 `CLS` 本身，并且模型里**没有位置编码**，那么：

- `CLS` 自己的 query 不变；
- 内容 token 的 key/value 只是跟着重排；
- attention 权重也只会跟着同样的方式重排。

设重排对应置换 $\pi$，则有：

$$
k_i' = k_{\pi(i)}, \quad v_i' = v_{\pi(i)}, \quad \alpha_i' = \alpha_{\pi(i)}
$$

于是新的 `CLS` 输出变成：

$$
\begin{aligned}
y_{\mathrm{cls}}'
&= \alpha_0 v_{\mathrm{cls}} + \sum_{i=1}^{L} \alpha_i' v_i' \\
&= \alpha_0 v_{\mathrm{cls}} + \sum_{i=1}^{L} \alpha_{\pi(i)} v_{\pi(i)} \\
&= \alpha_0 v_{\mathrm{cls}} + \sum_{i=1}^{L} \alpha_i v_i \\
&= y_{\mathrm{cls}}
\end{aligned}
$$

这说明：对于内容 token 的重排，`CLS` 读出的全局向量不变。此时它的行为，本质上就和“对一个无序元素集合做注意力池化”类似，所以这里把它叫作“集合式 CLS 聚合器”。

需要特别区分的是：**标准 BERT/ViT 里的 `[CLS]` 往往会和显式位置编码一起用。** 一旦位置进入模型，`CLS` 读到的就不再只是“有哪些内容”，还包括“这些内容在什么位置、以什么顺序出现”。这时它就不再是严格意义上的集合聚合器了。

## 工程意义

这就是“为什么 pooling 会让顺序信息全部丢失”的严格原因：不是 pooling 把某个顺序特征压弱了，而是它对输入排列天生对称。若前面算子又没有打破这种对称性，那么输出就必然对置换不敏感。

也因此，很多序列任务不能直接把无位置 self-attention 后接平均池化了事；必须**先通过位置机制或带顺序感**的聚合器，把“谁在前谁在后”写进表示。

## 常见误解

> [!warning] 常见误解
> - “pooling 只是把信息压缩了一下，不会改变性质。” 不对。它会改变对称性，从等变变成不变。
> - “CLS 一定保留顺序信息。” 不对。若 CLS 与其他 token 的交互本身没有位置锚点，它也可能退化成集合读取器。

## 例子或反例

序列 `A B C` 和 `C B A` 在无位置 self-attention 下会得到两组互为重排的 token 表示。若再对三行求平均，这两组输出完全相同，于是顺序差异被数学上彻底消掉。

## 相关链接

- [[01_置换等变性_形式化证明|置换等变性：形式化证明]]
- [[03_位置编码到底补回了什么|位置编码到底补回了什么]]
- [[01_绝对位置编码_Sinusoidal_Learnable|绝对位置编码]]
