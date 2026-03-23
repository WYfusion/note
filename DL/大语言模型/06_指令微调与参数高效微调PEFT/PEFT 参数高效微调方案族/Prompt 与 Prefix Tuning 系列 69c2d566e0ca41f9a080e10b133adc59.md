# Prompt 与 Prefix Tuning 系列

这一族方法的共同思想：**不改动模型参数，而是在输入或每层注意力中插入可训练的"虚拟 token"**。

---

## 方法对比

| 方法 | 可训练参数位置 | 特点 |
| --- | --- | --- |
| **Prompt Tuning** | 仅输入嵌入层前拼接 soft tokens | 最简单，参数最少 |
| **Prefix Tuning** | 每个 Transformer 层的 KV 前拼接可训练前缀 | 影响更深，效果更好 |
| **P-Tuning v2** | 与 Prefix Tuning 类似，每层插入可训练前缀 | 针对 NLU 任务优化，中等模型效果好 |

---

## Prompt Tuning

在输入序列前拼接 $m$ 个可训练的 soft token embedding $P in mathbb{R}^{m times d}$：

$$
\text{Input} = [P_1, P_2, \dots, P_m, x_1, x_2, \dots, x_n]
$$

- 只有 $P$ 参与梯度更新，模型其余参数全部冻结
- 可训练参数量 = $m times d$（非常小）
- 在超大模型（>10B）上效果接近全参数微调

---

## Prefix Tuning

在每个 Transformer 层的 Key 和 Value 前拼接可训练前缀：

$$
K' = [P_K; K], \quad V' = [P_V; V]
$$

- 前缀 $P_K, P_V \in \mathbb{R}^{m \times d}$ 每层独立
- 实际训练时用一个更小的 MLP 生成前缀（重参数化），训练完后丢掉 MLP
- 影响每一层的注意力计算，表达力比 Prompt Tuning 更强

---

## P-Tuning v2

本质与 Prefix Tuning 相同，但针对 NLU（自然语言理解）任务做了工程优化：

- 在每层都加可训练前缀
- 对 330M~10B 规模模型效果显著
- 实现上与 Prefix Tuning 几乎一致

---

## 📂 子页面（叶子层，含代码与公式）

`子页面创建后补充`