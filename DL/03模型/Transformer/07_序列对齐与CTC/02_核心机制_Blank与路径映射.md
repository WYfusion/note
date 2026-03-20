## 引入 Blank 符号

为了解决 $T \gg U$ 的问题，CTC 引入了一个特殊的 **空字符 (Blank Token)**，通常记为 $\epsilon$ 或 `<blank>`，也常用 `-` 表示。
输出词表扩展为：$V' = V \cup \{\epsilon\}$。

模型在每一个时间步 $t$ (1 到 $T$) 都会输出一个关于 $V'$ 的概率分布。

## 路径映射 (Many-to-One)

CTC 定义了一个映射函数 $\mathcal{B}$ (Many-to-One Mapping)，将帧级别的**路径 (Path)** 映射到最终的**标签序列 (Label Sequence)**。

**映射规则**：
1. **合并连续重复**：连续相同的非空字符会被合并为一个。
2. **移除 Blank**：在合并后，移除所有的 `<blank>` 符号。

### 示例演示
假设目标输出是单词 `cat`。
以下几条路径（假设 $T=6$）都可以映射到 `cat`：

- **路径 1**：`c a a t - -` $\xrightarrow{\text{合并}}$ `c a t - -` $\xrightarrow{\text{去空}}$ `c a t`
- **路径 2**：`c - a - t -` $\xrightarrow{\text{合并}}$ `c - a - t -` $\xrightarrow{\text{去空}}$ `c a t`
- **路径 3**：`c c a a t t` $\xrightarrow{\text{合并}}$ `c a t` $\xrightarrow{\text{去空}}$ `c a t`
- **路径 4**：`- - c a t -` $\xrightarrow{\text{合并}}$ `- - c a t -` $\xrightarrow{\text{去空}}$ `c a t`

### 关键区分：如何表示重复字符？
如果目标词是 `bee` (包含 `ee` 重复)，CTC 强制要求两个 `e` 之间**必须至少隔一个 `<blank>`**。
- `b e e` $\to$ `be` (错误)
- `b e - e` $\to$ `bee` (正确)
- `b b e - e e` $\to$ `bee` (正确)

这种机制巧妙地解决了“长音”与“重复字”的区分问题：
- 长音“啊——”：`a a a a` $\to$ `a`
- 重复字“啊啊”：`a a - a a` $\to$ `aa`

## 状态独立性假设 (Conditional Independence)

这是 CTC 最大的数学假设：
$$P(\pi|X) = \prod_{t=1}^T P(\pi_t | x_t)$$
即：**在给定输入 $X$ 的情况下，每一帧的输出是相互独立的。**

这意味着 CTC 模型（如 Transformer Encoder 输出接 CTC Head）在 $t$ 时刻的预测，不显式依赖于 $t-1$ 时刻预测出的**标签**（Token），只依赖于 Encoder 提取的**上下文特征**。
这与自回归（Auto-regressive）模型（预测 $y_t$ 依赖 $y_{<t}$）形成了鲜明对比。
