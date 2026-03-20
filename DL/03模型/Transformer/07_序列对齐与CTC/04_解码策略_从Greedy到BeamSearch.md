## 1. 贪心解码 (Greedy Search)

最简单的解码策略。不考虑上下文，直接在每个时间步取概率最大的 token。
$$\pi^*_t = \arg\max_k P(k|x_t)$$
$$Y_{pred} = \mathcal{B}(\pi^*_1, ..., \pi^*_T)$$

- **优点**：计算速度极快，$O(T)$。
- **缺点**：由于 CTC 的独立性假设，只能利用局部信息。虽然 $\mathcal{B}$ 映射能解决部分问题，但缺乏语言模型的约束，容易产生拼写错误或语法不通的句子。

## 2. 束搜索 (Beam Search)

为了找到全局最优解，我们需要维护 top-K 条候选路径。
但在 CTC 中，标准的 Beam Search 比较复杂，因为**不同的路径可能映射到同一个结果**（例如 `a-b` 和 `ab-` 都映射为 `ab`）。

### 核心区别：前缀合并 (Prefix Merge)
在 CTC Beam Search 中，我们跟踪的是**输出前缀 (Output Prefix)** 的概率，而不是单纯的路径概率。
对于每个前缀（如 "hello"），我们需要分别维护两个概率：
1. **$P_{nb}$ (Ending in Non-Blank)**：以非空字符结尾的路径概率。
2. **$P_{b}$ (Ending in Blank)**：以 Blank 结尾的路径概率。

**状态转移逻辑**：
当扩展前缀 $Y$ 加上新字符 $c$ 时：
- 如果 $c = \text{blank}$：更新 $P_b$（不改变文本内容）。
- 如果 $c = \text{last\_char}(Y)$：
    - 只能从 $P_b(Y)$ 转移过来（代表 `aa` 重复）。
    - 如果从 $P_{nb}(Y)$ 转移，会被合并掉（代表 `a` 长音）。
- 如果 $c \neq \text{last\_char}(Y)$：可以从 $P_b(Y)$ 和 $P_{nb}(Y)$ 同时转移。

## 3. 浅层融合 (Shallow Fusion) 与 语言模型

由于 CTC 缺乏语言模型（Language Model, LM）能力，通常在 Beam Search 阶段引入外部 LM。

$$\text{Score}(Y) = \log P_{CTC}(Y|X) + \lambda \cdot \log P_{LM}(Y) + \beta \cdot \text{WordCount}(Y)$$

- **$\\lambda$**：LM 权重，修正语法。
- **$\\beta$**：长度惩罚，防止生成过短的句子（因为 CTC 倾向于短结果）。

这种结合方式使得 CTC 模型在工业界（如语音搜索）非常流行，因为它结合了声学模型的强健性和语言模型的纠错能力。
