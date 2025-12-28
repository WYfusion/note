# Softmax 激活函数

## 1. 定义与公式
**Softmax** 函数通常用于多分类神经网络的**输出层**。它将一个 $K$ 维的实数向量（Logits）映射为一个 $K$ 维的概率分布，使得所有元素的和为 1，且每个元素都在 $(0, 1)$ 之间。

**公式**：
对于输入向量 $z = [z_1, z_2, \dots, z_K]$，第 $i$ 个元素的输出为：
$$ \text{Softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} $$

**数值稳定性**：
在实际计算中，为了防止 $e^{z_i}$ 溢出，通常会减去输入向量中的最大值：
$$ \text{Softmax}(z)_i = \frac{e^{z_i - \max(z)}}{\sum_{j=1}^{K} e^{z_j - \max(z)}} $$

## 2. 导函数
Softmax 的导数是一个雅可比矩阵（Jacobian Matrix）。
令 $a_i = \text{Softmax}(z)_i$。

**导数公式**：
$$ \frac{\partial a_i}{\partial z_j} = \begin{cases} a_i(1 - a_i), & \text{if } i = j \\ -a_i a_j, & \text{if } i \neq j \end{cases} $$
可以用 Kronecker delta 符号表示为：
$$ \frac{\partial a_i}{\partial z_j} = a_i (\delta_{ij} - a_j) $$

## 3. 优缺点分析

### 优点
1.  **概率解释**：输出直接对应类别的概率，非常适合多分类任务。
2.  **可微性**：处处可导，便于反向传播。
3.  **马太效应**：指数运算会拉大元素之间的差异，使得最大的 Logit 对应的概率显著增加，有助于模型做出确定的预测。

### 缺点
1.  **计算量**：涉及指数运算和全局求和，当类别数 $K$ 非常大时（如词表大小为 10万+），计算开销很大（此时常用 Hierarchical Softmax 或 Noise Contrastive Estimation 优化）。
2.  **竞争性**：各元素之间是竞争关系（和为 1），一个变大必然导致其他变小。
