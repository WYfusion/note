#### 思想
交叉熵衡量预测概率分布与真实分布的差异，通过最小化负对数似然提升分类置信度，是分类任务的核心损失。
#### 数学公式
$$
C(\mathbf{y}, \mathbf{\hat{y}}) = -\frac{\sum\limits_{i=1}^{N}\sum\limits_{c=1}^{C} y_{ic} \log(\hat{y}_{ic}) }{N}
$$
$$
\begin{aligned}

                C ={} & \text{number of classes} \\

                N ={} & \text{batch size} \\

            \end{aligned}
$$
其中，$y_{ic}$​为one-hot编码的真实标签，$\hat{y}_{ic}$​为softmax输出的预测概率。
#### 适用范围
- **多分类任务**：如图像分类、自然语言处理。
- **概率校准场景**：需输出类别概率时 。

#### 效果

- **优点**：梯度性质良好，加速模型收敛；与softmax结合增强类别区分度。
- **缺点**：对错误的高置信预测敏感；需配合标签平滑缓解过拟合