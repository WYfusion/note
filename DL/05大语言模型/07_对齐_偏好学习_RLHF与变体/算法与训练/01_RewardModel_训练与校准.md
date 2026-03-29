# Reward Model (RM) 训练与校准

奖励模型（Reward Model, RM）是 RLHF（Reinforcement Learning from Human Feedback）流程中的核心组件。它的作用是**代理人类的偏好**，为生成模型的输出提供一个标量评分（Scalar Reward），从而指导强化学习算法（如 PPO）优化策略模型。

---

## 1. 核心原理与数学基础

### 1.1 建模目标
RM 的本质是一个**排序模型（Ranking Model）**，而非绝对评分模型。因为不同人类标注者对分数的绝对值定义可能不同（有人打分宽松，有人严格），但对“哪个更好”的相对判断通常更一致。

### 1.2 Bradley-Terry 模型 (BT Model)
这是 RM 训练的理论基石。假设给定输入 $x$（Prompt），有两个候选回答 $y_w$（Winner，被人类选中的）和 $y_l$（Loser，被人类拒绝的）。[[01_偏好对_排序_标注一致性#^110e71|数据对的偏好]]
BT 模型假设人类选择 $y_w$ 优于 $y_l$ 的概率与两者的奖励差值的 Sigmoid 函数成正比：

$$ P(y_w \succ y_l | x) = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) = \frac{1}{1 + e^{-(r_\phi(x, y_w) - r_\phi(x, y_l))}} $$

其中：
*   $r_\phi(x, y)$ 是参数为 $\phi$ 的奖励模型给出的标量分数。
*   $\sigma$ 是 Sigmoid 激活函数，将分值差映射到 $(0, 1)$ 区间表示概率。

### 1.3 损失函数推导 (Ranking Loss)
训练 RM 的目标是最大化人类偏好数据的似然概率。即最小化负对数似然（Negative Log-Likelihood）：

$$ \mathcal{L}_{RM}(\phi) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log P(y_w \succ y_l | x) \right] $$

代入 BT 模型公式：

$$ \mathcal{L}_{RM}(\phi) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right] $$

**直观理解**：
*   如果 RM 给 $y_w$ 的分值远高于 $y_l$，则 $r_w - r_l$ 很大，$\sigma(\cdot) \approx 1$，$\log(1) = 0$，Loss 接近 0。
*   如果 RM 错误地给 $y_l$ 更高分，则 $r_w - r_l$ 为负，$\sigma(\cdot) \approx 0$，$\log(0) \to -\infty$，Loss 变得非常大（负负得正）。

---

## 2. 训练流程与关键细节

### 2.1 数据构建
RM 的训练数据通常是**成对数据（Pairwise Data）**。
*   **来源**：对于同一个 Prompt $x$，让模型生成 $K$ 个回答（如 $K=4$ 或 $9$），然后由人类标注者进行排序。
*   **组合**：从 $K$ 个排序好的回答中，可以构建 $C_K^2$ 个成对样本 $(y_w, y_l)$。
    *   *注意*：为了防止过拟合，通常会将同一个 Prompt 下的所有 Pair 放在同一个 Batch 中计算 Loss，并除以 $C_K^2$ 进行归一化。

### 2.2 模型架构
*   **基座模型**：通常使用与 SFT 模型（Policy Model）相同架构或略小的模型（如 LLaMA, Qwen）。
*   **修改点**：
    1.  移除最后的 Softmax 层（Unembedding Layer）。
    2.  添加一个**线性层（Linear Head）**，将隐藏层状态映射为维度为 1 的标量。
    3.  **输入形式**：通常将 Prompt 和 Response 拼接：`[CLS] Prompt [SEP] Response [EOS]`，取最后一个 Token 的输出作为 Reward。

### 2.3 训练技巧
*   **初始化**：从 SFT 模型初始化 RM，而不是从预训练模型初始化。这能保证 RM 理解指令遵循的基本模式。
*   **Reward Hacking (奖励黑客)**：
    *   *现象*：RM 可能会利用某些捷径（如偏好长文本、特定词汇）来打高分，而实际上文本质量并未提升。
    *   *缓解*：在 PPO 阶段引入 KL 散度惩罚，约束 Policy Model 不要偏离 SFT 模型太远。

---

## 3. 语音奖励模型（Audio Reward Model）
在语音大模型（Speech LLM）中，RM 面临多模态挑战。

### 3.1 多维度评估体系
语音的好坏是多维度的，单一标量难以概括。通常采用**分层奖励（Hierarchical Reward）**或**多头奖励（Multi-head Reward）**。

$$ R_{total} = w_1 R_{content} + w_2 R_{style} + w_3 R_{quality} $$

1.  **内容奖励 ($R_{content}$)**:
    *   评估语音内容的准确性、逻辑性。
    *   *实现*：使用 ASR 将语音转录为文本，然后使用文本 RM 打分。
2.  **风格/韵律奖励 ($R_{style}$)**:
    *   评估情感表达、说话人相似度、语调自然度。
    *   *实现*：基于音频特征（如 Prosody Embedding）的分类器或回归模型。
3.  **声学质量奖励 ($R_{quality}$)**:
    *   评估清晰度、噪声水平。
    *   *实现*：使用无参考音频质量评估模型（如 NISQA, DNSMOS）。

### 3.2 架构设计
*   **输入**：Prompt (Text/Audio) + Response (Audio).
*   **编码器**：通常使用 Whisper Encoder 或 Audio-LLM 的 Encoder 部分。
*   **聚合策略**：
    *   **Early Fusion**: 将文本和音频特征在早期拼接。
    *   **Late Fusion**: 分别计算文本分和音频分，最后加权。

---

## 4. 奖励校准与评估

### 4.1 为什么需要校准？
RM 输出的分数通常没有绝对意义（无界），且分布可能随训练发生漂移。
*   **问题**: 如果 RM 分数方差过大，会导致 PPO 训练不稳定（梯度爆炸）。

### 4.2 校准方法
*   **Z-score Normalization**: 对每个 Batch 的 Reward 进行标准化：
    $$ r_{norm} = \frac{r - \mu}{\sigma + \epsilon} $$
    通常使用滑动平均（Running Mean/Std）来统计全局分布。
*   **Clipping**: 将 Reward 截断在一定范围内（如 $[-5, 5]$）。

### 4.3 评估指标
*   **Accuracy**: 在验证集（保留的成对数据）上，RM 判断正确的比例（即给 $y_w$ 的分高于 $y_l$ 的比例）。
    *   *基准*：随着训练进行，Accuracy 应逐渐上升，最终达到 60%-70% 甚至更高（取决于数据难度）。
*   **Agreement**: RM 与不同人类标注者的一致性。
