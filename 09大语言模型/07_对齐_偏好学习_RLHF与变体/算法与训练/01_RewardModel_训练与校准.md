# Reward Model (RM) 训练与校准

## 1. 奖励模型基础
奖励模型（Reward Model, RM）的目标是模拟人类的偏好，为模型的生成结果打分。

### 1.1 Bradley-Terry 模型
假设人类在两个选项 $ 和 $ 之间选择 $ 的概率由两者的奖励差决定：
$$ P(y_w \succ y_l | x) = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) $$
其中 $\sigma$ 是 Sigmoid 函数， $\phi$ 是参数为 $\phi$ 的奖励模型。

### 1.2 损失函数
训练 RM 的目标是最小化负对数似然（Negative Log-Likelihood）：
$$ \mathcal{L}_{RM}(\phi) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right] $$

## 2. 语音奖励模型（Audio Reward Model）
在语音大模型中，RM 不仅要评估文本内容的质量，还要评估音频的声学质量。

### 2.1 架构设计
- **输入**：Prompt (Text/Audio) + Response (Audio).
- **编码器**：通常使用与 Policy Model 相同的 Audio-Text Encoder（如 Whisper Encoder + LLM），但在末尾添加一个标量输出头（Scalar Head）。

### 2.2 训练挑战与策略
- **多维度奖励聚合**：
  语音的好是多维的。可以训练多个独立的 RM，然后加权求和：
  $$ R_{total} = w_1 R_{content} + w_2 R_{style} + w_3 R_{quality} $$
  - ${content}$：基于 ASR 转录文本的文本 RM。
  - ${style}$：情感分类器或风格一致性打分。
  - ${quality}$：无参考音频质量评估模型（如 NISQA）。

- **长度偏见（Length Bias）**：
  音频越长可能包含更多信息，但也可能更拖沓。需要对 RM 进行长度归一化或在 Loss 中加入正则项。

## 3. 奖励校准（Reward Calibration）
- **问题**：RM 的分数可能分布不均，或者随着训练进行出现分数膨胀（Reward Hacking）。
- **解决方案**：
  - 对分数进行 Z-score 归一化。
  - 监控 RM 在验证集上的准确率（Accuracy）。
