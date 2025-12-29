# PPO (Proximal Policy Optimization，近端策略优化) for LLM

## 1. RLHF 核心流程
RLHF（Reinforcement Learning from Human Feedback）通常包含三个阶段：SFT -> RM Training -> PPO。

### 1.1 优化目标
我们希望最大化期望奖励，同时通过 KL 散度约束模型不偏离基座模型（Reference Model）太远，以防止模式崩塌（Mode Collapse）：
$$ \max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y|x)} [r_\phi(x, y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}] $$
- $\pi_\theta$：当前训练的策略模型（Policy Model）。
- $\pi_{ref}$：参考模型（通常是 SFT 后的模型，冻结参数）。
- $\beta$：KL 惩罚系数，控制探索与利用的平衡。

## 2. PPO 算法细节
### 2.1 价值函数（Value Function）
除了 Policy Model，还需要训练一个 Critic Model (x)$ 来预测当前状态的预期收益，用于计算优势函数（Advantage）。

### 2.2 优势函数（GAE）
使用广义优势估计（Generalized Advantage Estimation）：
$$ A_t = \delta_t + (\gamma \lambda) \delta_{t+1} + \dots $$

### 2.3 PPO Clip Loss
$$ \mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t [\min(r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t)] $$
其中 $(\theta)$ 是新旧策略的概率比。

## 3. 语音生成中的 PPO (Audio PPO)
将 PPO 应用于 Audio LLM（如 AudioGen, MusicGen, VALL-E）时存在特定挑战。

### 3.1 动作空间（Action Space）
- **离散 Token**：如果模型输出是离散的 Audio Codebook Indices（如 EnCodec），则与文本 LLM 的 PPO 类似。
- **连续信号**：如果模型直接输出频谱或波形（较少见），则需要连续动作空间的 PPO（Gaussian Policy）。

### 3.2 奖励设计案例
假设任务是生成一段悲伤的语音：
1.  **Prompt**：文本内容 + 悲伤情感。
2.  **Reward**：
    - 输入生成的 Audio 到情感识别模型（SER），获取悲伤类别的概率作为 Reward。
    - 输入 Audio 到 ASR 模型，计算 WER 作为负向 Reward（保证内容清晰）。

### 3.3 性能优化
- **采样效率**：音频生成通常比文本慢得多（序列更长）。
- **策略**：
  - 使用更短的音频片段进行训练。
  - **Token-level RL**：仅对关键的韵律 Token 进行强化学习，而非全序列。
