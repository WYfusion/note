# RLHF 算法与训练

本章节涵盖了从奖励模型训练到强化学习优化的全流程，详细解析了 PPO 算法及其变体 DPO，并针对语音大模型的长序列与多模态特性提供了优化方案。

## 目录

### [01_RewardModel_训练与校准.md](./01_RewardModel_训练与校准.md)
- **Bradley-Terry 模型**：基于成对比较的概率模型。
- **Loss 函数**：最小化负对数似然。
- **语音 RM**：多维度奖励聚合（内容+风格+音质），解决长度偏见问题。

### [02_PPO_for_LLM.md](./02_PPO_for_LLM.md)
- **PPO 核心**：Actor-Critic 架构，KL 散度约束，Clip Loss。
- **Audio PPO**：
  - 动作空间：离散 Codebook vs 连续信号。
  - 奖励设计：结合 SER（情感识别）与 ASR（内容准确性）。
  - 效率优化：Token-level RL 与短片段采样。

### [03_DPO_IPO_KTO_ORPO_SimPO.md](./03_DPO_IPO_KTO_ORPO_SimPO.md)
- **DPO (Direct Preference Optimization)**：无需显式 RM，直接优化策略满足偏好。
- **变体**：
  - **IPO**：增加正则化防止过拟合。
  - **KTO**：利用二元反馈（非成对）数据。
  - **SimPO/ORPO**：无 Reference Model 的高效方法。
- **Audio DPO**：解决长序列显存问题（Codec），TTS 情感风格优化。
