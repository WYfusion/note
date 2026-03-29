# DPO 及其变体 (IPO, KTO, ORPO, SimPO)

## 1. DPO (Direct Preference Optimization)

### 1.1 核心思想
DPO 的核心洞见是：最优策略 $\pi^*$ 和最优奖励函数 $r^*$ 之间存在解析映射关系。因此，我们可以直接优化策略模型来满足偏好，而无需显式训练奖励模型。

### 1.2 数学推导
在 RLHF 目标下，最优策略的解析解为：
$$ \pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) e^{\frac{1}{\beta}r(x,y)} $$
反解出奖励函数 $r(x,y)$ 并代入 Bradley-Terry 模型，得到 DPO 损失函数：
$$ \mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right] $$

### 1.3 优势
- **稳定性**：无需训练 RM，避免了 RM 训练不准带来的误差传播。
- **资源**：省去了加载 RM 和 Critic 模型的显存开销。

## 2. DPO 的变体

### 2.1 IPO (Identity Preference Optimization)
- **问题**：DPO 容易过拟合，导致 KL 散度剧烈漂移。
- **改进**：在 Loss 中引入正则化项，直接约束 $\pi$ 与 $\pi_{ref}$ 的距离。

### 2.2 KTO (Kahneman-Tversky Optimization)
- **特点**：不需要成对数据 $(y_w, y_l)$，只需要二元反馈（Good/Bad）。
- **价值**：在实际场景中，收集成对数据比收集点赞/点踩更难。KTO 利用前景理论（Prospect Theory）分别处理正负样本。

### 2.3 ORPO (Odds Ratio Preference Optimization)
- **特点**：无需 Reference Model。
- **方法**：在 SFT 过程中加入一个 Odds Ratio 惩罚项，使得生成 $y_w$ 的概率比率显著高于 $y_l$。

### 2.4 SimPO (Simple Preference Optimization)
- **特点**：无需 Reference Model，使用长度归一化的 Log-Prob 作为隐式 Reward。
- **公式**：
  $$ \mathcal{L}_{SimPO} = - \log \sigma \left( \frac{\beta}{L_w} \log \pi(y_w|x) - \frac{\beta}{L_l} \log \pi(y_l|x) - \gamma \right) $$
  其中 $\gamma$ 是目标间隔（Target Margin）。

## 3. 语音大模型中的 DPO (Audio DPO)

### 3.1 应用场景
- **TTS 优化**：给定同一文本，生成两条语音，人工挑选情感更到位的一条作为 $y_w$。
- **语音翻译**：选择翻译更准确且口音更纯正的语音。

### 3.2 挑战与解决方案
- **序列长度问题**：
  音频 Token 序列通常很长（例如 10秒音频对应 750 个 Token）。计算 $\log \pi(y|x)$ 需要对整个序列求和，显存消耗大。
  - **解决方案**：使用高压缩率的 Codec（如 EnCodec, DAC）减少序列长度。

- **Reference Model 开销**：
  DPO 需要同时前向传播 Policy 和 Reference 模型。对于 Audio LLM，这会使得显存占用翻倍。
  - **解决方案**：使用 **LoRA-DPO**，共享大部分权重，仅微调 LoRA 适配器；或使用 **ORPO/SimPO** 等无 Ref 方法。

- **数据构建**：
  构造 $(Text, Audio_W, Audio_L)$ 数据集。
  - **合成数据**：使用旧版模型生成 $y_l$，新版模型生成 $y_w$。
  - **ASR 辅助**：如果 $y_l$ 的 ASR 结果与文本不符，则自动标记为 Loser。