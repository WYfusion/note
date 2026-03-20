# PPO (Proximal Policy Optimization) for LLM

PPO（近端策略优化）是目前 RLHF（Reinforcement Learning from Human Feedback）中最主流的强化学习算法。它在稳定性（Stability）和样本效率（Sample Efficiency）之间取得了极佳的平衡。

---

## 1. 为什么需要 PPO？(Background)

在 RLHF 中，我们的目标是微调 LLM，使其生成的内容获得更高的 Reward（更符合人类偏好）。
直接使用简单的策略梯度算法（如 REINFORCE 或 Vanilla Policy Gradient）存在一个严重问题：**训练极其不稳定**。

*   **步长问题**：如果一次参数更新的步长太大，策略（Policy）可能会发生剧烈变化，导致模型进入“乱码区域”（生成无意义的 Token）。一旦进入该区域，模型很难再通过探索恢复回来，导致训练崩溃。
*   **PPO 的核心思想**：**限制每次参数更新的幅度**。它通过数学手段约束新策略 $\pi_{new}$ 不会偏离旧策略 $\pi_{old}$ 太远。这就是“近端（Proximal）”的含义。

---

## 2. RLHF 中的强化学习建模 (Formulation)

在将 PPO 应用于 LLM 时，我们需要明确强化学习的要素：

*   **Agent (智能体)**: 当前正在训练的 LLM（Policy Model, $\pi_\theta$）。
*   **Environment (环境)**: 用户输入的 Prompt 以及当前已经生成的 Token 序列。
*   **State (状态)**: 当前上下文 $s_t = [Prompt, y_1, y_2, ..., y_{t-1}]$。这是 **Policy Model (Agent)** 在 $t$ 时刻“看到”的输入。
*   **Action (动作)**: 下一个生成的 Token $a_t$（从词表 $V$ 中采样）。
*   **Reward (奖励)**: 整个序列生成完毕后，由 Reward Model 给出的标量分数。

### 2.1 关键模型辨析

| 模型名称                     | 角色           | 参数状态    | 作用                                                             |
| :----------------------- | :----------- | :------ | :------------------------------------------------------------- |
| **Policy Model (Actor)** | $\pi_\theta$ | **可训练** | 当前正在学习的智能体。输入 Prompt，输出 Token 的概率分布。                           |
| **Critic Model**         | $V_\psi$     | **可训练** | 价值函数网络。输入当前状态，输出一个标量（预测得分）。通常与 Actor 共享大部分底座参数，只有最后一层 Head 不同。 |
| **Reference Model**      | $\pi_{ref}$  | **冻结**  | 初始的 SFT 模型。作为“锚点”，防止 Actor 训练跑偏（忘却语言能力）。                       |
| **Reward Model (RM)**    | $r_\phi$     | **冻结**  | 人类偏好的代理。只在句子生成结束时给出最终打分。                                       |


### 2.2 最终优化目标 (The Objective)
我们希望最大化总收益，该收益由两部分组成：
1.  **原始奖励**: Reward Model 的打分 $r_\phi(x, y)$。
2.  **KL 散度惩罚**: 约束当前策略 $\pi_\theta$ 与 SFT 后的基座模型 $\pi_{ref}$ 之间的差异。

#### 奖励

$$ R(x, y) = r_\phi(x, y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} $$

**惩罚机制：**

- 如果 $\pi_{\theta}$ 生成某个token的概率远高于 $\pi_{ref}$（即模型想标新立异，偏离了原来的语言习惯），$\frac{\pi_{\theta}}{\pi_{ref}}$ 变大，KL变大，作为负项减去，导致总奖励降低。
- 这就迫使当前的策略 $\pi_{\theta}$ 在优化奖励的同时，**必须贴近** $\pi_{ref}$ 的分布。

*   **为什么要减去 KL 散度？**
    *   **防止 Reward Hacking**：RM 只是人类偏好的代理，并不完美。如果过度优化 RM 分数，模型可能会利用 RM 的漏洞（如输出乱码但包含特定高分词）来“骗分”。
    *   **保持语言能力**：$\pi_{ref}$（SFT 模型）拥有良好的语言通顺度，约束 $\pi_\theta$ 靠近它能防止语言崩坏。
    *  **保持多样性和流利度：** 我们希望模型在学习“人类偏好”的同时，**不要忘记**它在预训练（Pre-training）和监督微调（SFT）阶段学到的语法、逻辑和世界知识。$\pi_{ref}$（Reference Model，通常是冻结的SFT模型）就是这个“锚点”。

### 2.3 避免奖励黑客 (Avoiding Reward Hacking)
常见“骗分”现象：输出语义无效、模板化或极端重复的内容，却能让 RM 打高分。缓解手段：
* **KL/对数比惩罚**：保持与 $\pi_{ref}$ 接近，限制奇异分布（已体现在目标函数中）。
* **熵奖励与温度下限**：通过熵奖励、下限温度或 top-p 下限，避免过早塌缩到单一“刷分”模式。
* **长度/重复惩罚**：对异常短/长或高重复率输出加惩罚，避免模板刷分。
* **混合或多源奖励**：将 RM 分数与规则/过滤器（安全、事实性、格式正确性）加权组合；在 RM 前增加拒答或安全过滤。
* **动态 KL 系数/早停**：监控 KL、困惑度、毒性等指标，超阈值时提高 $\beta$ 或早停；训练中期可调节 $\beta$ 以平衡探索与回归。
* **对抗与困难样本**：在 rollout 或评测中加入对抗 prompt，检查并惩罚“花式骗分”行为；定期刷新对抗集。
* **RM 校准**：对 RM 做再标定（如温度缩放、分箱校准），减少高分尾部被滥用；必要时多 RM 取平均或取最小值。
* **人工抽查与离线指标**：定期人工抽样，结合 BLEU/ROUGE/毒性/事实性等指标，防止只追 RM 分数。

---

## 3. PPO 完整训练过程概述

### Step 1: 采样 (Rollout)
1.  从 Prompt 数据集中采样一个批次 $x$。
2.  使用当前 Policy $\pi_{\theta_{old}}$ 生成响应 $y = [y_1, ..., y_T]$。指的是**上一次 PPO 迭代结束后冻结下来的策略参数**。
3.  记录所有时刻的 $(s_t, a_t)$ 以及对数概率 $\log \pi_{\theta_{old}}(a_t|s_t)$。

### Step 2: 计算奖励 (Reward Calculation)
对于生成的完整序列 $y$：
1.  **RM Score**: $R_{RM} = r_\phi(x, y)$（通常加在最后一个 Token $y_T$ 上）。
2.  **KL Penalty**: 计算每一步的 KL 散度 $KL_t = \log \pi_{\theta_{old}}(a_t|s_t) - \log \pi_{ref}(a_t|s_t)$。
3.  **Total Reward**: $r_t = R_{RM} \cdot \mathbb{I}(t=T) - \beta \cdot KL_t$。

### Step 3: 优势估计 (Advantage Estimation)
1.  使用 Critic Model 计算所有时刻的状态价值 $V(s_t)$。
2.  计算 TD Error: $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$。
3.  反向递归计算 GAE: $\hat{A}_t = \delta_t + (\gamma \lambda) \hat{A}_{t+1}$。

### Step 4: 策略更新 (Policy Update)
对于多个 Epoch（通常 3-5 次）：
1.  计算新旧策略比率: $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$。
2.  **计算 Policy Loss (Actor 损失)**: 用于更新策略模型 $\pi_\theta$。
    $$ L^{CLIP} = \min(r_t \hat{A}_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t) $$
$\text{clip}(r_t, 1-\epsilon, 1+\epsilon)$ 部分：限制新策略要在一定的程度内更新迭代，即$r_t(\theta)$是有界的$[1-\epsilon, 1+\epsilon]$。见下文描述。
3.  **加入熵奖励 (Entropy Bonus)**: 提升策略的探索性，避免概率过度塌缩。  
    $$ L^{ENT} = - \alpha \cdot \mathbb{E}_t[ H(\pi_\theta(\cdot|s_t)) ] $$  
    其中 $\alpha$ 是熵系数（常用 0.01-0.1），$H$ 表示策略分布的熵。
4.  **计算 Value Loss (Critic 损失)**: 用于更新评价模型 $V_\psi$。
    $$ L^{VF} = (V_\psi(s_t) - V_{target})^2 $$
    其中 $V_{target} = \hat{A}_t + V_{old}(s_t)$。  
5.  **组合总损失并加权**:  
    $$ L_{\text{total}} = -\mathbb{E}_t[L^{CLIP}] + c_v \cdot L^{VF} + c_e \cdot L^{ENT} $$  
    * $c_v$ 控制价值损失权重（常用 0.5），$c_e$ 控制熵奖励权重（可与 $\alpha$ 相同或设置为 1）。  
6.  梯度下降更新 $\theta$ 和 $\psi$。
---

## 4. PPO 算法核心推导 (Derivation)

### 4.1 GAE (Generalized Advantage Estimation) 详解
GAE 是一种用于估计优势函数 $A_t$ 的技术，旨在平衡**偏差 (Bias)** 和 **方差 (Variance)**。
为了计算优势 $\hat{A}_t$，我们需要一个 **Critic Model** $V_\psi(s)$ 来估计状态价值（即从当前状态开始，预计还能获得多少 Reward）。
如何判定迭代的是否符合要求？给定一个**评价模型**，经过和**Policy Model**(Actor)的同步迭代更新，评价模型可以根据*当前token的输出***给出历史经验**展望未来可能还可以拿到多少分。

 **GAE 公式**:
$$ \hat{A}_t = \delta_t + (\gamma \lambda) \delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + \dots + (\gamma \lambda)^{T-t-1} \delta_{T-1} $$
    或者写成递归形式：
$$ \hat{A}_t = \delta_t + (\gamma \lambda) \hat{A}_{t+1} $$
    *   **$T$**: 轨迹（Trajectory）的总长度，即生成的 Token 序列的总长度。
    *   **$\lambda$**: GAE 的平滑系数（通常 0.95）。
        *   $\lambda = 0$: $\hat{A}_t = \delta_t$。偏差大（依赖 Critic 准确性），方差小。
        *   $\lambda = 1$: $\hat{A}_t = \sum \gamma^k r_{t+k} - V(s_t)$。偏差小（真实回报），方差大（随机性累积）。
**TD Error (时序差分误差)**:$$ \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) $$
*   $r_t$: 当前步获得的**即时奖励**（在 LLM 中，通常只有最后一个 Token 有 RM 打分，**中间 Token 的奖励**可能只有 KL 惩罚）。不是上面的概率比率 $r_t(\theta)$，当前只有一个动作，不是前后同一步的对比。
*   $V(s_t)$: Critic 对当前状态价值的估计。 **从当前状态开始，未来能获得的累积折扣奖励期望**。
*   $V(s_{t+1})$: Critic 模型读取“Prompt + 生成到第 $t$ 步的内容”后，通过网络前向传播输出的一个标量打分。[[Critic Model]]
*   **含义**: $\delta_t$ 表示“现实（即时奖励+未来估计）”比“预期（当前估计）”好了多少。

#### 1. 为什么即时奖励只在最后？
在标准的RLHF设定中，**Reward Model (RM)** 是针对 **“Prompt + 完整的Response”** 进行打分的。
- **语义完整性：** 人类评估一个回答好不好，必须看完整个句子或段落。你不能在模型只说了一个“The”的时候就断定这句话是好是坏。
- **因此：** 环境（RM）只能在模型生成结束符（EOS）那一刻，给出一个标量分数（Scalar Score）。

#### 2. 中间 Token 的奖励由谁提供？
虽然 RM 的打分只在最后，但为了让 PPO 算法能训练每一步，我们人为构造了每一步的奖励函数 $R_t$：
对于生成的一个长度为 $T$ 的序列，第 $t$ 步的奖励 $r_t$ 通常这样定义：
- 当 $t < T$ (中间 Token): $$r_t = - \beta \cdot \log \frac{\pi_\theta(a_t|s_t)}{\pi_{ref}(a_t|s_t)}$$
    (即：中间步只有KL惩罚，没有RM分数。如果模型这一步乱说话，偏离SFT模型，立马扣分。)
- 当 $t = T$ (最后一个 Token):$$r_T = R_{model}(Response) - \beta \cdot \log \frac{\pi_\theta(a_T|s_T)}{\pi_{ref}(a_T|s_T)}$$    (即：最后一步既包含RM给的最终大奖，也包含这一步的KL惩罚。)


### 4.2 重要性采样 (Importance Sampling)比率
为了利用旧策略 $\pi_{old}$ 采样的数据来更新新策略 $\pi_\theta$，我们需要引入概率比率 $r_t(\theta)$：
$$ r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)} $$
*   当 $r_t > 1$ 时，说明新策略比旧策略更倾向于该动作。
*   当 $r_t < 1$ 时，说明新策略比旧策略更排斥该动作。
它是用来计算PPO的 **Policy Loss（策略损失）** 的，用于衡量当前更新后的策略 $\pi_\theta$ 和上一步采样时的策略 $\pi_{old}$ 之间的步长。
- **它的作用：** 配合 `clip` 操作，防止模型一次更新步子跨得太大导致训练崩塌。
- **它不是奖励：** 它不代表模型生成的“好坏”，只代表模型概率变化的“幅度”。

### 4.3 截断目标函数 (Clipped Surrogate Objective)
这是 PPO 的灵魂。为了防止 $r_t(\theta)$ 变得过大（更新幅度失控），PPO 使用了截断操作：

$$ \mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right] $$

其中：
*   $\hat{A}_t$ 是**优势函数 (Advantage)**，表示动作 $a_t$ 比“平均情况”好多少。
*   $\epsilon$ 是超参数（通常为 0.2），限制比率在 $[0.8, 1.2]$ 之间。

**直观解释**：
*   **如果动作是好的 ($\hat{A}_t > 0$)**：我们希望增加其概率 ($r_t$ 变大)。但如果 $r_t$ 已经超过 $1+\epsilon$，就不再给予额外的奖励（梯度为 0），防止**过度自信**。
*   **如果动作是坏的 ($\hat{A}_t < 0$)**：我们希望减小其概率 ($r_t$ 变小)。但如果 $r_t$ 已经低于 $1-\epsilon$，就不再给予额外的惩罚，防止**策略崩塌**。

---

## 5. 语音生成中的 PPO (Audio PPO)

将 PPO 应用于 Audio LLM（如 AudioGen, MusicGen, VALL-E）时存在特定挑战。

### 5.1 挑战
*   **序列超长**: 音频 Token 序列（如 EnCodec）通常比文本长得多（1秒音频可能对应 50-100 个 Token），导致显存占用巨大，且 GAE 计算慢。
*   **评价困难**: 文本 RM 相对成熟，但 Audio RM 需要同时评估声学质量和内容。

### 5.2 策略与技巧
*   **Token-level RL**: 仅对关键的语义 Token 或韵律 Token 进行 PPO 更新，冻结其他部分。
*   **混合奖励**:
    $$ R = w_1 \cdot \text{ASR\_Accuracy} + w_2 \cdot \text{Emotion\_Score} + w_3 \cdot \text{Audio\_Quality} $$
*   **Per-token KL**: 在音频生成中，KL 散度通常计算在 Codebook 的 Categorical 分布上。
