# PPO 与 Critic Model：RLHF 核心算法之一的详解


## 1. 直观理解：RLHF 的“双胞胎”游戏

在深入数学公式之前，我们先通过一个通俗的例子来理解 RLHF 训练过程中的两个核心角色：**Actor (策略模型)** 和 **Critic (价值/评价模型)**。

想象它们是一对双胞胎，一起去参加一场“迷宫探险”游戏（即生成文本）：

1.  **Actor (哥哥/做题人)**：
    *   **职责**：负责实际走路（生成 Token）。
    *   **行为**：他走到一个岔路口，根据当前的直觉选了一条路。但他当时并不知道这条路最终通向宝藏还是陷阱。
    *   **困境**：如果每走一步都要等走到终点才知道好坏，反馈太慢了，学习效率极低。

2.  **Critic (弟弟/估分人)**：
    *   **职责**：负责站在旁边看，并预测“这一步走下去，未来能得多少分”。
    *   **行为**：虽然他也不确定未来，但他通过不断观察“哥哥的选择”和“最终的结果”，逐渐学会了判断局势。
    *   **作用**：当哥哥迈出一步时，弟弟会告诉他：“这一步走得好，比我预期的要好！”（正优势）或者“这一步走糟了，原本形势大好，你却选了条死路。”（负优势）。

3.  **Reward Model (上帝/老师)**：
    *   **职责**：只有在游戏彻底结束（句子生成完）时，才会出现，给出一个最终分数。
    *   **关系**：Critic 的智慧完全来自于模仿 Reward Model 的打分标准。

---

## 2. 核心组件与架构 (The Architecture)

在 RLHF 的 PPO 阶段，我们需要维护四个模型（通常是 Transformer 结构）：

| 模型名称                     | 角色           | 参数状态    | 作用                                                             |
| :----------------------- | :----------- | :------ | :------------------------------------------------------------- |
| **Policy Model (Actor)** | $\pi_\theta$ | **可训练** | 当前正在学习的智能体。输入 Prompt，输出 Token 的概率分布。                           |
| **Critic Model**         | $V_\psi$     | **可训练** | 价值函数网络。输入当前状态，输出一个标量（预测得分）。通常与 Actor 共享大部分底座参数，只有最后一层 Head 不同。 |
| **Reference Model**      | $\pi_{ref}$  | **冻结**  | 初始的 SFT 模型。作为“锚点”，防止 Actor 训练跑偏（忘却语言能力）。                       |
| **Reward Model (RM)**    | $r_\phi$     | **冻结**  | 人类偏好的代理。只在句子生成结束时给出最终打分。                                       |

---

## 3. 深入 Critic Model：价值的预言家

Critic 的核心任务是计算 **状态价值函数 (State Value Function)** $V(s)$，即：**“站在当前这个位置，展望未来，预计还能拿到多少分？”**

### 3.1 为什么需要 Critic？
直接用最终奖励来更新策略（如 REINFORCE 算法）方差极大。Critic 通过计算 **优势函数 (Advantage Function)** 来降低方差，使训练更稳定。

### 3.2 Critic 是如何变聪明的？
Critic 的训练目标是 **最小化预测误差 (MSE Loss)**。
*   **场景**：Actor 生成了“你好 坏”，RM 最终给了 -10 分。
*   **Critic 的反思**：如果在生成“坏”字之前，Critic 预测会有 +5 分，那么它生成"坏"就犯了巨大的错误（TD Error）。
*   **更新**：Critic 修改参数，记住：“下次看到‘你好’后面接‘坏’，要把价值预测得非常低。”

### 3.3 $V(s)$ 的计算逻辑
$V(s_t)$ 不是公式推导出来的，而是神经网络 **前向传播 (Inference)** 的结果：
1.  **输入**：序列 $s_t = [Prompt, y_1, ..., y_{t-1}]$。
2.  **处理**：经过 Transformer 提取特征，得到隐向量 $h_t$。
3.  **输出**：通过一个线性层映射为标量：$V(s_t) = \text{Linear}(h_t)$。

---

## 4. PPO 算法核心机制 (The Mechanics)

PPO (Proximal Policy Optimization) 的核心在于“**稳**”。它限制了每次参数更新的幅度，防止模型步幅太大学“崩”了，因为原模型已经有较好的效果了(SFT之后的)

### 4.1 奖励函数的设计 (The Reward)
我们希望模型得分高，但又不能乱说话。因此总奖励由两部分组成：
$$ R_t = r_{RM}(x, y) - \beta \cdot \text{KL}(\pi_\theta || \pi_{ref}) $$
*   **$r_{RM}$**：Reward Model 的打分（只在最后一步，即输出的最后一个token有）。
*   **KL 惩罚**：每一步都有。如果 Actor 生成的 Token 概率与 Reference Model 差异太大，就扣分。这迫使 Actor 保持原有的语言流利度。

### 4.2 优势函数与 GAE (关键难点)

**优势 (Advantage)** $\hat{A}_t$ 衡量了：**“这一步动作 $a_t$，比平均水平好了多少？”**

为了计算它，我们引入 **GAE (Generalized Advantage Estimation)**。让我们通过一个具体的例子来拆解。

#### 案例推导：生成 "你好 很高兴"
假设 Prompt="你好"，生成了 "很", "高", "兴" (T=3)。

**第一步：定义每一步的即时奖励 $r_t$**
*   $t=1$ ("很"): 中间步。$r_1 = -\beta \text{KL}_1$ (只有 KL 惩罚)。
*   $t=2$ ("高"): 中间步。$r_2 = -\beta \text{KL}_2$。
*   $t=3$ ("兴"): **最后一步**。$r_3 = \text{Score}_{RM} - \beta \text{KL}_3$ (RM 大奖在这里结算)。

**第二步：计算 TD Error ($\delta_t$)**
TD Error 代表“现实”与“预期”的差值：$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$。
*   **t=1**: Critic 看了 "你好 很"，预测 $V(s_2)$。$\delta_1 = r_1 + \gamma V(s_2) - V(s_1)$。
    *   *解读*：如果 $V(s_2)$ 很高，说明选了 "很" 之后前景一片光明，$\delta_1$ 为正，鼓励这一步。
*   **t=3**: 游戏结束，$V(s_{Final})=0$。$\delta_3 = r_3 + 0 - V(s_3)$。
    *   *解读*：这里 $r_3$ 包含了巨大的 RM 分数。这个分数会通过 GAE 往前传递。

**第三步：递归计算 GAE ($\hat{A}_t$)**
$$ \hat{A}_t = \delta_t + (\gamma \lambda) \hat{A}_{t+1} $$
*   $\hat{A}_3 = \delta_3$ (最后一步的优势就是 TD Error)。
*   $\hat{A}_2 = \delta_2 + (\gamma \lambda) \hat{A}_3$ (倒数第二步的优势 = 这一步的惊喜 + 打折后的未来优势)。
*   **结论**：虽然 RM 分数只加在最后一步，但通过 GAE 的递归，**第一步生成的 Token 也能分享到最后的荣耀**。

### 4.3 策略更新与 Clip 机制
有了优势 $\hat{A}_t$，我们就可以更新 Actor 了。但为了防止更新太猛，PPO 引入了 **Clip (截断)**。

$$ \mathcal{L}^{CLIP} = \min \left( \frac{\pi_{new}}{\pi_{old}} \hat{A}_t, \text{clip}\left(\frac{\pi_{new}}{\pi_{old}}, 1-\epsilon, 1+\epsilon\right) \hat{A}_t \right) $$

*   **比率 $r_t(\theta) = \frac{\pi_{new}}{\pi_{old}}$**：表示新策略相对于旧策略的变化幅度。
*   **Clip 逻辑**：
    *   如果动作是好的 ($\hat{A}_t > 0$)，我们想提高概率。但如果 $r_t > 1.2$（步子迈太大了），就强制截断，不再给予更多奖励。
    *   这保证了 $\pi_{new}$ 永远不会偏离 $\pi_{old}$ 太远，确保了训练的**单调提升**。

---

## 5. 完整训练流程总结 (The Pipeline)

1.  **采样 (Rollout)**：
    *   Actor 根据 Prompt 生成一批回答。
    *   记录下所有的 State, Action, LogProb。
2.  **评估 (Evaluation)**：
    *   **RM 打分**：给完整的回答打分。
    *   **Critic 估值**：计算每一步的 $V(s_t)$。
    *   **计算 KL**：对比 Actor 和 Reference Model 的概率分布。
3.  **优势计算 (Advantage)**：
    *   结合 $r_t$ 和 $V(s_t)$，利用 GAE 公式算出每一步的 $\hat{A}_t$。
4.  **优化 (Optimization)**：
    *   **Actor Loss**: PPO Clip Loss（最大化优势）。
    *   **Critic Loss**: MSE Loss（最小化价值预测误差）。
    *   反向传播更新 Actor 和 Critic 的参数。Reference 和 RM 保持不动。

---

## 6. 附录：语音生成中的特殊性
在 Audio LLM 中应用 PPO 时，由于音频 Token 序列极长（1秒~100 Token），GAE 计算和显存开销巨大。通常采用 **Token-level RL**（只优化关键 Token）或 **混合奖励**（ASR准确率 + 情感分 + 音质分）来适配。
