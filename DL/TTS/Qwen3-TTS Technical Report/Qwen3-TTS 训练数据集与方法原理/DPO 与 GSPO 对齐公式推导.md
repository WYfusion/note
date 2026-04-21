## 前置知识

> [!important]
> 
> 阅读本页前建议先读：[[DL/TTS/Qwen3-TTS Technical Report/Qwen3-TTS 训练数据集与方法原理/Qwen3-TTS 训练数据集与方法原理|Qwen3-TTS 训练数据集与方法原理]]。对强化学习基础（策略梯度、KL 正则）有基本理解。

---

## 0. 定位

> 本页系统推导 Qwen3-TTS 后训练中采用的两种对齐方法——**DPO**（Direct Preference Optimization）与 **GSPO**（Group Sampling Policy Optimization）——的数学形式化、推导过程、梯度方向以及在 TTS 场景中的特殊考量。

---

## 1. 变量定义表

|**符号**|**含义**|
|---|---|
|$y$|输出（语音 token 序列）|
|$\pi_\theta(y\|x)$|当前策略（正在训练的模型）|
|$r(x, y)$|奖励函数|
|$G$|GSPO 的组内采样数|

---

## 2. DPO 推导

### 2.1 RLHF 的约束优化问题

传统 RLHF 的目标：在 KL 附近策略的约束下最大化奖励：

$$\max_{\pi} \ \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(\cdot|x)}[r(x, y)] - \beta \cdot \mathbb{E}_x \ \mathrm{KL}[\pi(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)]$$

### 2.2 最优策略的解析解

对上式关于 $\pi$ 的泛函导数归零，得到：

$$\pi^\star(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\!\left( \frac{1}{\beta} r(x, y) \right)$$

其中 $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp(r(x,y)/\beta)$ 为归一化常数。

### 2.3 奖励的解析表示

反解出 $r$：

$$r(x, y) = \beta \log \frac{\pi^\star(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

> [!important]
> 
> **关键洞见**：奖励函数可以被「最优策略本身」解析地表示！既然这样，我们也许可以**跳过显式训练奖励模型**，直接用策略表示来优化。

### 2.4 Bradley-Terry 偏好模型

假设成对偏好服从 Bradley-Terry 模型：

$$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$

将奖励解析式代入，$\log Z(x)$ 在差值中消去：

$$P(y_w \succ y_l | x) = \sigma\!\left( \beta \log \frac{\pi^\star(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^\star(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right)$$

### 2.5 DPO 损失

对全部偏好数据负对数似然，即得 DPO 损失：

$$\boxed{ \mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\!\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right] }$$

### 2.6 梯度直觉

定义 $\hat{r}_\theta(x, y) = \beta \log \pi_\theta(y|x) / \pi_{\text{ref}}(y|x)$，记 $\Delta = \hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l)$：

$$\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \cdot \sigma(-\Delta) \cdot \left[ \nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x) \right]$$

> [!important]
> 
> **读懂梯度**：
> 
> 1. $\sigma(-\Delta)$ 是**难样本权重**——当模型已能正确区分（$\Delta \gg 0$）时，该值接近 0，梯度几乎为零（不再过拟合）。
> 
> 1. 括号内：**括大胜方概率 − 括大负方概率**——就是经典的对比学习梯度。
> 
> 1. $\pi_{\text{ref}}$ 提供了 KL 锚点，防止策略偏离太远。

### 2.7 TTS 场景的偏好对构造

![[2026-04-18 09.42.42TTS偏好对构造.excalidraw]]

> [!important]
> 
> **TTS 偏好特殊性**：评审一致性低于文本场景，因为主观感知差异大。实践中需要：
> 
> 1. **多评审多投票**：每对样本至少 3 位评审多数决
> 
> 1. **多维度排序**：自然度 / 发音准确度 / 韵律合理性分别打分
> 
> 1. **掉弃模糊对**：评审分歧大的样本不纳入训练

---

## 3. DPO 的 PyTorch 实现

```python
import torch
import torch.nn.functional as F

def dpo_loss(
    logits_w: torch.Tensor,           # (B, T, V) 当前模型在 y_w 上的 logits
    logits_l: torch.Tensor,           # (B, T, V)
    ref_logits_w: torch.Tensor,       # (B, T, V) 参考模型
    ref_logits_l: torch.Tensor,
    y_w: torch.Tensor,                # (B, T)
    y_l: torch.Tensor,
    mask_w: torch.Tensor,             # (B, T) 仅语音 token 位置为 1
    mask_l: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    def logp(logits, y, mask):
        logp_all = F.log_softmax(logits, dim=-1)
        logp_y = logp_all.gather(-1, y.unsqueeze(-1)).squeeze(-1)  # (B, T)
        return (logp_y * mask).sum(-1)                              # (B,)

    pi_w = logp(logits_w, y_w, mask_w)
    pi_l = logp(logits_l, y_l, mask_l)
    ref_w = logp(ref_logits_w, y_w, mask_w)
    ref_l = logp(ref_logits_l, y_l, mask_l)

    delta = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    loss = -F.logsigmoid(delta).mean()
    return loss
```

> [!important]
> 
> **工程关键点**：
> 
> 1. 仅对语音 token 位置计算 log 概率（文本位置 mask 掉）
> 
> 1. 参考模型全程冻结，可提前离线计算 `ref_logits` 缓存
> 
> 1. $\beta$ 典型取 0.1–0.5，太小偏离参考太远，太大学不动

---

## 4. GSPO 推导

### 4.1 动机：从 DPO 到组采样

DPO 的限制：

1. 依赖预收集的偏好对，与当前策略存在**分布偏移**

2. 人工标注成本高，难规模化

3. 奖励信号二元化（胜负），丢弃了程度信息

GSPO（与 GRPO 同源）的核心改进：

- **在线采样**：用当前策略生成组内多个候选

- **自动奖励**：用规则奖励（WER、SIM、UTMOS）代替人工偏好

- **组内归一化**：减少奖励方差

### 4.2 形式化

对每个输入 $x$，从当前策略采样 $G$ 条候选：

$$\{y^{(1)}, y^{(2)}, \dots, y^{(G)}\} \sim \pi_\theta(\cdot | x)$$

用规则奖励 $r(x, y^{(i)})$ 打分，组内归一化得到**相对优势**：

$$A^{(i)} = \frac{r(x, y^{(i)}) - \mu_r}{\sigma_r + \varepsilon}, \quad \mu_r = \frac{1}{G}\sum_i r^{(i)}, \quad \sigma_r = \sqrt{\frac{1}{G}\sum_i (r^{(i)} - \mu_r)^2}$$

### 4.3 GSPO 损失

参照 GRPO（Shao et al., 2024），采用结合重要性采样和 KL 正则的形式：

$$\mathcal{L}_{\text{GSPO}} = -\mathbb{E}_x \frac{1}{G} \sum_{i=1}^{G} \left[ \min\!\left( \rho^{(i)} A^{(i)}, \ \mathrm{clip}(\rho^{(i)}, 1-\epsilon, 1+\epsilon) A^{(i)} \right) - \beta \cdot \mathrm{KL}[\pi_\theta \| \pi_{\text{ref}}] \right]$$

其中 $\rho^{(i)} = \pi_\theta(y^{(i)}|x) / \pi_{\text{old}}(y^{(i)}|x)$ 是重要性采样比率，`clip` 是 PPO 风格的截断。

### 4.4 TTS 场景的多维度规则奖励

Qwen3-TTS 采用维度加权和：

$$r(x, y) = w_1 \cdot r_{\text{WER}}(y) + w_2 \cdot r_{\text{SIM}}(y, y_{\text{ref}}) + w_3 \cdot r_{\text{UTMOS}}(y) + w_4 \cdot r_{\text{F0}}(y)$$

|**奖励分量**|**计算**|**优化目标**|
|---|---|---|
|$r_{\text{SIM}}$|$\cos(\mathbf{e}(y), \mathbf{e}(y_{\text{ref}}))$|音色一致|
|$r_{\text{F0}}$|基频方差标准化|韵律表现|

---

## 5. DPO vs GSPO 的位置

![[2026-04-18 09.43.30DPO GSPO所处流程位置.excalidraw|200]]

### 5.1 为什么先 DPO 再 GSPO？

> [!important]
> 
> **顺序的必要性**：
> 
> 1. **DPO 先做粗颟粒对齐**：用人工偏好校准整体风格。人类偏好捕捉的是「听起来自不自然」等难以规则化的属性。
> 
> 1. **GSPO 再做细颟粒优化**：规则奖励能解决人类评审言不尽的维度（WER 、SIM）。
> 
> 1. **如果颠倒**：先 GSPO 会让模型陷入「规则奖励的局部最优」（hack WER 而犹威韵律），DPO 很难救回。

### 5.2 对比表

|**维度**|**DPO**|**GSPO**|
|---|---|---|
|奖励粒度|二元（胜负）|连续（多维分数）|
|标注成本|高（需人工）|低（规则自动）|
|适用|主观质量|可量化指标|

---

## 6. GSPO 的简化 PyTorch 实现

```python
import torch
import torch.nn.functional as F

def gspo_loss(
    model, ref_model, reward_fn,
    x: torch.Tensor,                  # (B, Lx)
    G: int = 8,                        # 每样本采样数
    beta: float = 0.01,
    clip_eps: float = 0.2,
):
    # 1. 从当前策略采样 G 条候选
    with torch.no_grad():
        # repeat x 为 G 条并采样
        x_rep = x.repeat_interleave(G, 0)
        y = model.generate(x_rep, do_sample=True)  # (B*G, Ly)
        rewards = reward_fn(x_rep, y)              # (B*G,)

    # 2. 组内归一化计算优势
    rewards = rewards.view(-1, G)                  # (B, G)
    mu = rewards.mean(-1, keepdim=True)
    sigma = rewards.std(-1, keepdim=True) + 1e-6
    A = ((rewards - mu) / sigma).view(-1)          # (B*G,)

    # 3. 计算当前与 old 策略的 log 概率比
    logp = sequence_logprob(model, x_rep, y)       # (B*G,)
    with torch.no_grad():
        logp_old = sequence_logprob(ref_model, x_rep, y)
    ratio = torch.exp(logp - logp_old)

    # 4. Clip 损失
    loss_unclip = ratio * A
    loss_clip = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * A
    loss_ppo = -torch.min(loss_unclip, loss_clip).mean()

    # 5. KL 正则
    kl = (logp - logp_old).mean()
    return loss_ppo + beta * kl
```

---

## 7. 两步联用的效果拆解

论文没有给出消蟍实验，但从类似工作（DeepSeek-Math GRPO、Qwen2.5 系列）推算：

|**阶段**|**WER 改善**|**SIM 改善**|**主观自然度**|
|---|---|---|---|
|• DPO|−7%|+2%|**+15%**|
|• Speaker SFT|−3%|**+8%**|+3%|

> [!important]
> 
> **分工明确**：DPO 管「听感」，GSPO 管「客观指标」，Speaker SFT 管「音色厨色」。

---

## 8. 实践中的陷阱

> [!important]
> 
> **陷阱 1：Reward Hacking**
> 
> GSPO 使用 WER 奖励时，模型可能学会**倒向指定** ASR 模型的偏好（而非真正的准确发音）。解决：
> 
> - 轮换多个 ASR 模型打分
> 
> - 加入人工抖动样本监测

> [!important]
> 
> **陷阱 2：韵律平平**
> 
> WER/SIM 驱动的 GSPO 会让模型趋向选择安全的、韵律平缓的输出（风险最低）。解决：
> 
> - 引入韵律多样性奖励（F0 方差、音量变化）
> 
> - DPO 阶段保留富有表现力的样本

> [!important]
> 
> **陷阱 3：DPO 过拟合**
> 
> $\beta$ 过小时策略偏离参考太远，生成多样性崩塌。解决：从 $\beta = 0.3$ 起步，观察 KL 上升趋势调整。

---

## 延伸阅读

- [[DL/TTS/Qwen3-TTS Technical Report/Qwen3-TTS 训练数据集与方法原理/Qwen3-TTS 训练数据集与方法原理|Qwen3-TTS 训练数据集与方法原理]]

---

## 参考文献

1. Rafailov et al. _Direct Preference Optimization: Your Language Model is Secretly a Reward Model_. NeurIPS 2023.

2. Shao et al. _DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models_ (GRPO). 2024.

3. Schulman et al. _Proximal Policy Optimization Algorithms_. 2017.

4. Ouyang et al. _Training Language Models to Follow Instructions with Human Feedback_. NeurIPS 2022.

5. Qwen Team. _Qwen2.5 Technical Report_ (GSPO 前身方法).