# 解码策略：Greedy Search 与 Beam Search

## 1. 贪婪搜索 (Greedy Search)

### 1.1 原理
在每一步生成时，直接选择概率最大的 Token 作为输出：
$$ x_{t+1} = \arg\max_{w} P(w | x_{1:t}) $$

### 1.2 优缺点
- **优点**：计算速度快，实现简单。
- **缺点**：容易陷入局部最优（Local Optima），生成的句子可能缺乏多样性，容易出现重复。

## 2. 集束搜索 (Beam Search)

### 2.1 原理
Beam Search 是一种启发式搜索算法，它在每一步保留 $k$ 个概率最高的候选序列（Hypotheses），其中 $k$ 称为 **Beam Width**。

**流程**：
1.  **t=1**：选择 Top-k 个 Token。
2.  **t=2**：对这 $k$ 个路径分别预测下一个 Token，得到 $k \times V$ 个候选路径。
3.  **剪枝**：计算所有候选路径的累积概率，保留 Top-k 个。
4.  **循环**：直到生成结束符 `<EOS>`。

### 2.2 长度惩罚 (Length Penalty)
由于概率是连乘的（对数概率相加），长序列的累积概率天然更低。为了避免模型倾向于生成短句子，引入长度惩罚：
$$Score(Y) = \frac{\log P(Y|X)}{L(Y)^\alpha}$$
其中 $\alpha$ 通常取 0.6 ~ 0.7。

## 3. 语音大模型中的应用

### 3.1 ASR 中的 Beam Search
在语音识别（ASR）任务中，Beam Search 是标配。
- **原因**：语音信号存在模糊性（Homophones，如同音词），单纯的 Greedy Search 容易选错词。保留多个候选路径，结合语言模型（LM）的上下文信息，可以纠正声学模型的错误。
- **解码约束**：ASR 解码时常加入词典约束（Lexicon Constraint），确保生成的词在词表中。

### 3.2 投机采样 (Speculative Decoding) 在语音中的变体
为了加速推理，可以使用一个小模型（Draft Model）快速生成草稿，然后用大模型（Target Model）验证。
- **语音场景**：
  - 使用轻量级的 TTS 模型快速生成粗糙的声学特征。
  - 使用高质量的 Audio LLM 进行修正（Refinement），提升音质和韵律。

### 3.3 强制对齐 (Forced Alignment)
在某些语音任务（如字幕生成）中，我们已知文本内容，需要生成对应的时间戳。这时会使用受限的 Beam Search，强制解码路径必须经过给定的文本序列，从而找到最优的时间对齐路径。
