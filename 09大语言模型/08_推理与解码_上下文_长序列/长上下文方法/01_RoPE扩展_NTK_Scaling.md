# RoPE 扩展：外推与长度外推技术

## 1. RoPE 回顾 (Rotary Position Embedding)
RoPE 是一种将位置信息编码进 Attention 计算的方法，被 LLaMA、Qwen 等模型广泛采用。

### 1.1 核心公式
RoPE 对 Query 和 Key 向量施加旋转变换：
$$ f_q(x_m, m) = (W_q x_m) e^{im\theta} $$
$$ f_k(x_n, n) = (W_k x_n) e^{in\theta} $$

其中 $m, n$ 是位置索引，$\theta$ 是频率参数（通常 $\theta_i = 10000^{-2i/d}$）。

**关键性质**：Query 和 Key 的内积只取决于相对位置 $m - n$：
$$ \langle f_q(x_m, m), f_k(x_n, n) \rangle = g(x_m, x_n, m - n) $$

## 2. 长度外推问题 (Length Extrapolation)
模型在训练时使用固定的上下文长度（如 2048）。当推理时输入超长序列（如 16384），模型性能急剧下降。

### 2.1 原因分析
- **位置编码超出范围**：训练时从未见过位置索引 > 2048，导致 Attention 分数计算异常。
- **困惑度飙升 (PPL Spike)**：在训练长度之外，模型预测能力崩溃。

## 3. 位置插值 (Position Interpolation, PI)
### 3.1 核心思想
将超长序列的位置索引压缩到训练范围内。

**公式**：
$$ m' = \frac{m}{s} $$
其中 $s = \frac{L_{target}}{L_{train}}$ 是缩放因子。例如，若训练长度 2048，目标长度 8192，则 $s = 4$。

### 3.2 优缺点
- **优点**：简单有效，只需少量微调即可适应新长度。
- **缺点**：压缩位置可能损失细粒度的位置信息（位置 1 和位置 2 变得难以区分）。

## 4. NTK-Aware Scaling (神经正切核感知缩放)
### 4.1 动机
Position Interpolation 对所有频率分量进行均匀压缩，但高频分量（捕捉局部位置）和低频分量（捕捉全局位置）应区别对待。

### 4.2 核心思想
- **高频分量**：保持不变（保留局部精度）。
- **低频分量**：进行插值（扩展全局范围）。

**修改 Base**：
$$ \theta'_i = \text{base}'^{-2i/d}, \quad \text{base}' = \text{base} \times \alpha^{d/(d-2)} $$
其中 $\alpha$ 是长度扩展比例。

### 4.3 变体
- **NTK-by-Parts**：对不同频段应用不同的缩放策略。
- **Dynamic NTK**：根据实际输入长度动态调整 $\alpha$。

## 5. 语音大模型中的长上下文挑战

### 5.1 序列长度远超文本
- **问题**：30秒音频 $\approx$ 2250 Token（使用 EnCodec），而训练时的上下文窗口可能只有 4096。更长的对话（如 5 分钟）将达到 ~22500 Token。
- **NTK 应用**：Audio LLM 需要更激进的长度扩展。Qwen-Audio 等模型会使用 NTK-Aware Scaling 来处理长音频。

### 5.2 流式场景的特殊性
- **流式 ASR/TTS**：实时语音处理不需要"一次性"看到全部历史，可以使用 **滑动窗口 Attention** + **Sink Token**（保留最开始的若干 Token 作为锚点）。
- **StreamingLLM**：让模型保持"流式"状态，只保留开头几个 Token 的 KV Cache 和最近的 N 个 Token，实现理论上无限长的对话。

### 5.3 Whisper 的处理方式
Whisper 采用 Encoder-Decoder 架构：
- **Encoder**：处理固定长度的音频段（30秒）。
- **长音频**：通过滑动窗口分段处理，然后拼接 Decoder 输出。
- **无需 RoPE 扩展**：因为 Encoder 始终处理固定长度，不存在外推问题。但 Decoder 的长度仍受限于训练配置。
