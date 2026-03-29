# 适配器微调 (Adapter Tuning)

Adapter Tuning 是最早期的 PEFT 方法之一（2019年），它通过在预训练模型的层之间插入轻量级的**适配器模块 (Adapter Modules)** 来实现微调。

---

## 1. 核心架构

### 1.1 插入位置
在 Transformer 的每个 Block 中，通常在以下两个位置插入 Adapter：
1.  **Self-Attention 层之后**，Add & Norm 之前。
2.  **Feed-Forward (FFN) 层之后**，Add & Norm 之前。

### 1.2 瓶颈结构 (Bottleneck Architecture)
为了保持参数效率，Adapter 采用“降维-非线性-升维”的瓶颈结构：
$$ \text{Adapter}(x) = W_{up} \cdot \sigma(W_{down} \cdot x) $$
其中：
*   $x \in \mathbb{R}^d$ 是输入特征（维度为 $d$）。
*   $W_{down} \in \mathbb{R}^{r \times d}$ 是降维矩阵，将特征压缩到低维 $r$ ($r \ll d$)。
*   $\sigma$ 是非线性激活函数（如 ReLU, GELU）。
*   $W_{up} \in \mathbb{R}^{d \times r}$ 是升维矩阵，将特征恢复到原始维度 $d$。
*   **残差连接**: 最终输出通常包含残差连接：$Output = x + \text{Adapter}(x)$。

---

## 2. 优势与劣势

### 2.1 优势
*   **参数效率**: 仅需训练 $W_{down}$ 和 $W_{up}$，参数量通常为原模型的 0.5% - 5%。
*   **模块化**: 可以为不同的任务训练不同的 Adapter，推理时根据任务动态切换，无需重新加载整个大模型。

### 2.2 劣势 (相比 LoRA)
*   **推理延迟 (Inference Latency)**: Adapter 增加了额外的串行计算层（虽然参数少，但增加了网络深度），导致推理速度变慢。LoRA 可以合并权重，无此问题。

---

## 3. 变体：AdapterFusion

### 3.1 概念
当针对多个任务训练了多个 Adapter 后，如何利用它们来解决新任务？
**AdapterFusion** 提出了一种两阶段学习策略：
1.  **Knowledge Extraction**: 为每个源任务训练一个 Adapter。
2.  **Knowledge Composition**: 固定 Adapter，训练一个 **Fusion Layer**（通常是 Attention 机制），根据输入动态组合不同 Adapter 的输出。

### 3.2 语音领域的应用
*   **多语种 ASR**: 为每种语言（英语、中文、日语）训练一个 Adapter。
*   **Code-Switching**: 在处理混合语言语音时，Fusion Layer 可以根据当前的语音特征，动态决定使用哪个语言的 Adapter，从而提高识别准确率。

---

## 4. LLaMA-Adapter

针对 LLaMA 等大模型的改进版 Adapter。
*   **Zero-init Attention**: 在训练初期，将 Adapter 的输出通过门控机制置零，确保微调开始时模型行为与预训练模型完全一致，保持训练稳定性。
*   **应用**: 在多模态模型（如 ImageBind-LLM）中，用于将视觉/音频特征注入到 LLM 中。
