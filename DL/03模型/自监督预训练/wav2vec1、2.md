# 深度学习笔记：wav2vec 与 wav2vec 2.0 详解

## 1. 背景与意义：为什么要搞 "wav2vec"？

### 1.1 核心痛点：标注数据的稀缺
传统的语音识别（ASR）模型（如 Deep Speech 2）通常是全监督学习，依赖海量的**已标注音频数据**（音频+对应的文本）才能达到好的效果 。
* **问题**：收集标注数据非常昂贵且耗时。世界上有约 7,000 种语言，绝大多数没有足够的标注数据来进行训练 。
* **灵感**：人类婴儿学习语言时，是先通过“听”大人的声音来学习语音结构（无监督），而不是一开始就看着字幕学习（有监督）。

### 1.2 解决方案：自监督预训练 (Self-Supervised Pre-training)
核心思想是**先从无标注的纯音频中学习通用的语音表示**，然后再用少量的标注数据进行微调（Fine-tuning）以完成特定的识别任务。
* **wav2vec (2019)**：验证了无监督预训练可以提升监督学习的效果。
* **wav2vec 2.0 (2020)**：证明了仅需极少量数据（如10分钟）即可通过强大的预训练实现高精度识别。

---

## 2. wav2vec (1.0)：听音辨位，预测未来

### 2.1 核心思路
wav2vec 的灵感来源于 NLP 中的 Word2Vec。它的目标不是生成语音，而是通过**预测未来**来学习当前的特征表示。它使用**对比损失**（Contrastive Loss）来区分“真实的未来音频”和“负样本” 。
**任务：** 给定当前的语音上下文，模型需要从一堆选项中找出真正的“未来音频片段”，区分于其他干扰项（负样本）。

### 2.2 模型架构
#wav2vec 是一个**全卷积神经网络 (Fully Convolutional Network)**，易于在现代硬件上并行化。主要包含两个堆叠的卷积网络：

1.  **编码器网络 (Encoder Network, $f$)**：
    * **输入**：原始音频波形 $x$。
    * **结构**：5 层卷积网络。
    * **功能**：将高频的原始波形压缩为低频的潜在特征表示 $z$（每 10ms 生成一个特征向量，编码约 30ms 的音频）。
2.  **上下文网络 (Context Network, $g$)**：
    * **输入**：编码器的输出 $z$。
    * **结构**：9 层卷积网络（大模型为 12 层）。
    * **功能**：混合多个时间步的 $z$，生成这就包含上下文信息的表示 $c$。其感受野（Receptive Field）可覆盖约 210ms 到 810ms 的音频。

### 2.3 训练目标 (Objective)
模型通过解决**二分类任务**进行优化：
* 给定当前上下文 $c_i$，预测 $k$ 步之后的特征 $z_{i+k}$。
* **损失函数**：最小化对比损失。模型需要区分**真实的未来样本** $z_{i+k}$ 和从同一序列中随机抽取的**干扰样本（Distractors/Negatives）**。
$$ \mathcal{L}_k = -\sum_{i=1}^{T-k} \left( \log \sigma\!\left(\mathbf{z}_{i+k}^{\top} h_k(\mathbf{c}_i)\right) + \lambda \, \mathbb{E}_{\tilde{\mathbf{z}}\sim p_n} \left[\log \sigma\!\left(-\tilde{\mathbf{z}}^{\top} h_k(\mathbf{c}_i)\right)\right] \right) $$
    $$\mathcal{L} = \text{区分真样本的概率} + \text{区分假样本的概率}$$
## 参数与符号说明
- **步长损失：**$\mathcal{L}_k \in \mathbb{R}$ 表示给定预测步长（time step / step size）$k$ 时的对比损失。
- **总损失：**$\mathcal{L}=\sum_\limits{k=1}^{K}\mathcal{L}_k$  表示对不同步长 $k$ 的损失求和后的目标函数。
- **索引与序列长度：**$T\in\mathbb{N},\quad k\in\{1,\dots,K\},\quad i\in\{1,\dots,T-k\}$  其中 $T$为序列长度；为了使$i+k≤T$，求和上界为 $T−k$。
- **表示向量（正样本与上下文）：**$\mathbf{z}_{i+k}\in\mathbb{R}^{d_z},\qquad \mathbf{c}_i\in\mathbb{R}^{d_c}$,$z_{i+k}​$ 表示时间步$i+k$的真实样本表示（positive sample）；$c_i$表示时间步 $i$ 的上下文表示（context）。
- **步长相关的仿射映射**：$h_k:\mathbb{R}^{d_c}\to\mathbb{R}^{d_z},\qquad h_k(\mathbf{c}_i)=\mathbf{W}_k\mathbf{c}_i+\mathbf{b}_k,\qquad$ $\mathbf{W}_k \in \mathbb{R}^{d_z\times d_c},\qquad\mathbf{b}_k\in\mathbb{R}^{d_z}$文中称其为 step-specific affine transformation，对每个步长 $k$ 使用不同参数 $(W_k,b_k)$。
- **sigmoid 函数：**$\sigma:\mathbb{R}\to(0,1),\qquad\sigma(x)=\frac{1}{1+\exp(-x)}$          $\sigma\!\left(\mathbf{z}_{i+k}^{\top}h_k(\mathbf{c}_i)\right)$在文中被解释为 “$z_{i+k}$​ 为真实样本（true sample）” 的概率。
- **负样本与负样本分布：**$\tilde{\mathbf{z}}\sim p_n,\qquad \tilde{\mathbf{z}}\in\mathbb{R}^{d_z}$   $\tilde{\mathbf{z}}$ 为从负样本分布 $p_n$​ 采样得到的负例（negative / distractor）。 文中实现里采用均匀抽取 distractors 的近似：  $p_n(\mathbf{z})=\frac{1}{T}$(等价理解：在长度为 $T$ 的序列位置上均匀抽取负例来源。）
- **期望算子：** $\mathbb{E}_{\tilde{\mathbf{z}}\sim p_n}\big[\cdot\big]$表示对负样本采样的期望；文中说明实际通过采样若干个负例来近似该期望。
- **负样本项权重**：$\lambda\in\mathbb{R}_{>0}$    文中说明实践中将 $λ$ 设为负样本数量（the number of negatives）。
**通俗解释：** 对于时刻 $i$ 的上下文 $c_i$，模型需要判断 $k$ 步之后的特征 $z_{i+k}$ 是真实的样本，还是从其他地方随机抽取的“假”样本（负样本）。如果模型能以高概率选出真样本，说明它听懂了语音的连贯性。

### 2.4 效果
* 在 WSJ 数据集上，相比 Deep Speech 2（字符级最佳基线），wav2vec 在仅使用少量标注数据时，WER（词错误率）降低了 36%。

---

## 3. wav2vec2：完形填空与离散化革命

### 3.1 核心思路
#wav2vec2 是一次巨大的架构升级。它不再单纯预测未来，而是采用了类似 BERT 的**掩码机制**（Masking），并引入了**量化**（Quantization）模块。它的目标是联合学习上下文表示和离散的语音单元 。

### 3.2 模型架构详情
模型由三个关键模块组成：

1.  **特征编码器 (Feature Encoder)**：
    * **结构**：多层时域卷积（Temporal Convolution），包含 7 个卷积块。
    * **功能**：将原始波形 $X$ 转换为潜在语音表示 $Z$。这部分与 1.0 类似，负责底层特征提取。

2.  **Transformer 上下文网络 (Context Network)**：
    * **结构**：标准的 Transformer 架构（Base 版 12 层，Large 版 24 层）。
    * **功能**：输入经过**掩码**（Masking）处理的特征 $Z$，利用 Self-Attention 机制捕捉整个序列的依赖关系，输出上下文表示 $C$ 。
    * **位置编码**：使用卷积层代替固定的位置编码，以学习相对位置信息。

3.  **量化模块 (Quantization Module)** —— **核心创新**：
    * **背景**：语音是连续信号，而语言（文字）是离散的。为了让模型更好地理解语言结构，wav2vec 2.0 强制将连续特征 $Z$ 映射为有限的离散码本（Codebook）条目。
    * **实现原理**：
        * **乘积量化 (Product Quantization)**：将特征向量分成 $G$ 组，每组从 $V$ 个条目中选一个，最后拼接。例如 $G=2, V=320$，理论上可以组合出 $320 \times 320 = 102,400$ 种离散表示。
        * [[Gumbel-Softmax（Gumbel Softmax）]]：为了让这个“选择离散条目”的过程可导（可以反向传播训练），使用了 Gumbel Softmax 技巧进行微分近似。
![[Pasted image 20260105170705.png|900]]

### 3.3 预训练方法 (Pre-training)
训练是一个**自监督**过程，包含两个损失函数：

1.  **对比损失 (Contrastive Loss, $\mathcal{L}_m$)**：
    * **掩码机制**：随机“挖掉”一部分编码器输出的特征（Mask 掉约 49% 的时间步）。
    * **任务**：模型读取被 Mask 后的序列，Transformer 输出上下文 $c_t$。模型必须在 $K+1$ 个候选项中（包含 1 个真实的**量化表示** $q_t$ 和 $K$ 个干扰项），正确识别出被挖掉的那个位置原本是什么。
    * **关键点**：输入给 Transformer 的是连续特征（为了保留信息），但对比任务的目标是离散化的量化特征（为了鲁棒性和学习语言结构）。

2.  **多样性损失 (Diversity Loss, $\mathcal{L}_d$)**：
    * **目的**：防止模型“偷懒”，只使用码本中的某几个条目。
    * **方法**：鼓励模型在通过 Softmax 选择码本条目时，概率分布的熵（Entropy）最大化，即尽可能均匀地使用所有码本条目。
* **wav2vec 2.0 预训练损失函数**：由**对比损失**（masked step 的正确量化表示识别）与**码本多样性损失**（鼓励均匀使用码本条目）组成： $$ \mathcal{L}=\mathcal{L}_m+\alpha\,\mathcal{L}_d $$
1) Contrastive Loss（对比损失） 给定以被 mask 的时间步 $t$ 为中心的上下文网络输出 $\mathbf{c}_t$，模型需要在候选集合 $\tilde{\mathbf{q}}\in\mathbf{Q}_t$（大小为 $K+1$）中识别真实的量化潜表示 $\mathbf{q}_t$。候选集合包含正样本 $\mathbf{q}_t$ 和 $K$ 个干扰样本（负样本）。 $$ \mathcal{L}_m = -\log \frac{\exp\!\left(\mathrm{sim}(\mathbf{c}_t,\mathbf{q}_t)/\kappa\right)} {\sum\limits_{\tilde{\mathbf{q}}\in \mathbf{Q}_t}\exp\!\left(\mathrm{sim}(\mathbf{c}_t,\tilde{\mathbf{q}})/\kappa\right)} $$ 其中余弦相似度定义为： $$ \mathrm{sim}(\mathbf{a},\mathbf{b}) = \frac{\mathbf{a}^\top\mathbf{b}}{\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert} $$
2) Diversity Loss（多样性损失） 对比任务依赖量化码本（codebook）提供正负样本表示。为鼓励模型更均匀地使用每个码本的条目，定义多样性损失为（最大化平均 softmax 分布的熵；以“最小化负熵”的形式写入损失）： $$ \mathcal{L}_d = \frac{1}{GV}\sum_{g=1}^{G}-H(\bar{\mathbf{p}}_g) = \frac{1}{GV}\sum_{g=1}^{G}\sum_{v=1}^{V}\bar{p}_{g,v}\log \bar{p}_{g,v} $$
#### 参数与符号说明
- **总损失：**$\mathcal{L}\in\mathbb{R}$，预训练最小化目标： $$ \mathcal{L}=\mathcal{L}_m+\alpha\,\mathcal{L}_d $$ 
- **权重系数：**$\alpha\in\mathbb{R}_{\ge 0}$，用于平衡对比损失与多样性损失（文中说明为 tuned hyperparameter）。 
对比损失相关 
- **时间索引：**$t\in\{1,\dots,T\}$，表示序列中的时间步（被 mask 的位置之一）。 
- **上下文表示：**$\mathbf{c}_t\in\mathbb{R}^{d}$，上下文网络（context network）在时间步 $t$ 的输出表示。 
- **量化潜表示（正样本）：**$\mathbf{q}_t\in\mathbb{R}^{d}$，对应时间步 $t$ 的真实量化潜在语音表示（true quantized latent speech representation）。 
- **候选集合：**$\mathbf{Q}_t$，量化候选表示集合： $$ \mathbf{Q}_t=\{\mathbf{q}_t\}\cup\{\tilde{\mathbf{q}}_{t,1},\dots,\tilde{\mathbf{q}}_{t,K}\},\qquad |\mathbf{Q}_t|=K+1 $$ 其中 $\tilde{\mathbf{q}}$ 表示干扰样本（distractors / negatives）。
- **负样本数量：**$K\in\mathbb{N}$，每个被 mask 的时间步用于对比的负样本个数；因此候选总数为 $K+1$。 
- **负样本采样方式**：干扰样本从**同一条语音**的其他被 mask 时间步中**均匀采样**（文中描述）。 
- **温度参数：**$\kappa\in\mathbb{R}_{>0}$（注意：论文里常用 $\kappa$ 或 $\tau$ 表示 temperature；此处与原文一致写为 $\kappa$），用于缩放相似度 logits： $$ \exp(\mathrm{sim}(\cdot,\cdot)/\kappa) $$ $\kappa$ 越小分布越尖锐，越强调 hardest negative。 
- **相似度函数：**$\mathrm{sim}(\mathbf{a},\mathbf{b})$ 使用余弦相似度： $$ \mathrm{sim}(\mathbf{a},\mathbf{b})=\frac{\mathbf{a}^\top\mathbf{b}}{\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert} $$ 其中 $\lVert\cdot\rVert$ 为 $\ell_2$ 范数。 
- **对比损失：**$\mathcal{L}_m\in\mathbb{R}$，等价于在候选集合 $\mathbf{Q}_t$ 上做 softmax 分类并取负对数似然，使正样本 $\mathbf{q}_t$ 的概率最大。 多样性损失相关 
- **码本个数：**$G\in\mathbb{N}$，表示使用的 codebook 组数（groups）。 
- **每个码本条目数：**$V\in\mathbb{N}$，每个 codebook 中可选的离散条目（entries）的数量。 
- **平均 softmax 分布：**$\bar{\mathbf{p}}_g\in\mathbb{R}^{V}$，第 $g$ 个 codebook 在一个 batch（跨多条 utterances）上、对条目选择概率的平均分布： $$ \bar{\mathbf{p}}_g=\big(\bar{p}_{g,1},\dots,\bar{p}_{g,V}\big),\qquad \sum_{v=1}^{V}\bar{p}_{g,v}=1,\ \bar{p}_{g,v}\ge 0 $$ 文中说该 softmax 分布不包含 gumbel noise，也不包含 temperature（即用于统计“使用频率”的分布是普通 softmax 概率）。 
- **熵：**$H(\bar{\mathbf{p}}_g)$ 为离散分布熵： $$ H(\bar{\mathbf{p}}_g)=-\sum_{v=1}^{V}\bar{p}_{g,v}\log \bar{p}_{g,v} $$ 
- **多样性损失：**$\mathcal{L}_d\in\mathbb{R}$，对所有 codebook 的平均负熵（或等价的 $\sum \bar{p}\log \bar{p}$ 形式）： $$ \mathcal{L}_d=\frac{1}{GV}\sum_{g=1}^{G}\sum_{v=1}^{V}\bar{p}_{g,v}\log \bar{p}_{g,v} $$ 最小化 $\mathcal{L}_d$ 等价于最大化熵，从而鼓励 $\bar{\mathbf{p}}_g$ 更接近均匀分布（更均匀使用码本条目）。
- **通俗解释：** 
$\mathcal{L}_m$：在每个被 mask 的时间步 $t$，让模型从 $K+1$ 个量化候选里挑出真正属于该位置的 $\mathbf{q}_t$（其余为同一句话中其他位置的干扰量化表示）。 
$\mathcal{L}_d$：防止模型总是偏好少数几个码本条目，鼓励“用得更均匀”，让离散表征更有信息量。

### 3.4 效果与突破
* **极低资源**：在仅使用 **10分钟** 标注数据的情况下，Librispeech 测试集的 WER 仅为 4.8/8.2（clean/other），证明了超低资源语音识别的可行性。
* **SOTA 性能**：在使用全部 960 小时数据微调后，WER 达到 1.8/3.3，超越了当时的 Deep Speech 2 和 Noisy Student 等模型。

---

## 4. 总结：两代模型的对比

| 特性 | wav2vec (1.0) | wav2vec 2.0 |
| :--- | :--- | :--- |
| **核心范式** | 预测未来 (Predict Future) | 掩码预测 (Masked Prediction / BERT-style) |
| **上下文建模** | 卷积神经网络 (CNN) | **Transformer** (Attention 机制) |
| **特征表示** | 连续特征 | **离散化 (Quantized)** + 连续特征混合 |
| **训练目标** | 二分类对比损失 | 多分类对比损失 + 码本多样性损失 |
| **数据利用率** | 较高，但仍需一定量标注数据 | **极高**，10分钟标注数据即可用 |
| **微调方式** | 提取特征输入到下游声学模型 | 添加输出层后进行**端到端**微调 (CTC Loss) |

---

## 5. 实际应用中的缺点与局限 (基于论文与现状)

尽管 wav2vec 2.0 性能强大，但在实际落地中仍存在以下问题：

1.  **预训练算力消耗巨大**：
    * 训练 wav2vec 2.0 Large 模型需要 128 块 V100 GPU 训练数天。对于普通开发者或小公司，从头预训练极其困难，只能依赖大厂发布的开源模型。
2.  **词表与解码器的不匹配**：
    * 论文中指出，声学模型（wav2vec 2.0）使用的是字符（Character）级别的输出，而语言模型（LM）通常基于单词（Word）。这种不匹配会延迟语言模型的反馈，影响最终解码效果。后续工作多结合 Word Pieces 来优化。
3.  **对噪声和特定领域的适应性**：
    * 虽然在 Librispeech（有声书）上效果极佳，但如果应用场景（如医疗、方言、强噪环境）与预训练数据的分布差异过大，模型效果会下降，且微调需要极其谨慎地调整超参数（如 Mask 策略）。
4.  **实时性挑战**：
    * wav2vec 2.0 使用了 Transformer，全序列的 Self-Attention 计算复杂度较高，且通常是双向的（Bidirectional），这使得它在需要低延迟的流式（Streaming）语音识别场景中应用较为困难，不如传统的流式 RNN-T 或专门设计的流式 Conformer 方便。