本页面对 Qwen3-TTS 的整体架构及各核心组件进行深入剖析，涵盖骨干网络、双 Tokenizer、解码器、Speaker Encoder 以及 MTP 模块等关键模块的设计细节。

---

## 1. 整体架构概览

> [!important]
> 
> **设计哲学**：Qwen3-TTS 采用「LM 即 TTS」的范式，将文本转语音建模为一个**条件自回归序列生成**问题。骨干网络复用 Qwen3 大语言模型家族，通过双轨道表征将文本和语音 token 统一到同一个序列空间中进行联合建模。

**核心数据流**：

1. 输入文本 → Qwen Tokenizer → 文本 token 序列

1. （可选）参考语音 → Speaker Encoder → Speaker Embedding

1. 文本 token + Speaker Embedding → Qwen3 LM 骨干 → 预测语音 token

1. 语音 token → Code2Wav 模块（Tokenizer 对应的解码器）→ 音频波形

整个过程是**流式**的：LM 每生成一个（或一组）语音 token，解码器就可以立即将其转换为音频输出。

---

## 2. 骨干网络：Qwen3 LM

### 2.1 基础架构

- **模型家族**：基于 Qwen3 系列大语言模型

- **参数规模**：提供 **0.6B** 和 **1.7B** 两种规格

- **基础结构**：标准 Transformer Decoder-only 架构
    
    - Multi-Head Self-Attention
    
    - RoPE（Rotary Position Embedding）位置编码
    
    - SwiGLU 激活函数的 Feed-Forward Network
    
    - RMSNorm 归一化
    

### 2.2 双轨道表征（Dual-track Representation）

这是 Qwen3-TTS 最关键的架构创新之一，将文本和语音在**通道维度**上融合：

> [!important]
> 
> **核心机制**
> 
> - 文本 token 和语音 token 不是简单地拼接成更长的序列（会导致序列过长、注意力计算量剧增）
> 
> - 而是在同一个时间步上，沿**通道轴（channel axis）**拼接
> 
> - 即每个时间步的输入向量 = [文本 embedding | 语音 embedding]
> 
> - 这样文本和语音在序列长度上保持**一一对应**

**工作流程**：

1. 接收到一个文本 token 后，将其 embedding 与前一步预测的语音 token embedding 沿通道轴拼接

1. 拼接后的向量送入 Transformer 层进行自回归计算

1. 模型**立即预测**当前时间步对应的语音 token

1. 预测的语音 token 被送入解码器生成音频片段

**优势**：

- 避免了交织（interleaving）方式导致的序列膨胀问题

- 保持了文本与语音之间的严格时间对齐

- 支持实时流式输出

---

## 3. 语音 Tokenizer 详细架构

Qwen3-TTS 提出了两种互补的 Tokenizer，分别优化不同的使用场景。

### 3.1 Qwen-TTS-Tokenizer-25Hz

#### 3.1.1 编码器架构

**基座**：Qwen2-Audio（一个多模态音频理解模型）

[[DL/TTS/Qwen3 TTS/Qwen3-TTS 模型组件架构细节/Qwen2-Audio 多模态音频理解模型详解|Qwen2-Audio 多模态音频理解模型详解]]

**两阶段训练策略**：

**Stage 1 — 语义对齐**：

- 在 Qwen2-Audio 的中间层插入两个新模块：
    
    - **Resampling Layer**：将音频特征的时间分辨率降采样到 25Hz（即每秒 25 帧，每帧对应 40ms 音频）
    
    - **Vector Quantization（VQ）层**：将连续特征离散化为码本索引
    

- 码本大小：**32,768**（$2^{15}$）

- 训练目标：在 ASR 任务上继续预训练，确保量化后的 token 保留丰富的**语义信息**

- 此阶段的 token 主要编码"说了什么"（语义内容），对"怎么说的"（声学细节）编码较弱

**Stage 2 — 声学注入**：

- 在 VQ 层之后添加基于卷积的 **Mel 频谱图解码器**

- 训练目标：从量化后的 token 重建 Mel 频谱图

- 这迫使 VQ 层在保持语义的同时，也编码足够的**声学细节**（音高、音色、韵律等）

- 这是**语义-声学融合**的关键步骤

> [!important]
> 
> **Trade-off 分析**：S2 阶段注入声学信息后，ASR WER 会轻微上升（纯语义可辨性略降），但换来的是下游 TTS 合成质量的显著提升。这是有意为之的设计权衡。

#### 3.1.2 解码器架构（Streaming Detokenizer）

25Hz Tokenizer 的解码管线由两个阶段组成：

**阶段 A — Chunk-wise Diffusion Transformer (DiT)**：

[[DL/TTS/Qwen3 TTS/Qwen3-TTS 模型组件架构细节/DiT（Diffusion Transformer）在语音合成中的应用详解|DiT（Diffusion Transformer）在语音合成中的应用详解]]

- 功能：将离散语音 token 序列映射为连续 Mel 频谱图

- 使用 **Flow Matching** 生成框架（一种高效的扩散模型变体）

- 关键流式策略 — **滑动窗口块注意力**：
    
    - 将 token 序列分成固定大小的 block（chunk size = 8 tokens）
    
    - 每个 block 的 DiT 感受野限制为：
        
        - 当前 block（正在处理的 8 个 token）
        
        - 3 个历史 block（回看上下文）
        
        - 1 个未来 block（前看上下文，look-ahead）
        
    
    - 总感受野 = 5 × 8 = 40 tokens = 1.6 秒音频
    

- 每个 chunk 输出 8 × 40ms = **320ms** 的 Mel 频谱图

**阶段 B — BigVGAN 声码器**：

[[DL/TTS/Qwen3 TTS/Qwen3-TTS 模型组件架构细节/BigVGAN 通用神经声码器详解|BigVGAN 通用神经声码器详解]]

- 功能：将 Mel 频谱图转换为时域音频波形

- 基于修改版的 **BigVGAN**（一种高保真神经声码器）

- 也采用因果/流式设计

- 右上下文需求：约 130ms

**首包延迟计算**：

- LM 需生成 **16 个 token** 才能开始首个 block 的解码（因 look-ahead 需 1 个未来 block = 8 tokens，加上当前 block 8 tokens）

- 首个 chunk 生成 320ms Mel，减去 BigVGAN 的 130ms 右上下文 → 首包约含 **190ms** 可播放音频

- 实测首包延迟：**150ms**（1.7B，并发=1）

### 3.2 Qwen-TTS-Tokenizer-12Hz

#### 3.2.1 编码器架构

**设计灵感**：借鉴 **Mimi 架构**（Meta/Kyutai, Défossez et al., 2024）的语义-声学解耦策略

[[DL/TTS/Qwen3 TTS/Qwen3-TTS 模型组件架构细节/Mimi 音频编解码器架构详解|Mimi 音频编解码器架构详解]]

**核心参数**：

- 帧率：**12.5 Hz**（每帧 80ms 音频，每秒 12.5 个 token）

- 码本结构：**16 层**（1 + 15）
    
    - **第 0 层：语义码本**
        
        - 码本大小：2,048
        
        - 由 **WavLM** 作为教师模型引导训练
        
        - 主要编码高层语义特征（说话内容、语言结构）
        
    
    - **第 1-15 层：声学 RVQ 码本**
        
        - 每层码本大小：2,048
        
        - 采用**残差向量量化（Residual Vector Quantization, RVQ）**
        
        - 逐层细化语义码本未能捕获的声学细节（音高细节、声道共振、环境信息等）
        
    

**RVQ 的工作原理**：

> [!important]
> 
> **残差向量量化（RVQ）逐层细化过程**
> 
> 1. 第 0 层：对原始特征 $x$ 进行量化，得到 $hat{x}_0$，残差 $r_1 = x - \hat{x}_0$
> 
> 1. 第 1 层：对残差 $r_1$ 进行量化，得到 $hat{r}_1$，新残差 $r_2 = r_1 - \hat{r}_1$
> 
> 1. 第 $k$ 层：对 $r_k$ 进行量化……
> 
> 1. 最终重建：$hat{x} = hat{x}0 + hat{r}_1 + hat{r}_2 + ... + hat{r}{15}$
> 
> 每一层都在纠正上一层的量化误差，层数越深，还原越精细。

**训练框架 — GAN-based**：

- **Generator（生成器）**：
    
    - 编码器：直接在原始波形上操作，提取时域特征
    
    - 量化器：语义 VQ（WavLM 监督）+ 15 层 RVQ
    
    - 解码器：从量化 token 重建波形
    

- **Discriminator（判别器）**：
    
    - 多尺度判别器，区分真实和重建语音
    
    - 推动生成器产生更自然的语音
    

- **损失函数**：
    
    - 对抗损失（GAN loss）：提升自然度
    
    - 多尺度 Mel 频谱图重建损失：强制时频一致性
    
    - 余弦相似度损失：约束编码器输出与教师模型特征对齐
    
    - WavLM 蒸馏损失：确保第 0 层码本编码语义
    

**全因果设计**（关键特性）：

- 编码器和解码器均为**纯因果**架构

- 编码器：顺序处理每帧，不需要任何未来帧信息

- 解码器：逐步从 token 重建音频，无 look-ahead

- 这使得 12Hz 变体可以实现**即时编码和解码**

#### 3.2.2 解码器架构

- **结构**：轻量级**因果卷积网络（Causal ConvNet）**

- 输入：16 层 RVQ token 的嵌入之和

- 输出：时域音频波形

- **零 look-ahead**：纯左上下文，收到 token 即可解码

- 解码延迟：仅 **4-5ms**（对比 25Hz 的 25-147ms）

---

## 4. Speaker Encoder（说话人编码器）

> [!important]
> 
> **功能**：从参考语音中提取说话人身份信息，注入 TTS 模型实现语音克隆。

**架构特点**：

- **可学习的 Speaker Encoder**，与 LM 骨干**联合训练**

- 从短至 **3 秒**的参考语音中提取 Speaker Embedding

- Embedding 被注入 Transformer 的输入序列中，作为全局条件控制说话人身份

**两种使用方式**：

1. **Speaker Embedding 克隆**：直接从参考语音提取 embedding，注入生成过程

1. **In-Context Learning 克隆**：将参考语音的文本-语音对作为上下文前缀，利用 LM 的 few-shot 能力进行更精确的韵律模仿

---

## 5. MTP 模块（Multi-Token Prediction）

[[DL/TTS/Qwen3 TTS/Qwen3-TTS 模型组件架构细节/MTP 模块（Multi-Token Prediction）详解|MTP 模块（Multi-Token Prediction）详解]]

> [!important]
> 
> **仅存在于 12Hz 变体中**，是其分层预测架构的核心组件。

### 5.1 设计动机

12Hz Tokenizer 产生 16 层 RVQ token。如果让 LM 逐层自回归预测所有 16 层，则：

- 序列长度膨胀 16 倍

- 底层声学 token（第 5-15 层）包含极其细微的声学差异，LM 的 Transformer 架构对此建模效率很低

- 延迟大幅增加

### 5.2 分层预测方案

**两阶段预测**：

**阶段 1 — LM 骨干预测语义层**：

- Qwen3 LM 骨干网络聚合输入特征（文本 + 前序语音 token）

- 通过一个线性头预测**第 0 层码本**（语义码本）的 token

- 这是最重要的一层，决定了"说什么"和"怎么说"的高层信息

**阶段 2 — MTP 模块预测残差层**：

- MTP 模块接收 LM 骨干的隐藏状态和第 0 层预测结果

- **并行**生成第 1-15 层 RVQ token

- MTP 模块的结构比 LM 骨干轻量得多（可能是几层 Transformer 或 MLP）

- 每层的预测以上一层的结果为条件

### 5.3 效率优势

- LM 骨干只需预测 1 个 token/帧（而非 16 个）

- MTP 的残差层预测非常快速（轻量网络 + 信息增量小）

- 实现了**单帧即时生成（single-frame instant generation）**

- 配合因果 ConvNet 解码器，总延迟极低

---

## 6. 两种变体的完整数据流对比

### 6.1 Qwen3-TTS-25Hz 完整流程

1. 文本 → Qwen Tokenizer → 文本 token

1. （参考语音 → Speaker Encoder → Speaker Embedding）

1. 文本 token + 前序语音 token → 沿通道轴拼接 → Qwen3 LM（0.6B/1.7B）

1. LM 线性头 → 预测当前帧的**单个**语音 token（25Hz 单码本）

1. 累积 8 个 token → 送入 Chunk-wise DiT（Flow Matching）→ 320ms Mel 频谱图

1. Mel 频谱图 → BigVGAN 声码器 → 音频波形

1. 输出首个音频包（约 190ms 可播放音频）

### 6.2 Qwen3-TTS-12Hz 完整流程

1. 文本 → Qwen Tokenizer → 文本 token

1. （参考语音 → Speaker Encoder → Speaker Embedding）

1. 文本 token + 前序语音 token → 沿通道轴拼接 → Qwen3 LM（0.6B/1.7B）

1. LM 线性头 → 预测当前帧的**第 0 层**码本 token（语义层）

1. LM 隐藏状态 + 第 0 层 token → **MTP 模块** → 并行预测第 1-15 层 RVQ token

1. 16 层 token embeddings 求和 → 因果 ConvNet 解码器 → 80ms 音频波形

1. 输出首个音频包（首包延迟 ~97ms）

---

## 7. ChatML 输入格式

所有输入统一使用 **ChatML 格式**进行标准化，支持多种控制模式：

|**控制模式**|**输入组成**|**说明**|
|---|---|---|
|基础 TTS|system prompt + 文本|标准文本转语音|
|语音克隆|system prompt + 参考音频 + 文本|3秒参考语音 → Speaker Embedding|
|Voice Design|system prompt + 自然语言描述 + 文本|如"年轻女性，温柔声音，语速中等"|
|细粒度控制|system prompt + 控制指令 + 文本|控制预定义语音的具体风格属性|

---

## 8. 架构设计总结

> [!important]
> 
> **关键设计选择及其理由**
> 
> 1. **复用 Qwen3 LM** → 继承强大的文本理解能力和多语言支持，减少从头训练成本
> 
> 1. **双轨道通道拼接** → 避免序列膨胀，保持文本-语音对齐，支持流式生成
> 
> 1. **双 Tokenizer 路线** → 25Hz 优化质量与长语音稳定性，12Hz 优化延迟与效率
> 
> 1. **MTP 分层预测** → 解决多码本预测的效率瓶颈，实现单帧即时生成
> 
> 1. **联合训练 Speaker Encoder** → 端到端优化说话人克隆质量
> 
> 1. **ChatML 统一格式** → 灵活支持多种控制模式，复用 LM 的指令跟随能力