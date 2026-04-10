---
中文标题: Qwen3-TTS 技术报告：多语言可控鲁棒流式文本转语音模型
作者: Hangrui Hu, Xinfa Zhu, Ting He, Dake Guo, Bin Zhang, Xiong Wang, Zhifang Guo, Ziyue Jiang, Hongkun Hao, Zishan Guo, Xinyu Zhang, Pei Zhang, Baosong Yang, Jin Xu, Jingren Zhou, Junyang Lin (Qwen Team)
发表年份: 2026
研究领域: TTS
阅读状态: 已完成
重要程度: 高
论文链接: https://arxiv.org/abs/2601.15621
核心思路: Qwen3-TTS 是一个多语言、可控、鲁棒的流式文本转语音模型，支持3秒语音克隆和自然语言控制语音属性，具有低至97ms的首包延迟和可连续合成超过10分钟的流畅语音。模型采用双轨道自回归架构，结合两种互补的语音 Tokenizer，分别针对不同的延迟和质量需求，经过三阶段的预训练和后训练，展示了在多语言生成和长语音合成中的优越性能。
解决的问题: 1) 现有 TTS 缺乏对语音属性的细粒度自然语言控制能力；2) 超低延迟流式合成与高质量之间的矛盾；3) 多语言（10+ 语种）零样本语音克隆中的内容一致性和说话人相似度不足；4) 长语音（>10分钟）合成中的重复、遗漏和韵律断裂问题；5) 语义 Tokenizer 表达力不足 vs 纯声学 Tokenizer 低级细节过多导致 LLM 建模困难的平衡。
实现方法: 核心架构：基于 Qwen3 LM 的双轨道自回归架构，文本和声学 token 沿通道轴拼接进行实时合成。两种 Tokenizer 路线：(1) Qwen-TTS-Tokenizer-25Hz — 基于 Qwen2-Audio 的 25Hz 单码本语义-声学融合 Tokenizer，配合 Block-wise DiT + BigVGAN 流式解码；(2) Qwen-TTS-Tokenizer-12Hz — 12.5Hz 16层多码本 RVQ Tokenizer（WavLM 语义监督 + 对抗训练），配合轻量因果 ConvNet 解码，首包延迟低至 97ms。12Hz 变体引入 MTP（Multi-Token Prediction）模块分层预测残差码本。
详细笔记: 见页面正文——包含完整架构、Tokenizer 设计、三阶段预训练 + 三阶段后训练流程、效率分析、以及全部实验结果。
代码仓库: https://github.com/QwenLM/Qwen3-TTS
引用次数: 0
相关论文: CosyVoice 系列 (Du et al., 2024a/b, 2025), Seed-TTS (Anastassiou et al., 2024), MiniMax-Speech (Zhang et al., 2025a), F5-TTS (Chen et al., 2024), Moshi/Mimi (Défossez et al., 2024), SpeechTokenizer (Zhang et al., 2023), BigVGAN (Lee et al.), WavLM (Chen et al., 2022), GPT-4o-mini-tts, ElevenLabs
---
[[DL/TTS/Qwen3 TTS/Qwen3-TTS 评价指标详解|Qwen3-TTS 评价指标详解]]

[[0-TTS 评测基准全景指南]]

[[DL/TTS/Qwen3 TTS/Qwen3-TTS 训练数据集与方法原理|Qwen3-TTS 训练数据集与方法原理]]

[[DL/TTS/Qwen3 TTS/Qwen3-TTS 模型组件架构细节/Qwen3-TTS 模型组件架构细节|Qwen3-TTS 模型组件架构细节]]

## 1. 论文概述

Qwen3-TTS 是 Qwen 系列首个文本转语音（TTS）模型，是一个**多语言、可控、鲁棒、流式**的大规模 TTS 系统。

> [!important]
> 
> **核心亮点**：支持 3 秒语音克隆、自然语言描述控制语音属性、10+ 语种多语言生成、首包延迟低至 97ms、可连续合成 10+ 分钟流畅语音。

### 模型家族

|**模型名称**|**Streaming**|**多语言**|**语音克隆**|**指令跟随**|
|---|---|---|---|---|
|Qwen3-TTS-12Hz-1.7B-Base|✅|✅|✅||
|Qwen3-TTS-12Hz-1.7B-VoiceDesign|✅|✅||✅|
|Qwen3-TTS-12Hz-1.7B-CustomVoice|✅|✅||✅|
|Qwen3-TTS-12Hz-0.6B-Base|✅|✅|✅||
|Qwen3-TTS-25Hz-1.7B-Base|✅|✅|✅||
|Qwen3-TTS-25Hz-1.7B-VoiceEditing|✅|✅|✅|✅|
|Qwen3-TTS-25Hz-1.7B-CustomVoice|✅|✅||✅|
|Qwen3-TTS-25Hz-0.6B-Base|✅|✅|✅||

---

## 2. 语音 Tokenizer 设计

Qwen3-TTS 的核心创新之一是提出了**两种互补的语音 Tokenizer**，分别面向不同的延迟和质量需求。

### 2.1 Qwen-TTS-Tokenizer-25Hz（单码本语义-声学融合）

> [!important]
> 
> **定位**：语义丰富、与 Qwen-Audio 无缝集成、适合高质量长语音合成

**架构与训练**：

- 基于 **Qwen2-Audio** 构建，采用两阶段训练

- **Stage 1**：在 ASR 任务上继续预训练 Qwen2-Audio，在中间位置插入 Resampling Layer + Vector Quantization（VQ）层
    
    - 码本大小：32,768
    
    - 帧率：25 Hz（每 token 对应 40ms 音频）
    

- **Stage 2**：引入基于卷积的 Mel 频谱图解码器，通过重建 Mel 频谱图将声学信息注入 token 表征
    
    - 这一步是**语义与声学融合**的关键——纯语义 Tokenizer 表达力不足，纯声学 Tokenizer 低级细节过多，此设计在二者间取得平衡
    

**流式解码器（Streaming Detokenizer）**：

- 使用 **Diffusion Transformer (DiT)** + **Flow Matching** 将 code 序列映射为 Mel 频谱图

- 采用**滑动窗口块注意力**机制：DiT 的感受野限制为 4 个 block（当前 block + 3 block 回看 + 1 block 前看）

- Chunk size = 8 tokens → 每包 320ms 音频

- 使用修改版 **BigVGAN** 从 Mel 频谱图重建波形

- 首包需等待 LM 生成 16 个 token 后才能开始解码（因 look-ahead 需求）

- 首包实际包含约 190ms 音频（320ms mel - 130ms BigVGAN 右上下文）

**ASR 评测**（验证 token 语义保留能力）：

|**模型**|**Codebook**|**FPS**|**C.V. EN**|**C.V. CN**|**Fleurs EN**|**Fleurs CN**|
|---|---|---|---|---|---|---|
|S3 Tokenizer(FSQ)|6561|25|10.67|**7.29**|6.58|4.43|
|**Qwen-TTS-25Hz (S1)**|32768|25|**7.51**|10.73|**3.07**|**4.23**|
|Qwen-TTS-25Hz (S2)|32768|25|10.40|14.99|4.14|4.67|

> S2 阶段 ASR 性能略有下降是**符合预期的 trade-off**——注入更多声学细节提升下游 TTS 质量，同时轻微降低纯语义可辨性。

### 2.2 Qwen-TTS-Tokenizer-12Hz（多码本语义-声学解耦）

> [!important]
> 
> **定位**：极致低比特率、超低延迟流式（首包 97ms）、适合实时在线服务

**架构与训练**：

- 借鉴 **Mimi 架构**（Défossez et al., 2024）的语义-声学解耦量化策略

- 帧率：**12.5 Hz**（每 token 对应 80ms 音频，1s有12.5个token）

- 码本结构：**1 层语义码本 + 15 层声学 RVQ**（共 16 层）
    
    - 语义码本：由 **WavLM** 作为教师模型引导，对齐语义特征
    
    - 声学码本：15 层残差向量量化（RVQ），逐层细化语义码本未捕获的细节
    
    - 码本大小：2048
    

- 训练框架：**GAN-based**
    
    - Generator：直接在原始波形上操作，提取并量化语义和声学表征
    
    - Discriminator：提升重建语音的自然度和保真度
    
    - 多尺度 Mel 频谱图重建损失：强制时频一致性
    
    - 余弦相似度损失：约束编码器输出
    

- **全因果设计**：编码器和解码器均为因果架构
    
    - 编码器：顺序处理帧并以 12.5Hz 发射 token，无前看
    
    - 解码器：从 token 逐步重建音频（轻量因果 ConvNet）
    

**重建质量评测**（LibriSpeech test-clean）：

|**模型**|**NQ**|**FPS**|**PESQ_WB**|**STOI**|**UTMOS**|**SIM**|
|---|---|---|---|---|---|---|
|Mimi|16|12.5|2.88|0.94|3.87|0.87|
|FireredTTS 2|16|12.5|2.73|0.94|3.88|0.87|
|**Qwen-TTS-12Hz**|**16**|**12.5**|**3.21**|**0.96**|**4.16**|**0.95**|

> 在所有关键指标上全面 SOTA，尤其说话人相似度（SIM）从 0.87 跃升至 **0.95**。

### 2.3 两种 Tokenizer 对比总结

|**维度**|**Tokenizer-25Hz**|**Tokenizer-12Hz**|
|---|---|---|
|帧率|25 Hz|12.5 Hz|
|码本结构|单码本（32768）|16 层多码本（2048）|
|语义监督|Qwen2-Audio ASR|WavLM 教师模型|
|解码器|Block-wise DiT + BigVGAN|轻量因果 ConvNet|
|流式策略|需 look-ahead（16 tokens 后开始）|纯左上下文，立即解码|
|首包延迟（1.7B, concurrency=1）|150 ms|**101 ms**|
|长语音稳定性|**更优**（WER 1.517/1.225）|略逊（WER 2.356/2.812）|
|内容准确性（WER）|略逊|**更优**（12Hz 变体始终优于 25Hz）|

---

## 3. 模型架构（Qwen3-TTS）

### 3.1 整体架构

> [!important]
> 
> **骨干网络**：基于 Qwen3 LM 家族（0.6B / 1.7B 参数规模）

**核心设计 —— 双轨道表征（Dual-track Representation）**：

1. 文本使用标准 **Qwen Tokenizer** 编码

1. 语音使用 **Qwen-TTS-Tokenizer** 编码

1. 将文本 token 和声学 token 沿**通道轴（channel axis）**拼接

1. 接收到一个文本 token 后，模型**立即预测**对应的声学 token

1. 声学 token 由 **Code2Wav** 模块转换为波形

**说话人控制**：联合训练一个**可学习的 Speaker Encoder**，实现精确的身份控制。

### 3.2 Qwen3-TTS-25Hz 变体

- 使用 Tokenizer-25Hz 提取**单层语音 token**

- 骨干网络融合文本特征与前序语音 token，通过**线性头**预测当前语音 token

- 输出序列由 **Chunk-wise DiT** 模块进行高保真波形重建

### 3.3 Qwen3-TTS-12Hz 变体

- 使用 Tokenizer-12Hz 的 **RVQ token**

- 采用**分层预测方案（Hierarchical Prediction）**：
    
    1. 骨干网络聚合码本特征 → 预测**第 0 层码本**（语义层）
    
    1. **MTP（Multi-Token Prediction）模块** → 生成所有**残差码本**（第 1-15 层）
    

- 优势：捕获精细声学细节，显著提升音色一致性和表现力

- 延迟：单帧即时生成（single-frame instant generation）

---

## 4. 完整训练流程

### 4.1 预训练（Pre-training）—— 三阶段

所有数据使用 **ChatML 格式**标准化输入，支持可控语音生成。

#### Stage 1：通用阶段（General Stage）

- **数据**：超过 **500 万小时**多语言语音数据

- **目标**：建立多语言文本表征到语音的**单调映射**，构建 Qwen3-TTS 的基础能力

- **最大 token 长度**：8,192

#### Stage 2：高质量阶段（High-Quality Stage）

- **方法**：通过专用 pipeline 对数据质量进行分层，使用高质量数据进行 **CPT（Continual Pre-Training）**

- **目标**：缓解初始阶段噪声数据导致的**幻觉（hallucinations）**，显著提升生成语音质量

#### Stage 3：长上下文阶段（Long-Context Stage）

- **最大 token 长度**：从 8,192 扩展到 **32,768**

- **方法**：上采样训练数据中的长语音样本

- **效果**：增强模型处理**长文本和复杂输入**的能力，生成上下文恰当的语音响应

### 4.2 后训练（Post-training）—— 三阶段

#### Stage 1：DPO（Direct Preference Optimization）

- 基于**人类反馈**构建多语言语音样本的偏好对

- 使用 DPO 对齐模型输出与人类偏好

#### Stage 2：GSPO + Rule-based Rewards

- 使用**基于规则的奖励**

- 利用 **GSPO** 全面增强模型在各任务上的能力和稳定性

#### Stage 3：Speaker Fine-tuning

- 在 Base 模型上进行**轻量级说话人微调**

- 使 Qwen3-TTS 能够采用特定语音

- 进一步提升自然度、表现力和可控性

---

## 5. 功能特性

### 5.1 语音克隆（Voice Cloning）

- **方式一**：通过 **Speaker Embedding** 从参考语音克隆（实时克隆，3 秒参考音频）

- **方式二**：通过**文本-语音对**进行 **In-Context Learning**（更好地保留韵律）

### 5.2 语音设计（Voice Design）

- 继承 Qwen3 文本模型的**强大文本理解能力**

- 训练中引入**概率激活的 Thinking Pattern**，提升对复杂描述的指令跟随能力

- 可以根据自然语言描述创建全新的语音

### 5.3 细粒度控制（Fine-grained Control）

- 通过在输入序列前追加用户指令（包含细粒度控制信号）

- 控制预定义语音的风格（如广播员特质、新闻主播腔调等）

---

## 6. 流式效率分析

> [!important]
> 
> 所有延迟数据在单卡上使用内部 vLLM 引擎（V0 backend）测量，优化包括 `torch.compile` 和 CUDA Graph 加速。

|**模型**|**并发**|**LM TTFP**|**Tokenizer 解码**|**首包延迟**|**LM TPP**|**RTF**|
|---|---|---|---|---|---|---|
|12Hz-1.7B|1|97 ms|4 ms|**101 ms**|21 ms|0.313|
|12Hz-0.6B|1|93 ms|4 ms|**97 ms**|19 ms|0.288|
|25Hz-1.7B|1|125 ms|25 ms|150 ms|56 ms|0.253|
|25Hz-0.6B|1|113 ms|25 ms|138 ms|50 ms|0.234|
|12Hz-1.7B|6|328 ms|5 ms|333 ms|32 ms|0.463|
|25Hz-1.7B|6|376 ms|147 ms|523 ms|85 ms|0.725|

**关键洞察**：

- 12Hz Tokenizer 解码时间仅 4-5ms（因果 ConvNet），25Hz 需 25-147ms（DiT + BigVGAN）

- 12Hz 在高并发场景下延迟增长更缓慢，更适合在线服务

- 25Hz 的 RTF 略低（0.253 vs 0.313），但首包延迟更高

---

## 7. 实验结果详解

### 7.1 零样本语音克隆（Seed-TTS 基准）

| **模型**                  | **test-zh WER ↓** | **test-en WER ↓** |
| ----------------------- | ----------------- | ----------------- |
| CosyVoice 3             | **0.71**          | 1.45              |
| KALL-E                  | 0.96              | 1.94              |
| MiniMax-Speech          | 0.83              | 1.65              |
| **Qwen3-TTS-12Hz-1.7B** | 0.77              | **1.24**          |

**发现**：

1. 12Hz 变体**始终优于** 25Hz（更粗的时间分辨率让自回归模型更好建模长期依赖）

1. 1.7B → 0.6B scaling 带来一致性提升

1. **12Hz-1.7B 在 test-en 上达到 SOTA**（WER 1.24）

### 7.2 多语言生成（10 语种）

- 在 10 种语言中的 **6 种**达到最低 WER（中、英、意、法、韩、俄）

- 在**所有 10 种语言**上说话人相似度（SIM）均超越 MiniMax 和 ElevenLabs

- 充分展示了跨语言音色保持能力

### 7.3 跨语言语音克隆

- 在**中→韩**任务上，将错误率从 CosyVoice3 的 14.4 降至 **4.82**（降低约 66%）

- 在常用翻译对（中→英、英→中）上均超越基线

- 在所有评测方向上保持一致的低错误率

### 7.4 可控语音生成（InstructTTSEval）

**Voice Design（语音创建）**：

- **12Hz-1.7B-VD** 在开源模型中达到 SOTA

- 超越商业系统 Hume 和专用模型 VoiceSculptor

**Target Speaker（语音编辑）**：

- 显著超越 GPT-4o-mini-tts（中文 APS 提升 +28%）

- Gemini 系列仍为上界，但 Qwen3-TTS 展现出竞争力

### 7.5 长语音生成（>10 分钟）

|**模型**|**long-zh WER ↓**|**long-en WER ↓**|
|---|---|---|
|Higgs-Audio-v2 (chunk)|5.505|6.917|
|VibeVoice|22.619|1.780|
|VoxCPM|4.835|7.474|
|**Qwen3-TTS-25Hz-1.7B**|**1.517**|**1.225**|

> 长语音场景中 **25Hz 变体优于 12Hz**，语义 token 对长序列稳定性更有利。

---

## 8. 关键技术总结

> [!important]
> 
> **核心创新点**
> 
> 1. **双 Tokenizer 路线**：25Hz 语义-声学融合（质量优先）vs 12Hz 语义-声学解耦（延迟优先）
> 
> 1. **双轨道 LM 架构**：文本与声学 token 沿通道轴拼接，实现实时流式合成
> 
> 1. **MTP 模块**：分层预测多码本序列，单帧即时生成，极大降低延迟
> 
> 1. **三阶段预训练 + 三阶段后训练**：从通用能力到高质量到长上下文，再经 DPO → GSPO → Speaker SFT
> 
> 1. **概率激活 Thinking Pattern**：增强复杂指令跟随能力
> 
> 1. **500 万小时**多语言训练数据，覆盖 10+ 语种

---

## 9. 开源信息

- **许可证**：Apache 2.0

- **HuggingFace**：[Qwen3-TTS Collection](https://huggingface.co/collections/Qwen/qwen3-tts)

- **ModelScope**：[Qwen3-TTS Collection](https://modelscope.cn/collections/Qwen/Qwen3-TTS)

- **GitHub**：[QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)

- **论文**：[arXiv:2601.15621](https://arxiv.org/abs/2601.15621)