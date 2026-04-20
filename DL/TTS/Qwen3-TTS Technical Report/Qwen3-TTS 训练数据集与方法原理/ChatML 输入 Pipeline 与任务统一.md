## 前置知识

> [!important]
> 
> 阅读本页前建议先读：[[论文库/Qwen3-TTS Technical Report/Qwen3-TTS 模型组件架构细节|Qwen3-TTS 模型组件架构细节]]。对 Transformer/LLM 的 tokenizer 和输入格式有基本理解。

---

## 0. 定位

> 本页系统推导 Qwen3-TTS 如何将「基础 TTS / 语音克隆 / Voice Design / 细粒度控制」四种任务**统一**到同一份 ChatML 格式，以及文本/音频 token 在双轨道架构中的拼接规则、掩码策略和训练/推理的具体流动。

---

## 1. ChatML 基础格式

### 1.1 通用模板

```jsx
<|im_start|>system
[系统提示: 说话人描述 / 控制指令 / 语言 / 风格]
<|im_end|>
<|im_start|>user
[文本内容]
<|im_end|>
<|im_start|>assistant
<|speech_start|>[语音 token 序列]<|speech_end|>
<|im_end|>
```

### 1.2 特殊 token 表

| **特殊 token**                              | **作用**                 | `<\|im_start\|>` / `<\|im_end\|>` | ChatML 角色边界        |
| ----------------------------------------- | ---------------------- | --------------------------------- | ------------------ |
| `<\|speech_start\|>` / `<\|speech_end\|>` | 语音 token 段边界           | `<\|ref_audio\|>`                 | 参考音频占位符            |
| `<\|think_start\|>` / `<\|think_end\|>`   | Thinking Pattern 的推理边界 | `<\|voice_desc\|>`                | Voice Design 的描述字段 |

---

## 2. 四种任务的统一模板

### 2.1 任务 1：基础 TTS（预置音色）

```jsx
<|im_start|>system
You are a TTS model. Language: zh. Speaker: default_male_01.
<|im_end|>
<|im_start|>user
欢迎来到 Notion。
<|im_end|>
<|im_start|>assistant
<|speech_start|>[SPEECH TOKENS]<|speech_end|>
<|im_end|>
```

### 2.2 任务 2：零样本克隆

```jsx
<|im_start|>system
You are a TTS model. Language: zh. Clone the reference speaker's voice.
<|im_end|>
<|im_start|>user
<|ref_audio|>[REF_SPEECH_TOKENS] Reference text: 你好，我是小明。
Target: 今天天气真不错。
<|im_end|>
<|im_start|>assistant
<|speech_start|>[SPEECH TOKENS]<|speech_end|>
<|im_end|>
```

> [!important]
> 
> **关键设计**：参考音频的 token 直接放在 user 消息中，而非系统提示。这样的好处是：
> 
> 1. 模型通过 In-Context Learning 直接模仿参考语音的韵律、声调
> 
> 1. 与Text-Speech pair 的结构天然兼容（可在训练时复用）

### 2.3 任务 3：Voice Design

```jsx
<|im_start|>system
You are a TTS model. Language: en.
<|voice_desc|>A young female with a warm, slightly breathy voice and moderate pace.
<|im_end|>
<|im_start|>user
Good morning, everyone!
<|im_end|>
<|im_start|>assistant
<|think_start|>
The description suggests: female, age ≈20-30, warm tone, breathy quality,
medium speaking rate (~150 wpm). F0 likely 200-250 Hz, moderate H1-H2.
<|think_end|>
<|speech_start|>[SPEECH TOKENS]<|speech_end|>
<|im_end|>
```

> [!important]
> 
> **Thinking Pattern 的概率激活训练**：
> 
> 训练时以约 20–30% 概率让样本保留 `<|think_start|>...<|think_end|>` 段，其余时候除去。这样：
> 
> - 简单描述时模型无需恣思（省延迟）
> 
> - 复杂描述时模型自发触发恣思（需通过指令调节或采样选择）

### 2.4 任务 4：细粒度控制

```jsx
<|im_start|>system
You are a TTS model. Language: zh. Speaker: news_anchor_f01.
Style: 新闻主播. Emotion: 庄重. Speed: 稍慢. Pitch: 中等偏低.
<|im_end|>
<|im_start|>user
欢迎收听《新闻联播》。
<|im_end|>
<|im_start|>assistant
<|speech_start|>[SPEECH TOKENS]<|speech_end|>
<|im_end|>
```