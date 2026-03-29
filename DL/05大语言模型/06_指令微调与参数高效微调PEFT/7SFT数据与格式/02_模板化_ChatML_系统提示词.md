# 模板化、ChatML 与系统提示词

为了让模型区分“指令”、“输入”和“回答”，我们需要将结构化数据转换为模型能理解的字符串序列，这就是**模板化 (Templating)**。

## 1. 常见 Prompt 模板

### 1.1 Alpaca 模板
最经典的指令微调模板。
```text
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

### 1.2 Vicuna 模板
用于多轮对话。
```text
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

USER: {instruction}
ASSISTANT: {output}</s>
```

## 2. ChatML (Chat Markup Language)

OpenAI 提出的格式，现在被广泛采用（如 Qwen, Yi）。它使用特殊的 **Control Tokens** 来明确分隔对话轮次。

### 2.1 格式结构
```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
Hi there! How can I help you?<|im_end|>
```

### 2.2 优势
*   **鲁棒性**: 防止 Prompt Injection 攻击（用户伪造指令）。
*   **清晰**: 模型能明确知道哪部分是 System，哪部分是 User。

## 3. 语音大模型的模板化

语音模型需要处理 `<Audio>` 标签，通常将其视为特殊的 Token 或占位符。

### 3.1 Qwen-Audio 的处理
Qwen-Audio 扩展了 ChatML 格式，允许在 `user` 字段中嵌入音频。

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Audio 1: <audio>path/to/audio.wav</audio>
What is the sound?<|im_end|>
<|im_start|>assistant
This is the sound of a dog barking.<|im_end|>
```
*   **实现细节**: 在 Tokenization 阶段，`<audio>...</audio>` 会被替换为特殊的 `AUDIO` token，或者直接替换为 Speech Encoder 输出的 Embedding 序列。

### 3.2 动态系统提示词 (Dynamic System Prompts)
在语音交互中，System Prompt 可以动态调整模型的行为。
*   **静音模式**: `<|im_start|>system\nPlease answer briefly and do not generate audio.<|im_end|>`
*   **语音模式**: `<|im_start|>system\nPlease answer in a spoken style, using simple words.<|im_end|>`

## 4. 训练时的 Masking 策略

在计算 Loss 时，我们只关心模型生成的 `assistant` 部分。
*   **System & User**: Loss Mask = 0 (不学习)。
*   **Assistant**: Loss Mask = 1 (学习)。
*   **Audio Tokens**: 通常 Mask 掉，不预测音频输入本身（除非是自监督任务）。
