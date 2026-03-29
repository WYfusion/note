---
tags:
  - LLM/语言建模
aliases:
  - 特殊Token
  - 对齐
  - Chat模板
created: 2025-01-01
updated: 2026-03-28
---

# 特殊 Token 与对齐

> [!abstract] 摘要
> 特殊 Token 是 LLM 通信的"系统指令"，包括序列开始/结束、掩码、填充等。本节详解特殊 Token 的分类、作用机制，以及 Chat 模板中的 Token 对齐技巧，帮助理解 LLM 如何通过特殊标记实现结构化交互。

## 0. 统一概念：什么是特殊 Token？

特殊 Token 是词表中除常规词元外，用于控制模型行为、标记序列结构或辅助训练的**特殊标记**。

> [!important] 特殊 Token 的本质
> 特殊 Token 不是"文字"，而是**模型理解的"元指令"**，就像编程语言中的关键字（if、else、for）一样，有特殊语法和语义。

### 特殊 Token 的基本分类

| 类型 | 常见 Token | 作用 | 模型示例 |
|------|------------|------|----------|
| **序列控制** | `<s>`, `</s>`, `<bos>`, `<eos>` | 标记序列边界 | BERT, GPT |
| **掩码** | `<mask>`, `<mask:0.8>` | 掩码语言建模 | BERT, RoBERTa |
| **填充** | `<pad>`, `<padding>` | 序列对齐 | 所有模型 |
| **分类** | `<cls>`, `[CLS]` | 分类任务标记 | BERT 家族 |
| **连续** | `<sep>`, `[SEP]` | 分隔不同部分 | BERT 家族 |
| **对话** | `<|im_start|>`, `<|im_end|>` | 对话结构 | ChatML, Llama |
| **系统** | `<|system|>`, `<|user|>` | 角色标识 | Chat 模板 |

---

## 1. 特殊 Token 的详细解析

### 1.1 序列控制 Token

#### BOS (Begin of Sequence)

> [!important] BOS Token #关键定义
> **BOS (序列开始标记)**：指示序列开始的特殊标记，常见于解码器生成任务。

- **作用**：为生成过程提供初始状态
- **位置**：序列的第一个位置
- **示例**：`<s>` 在法语模型中强制从法语开始生成

#### EOS (End of Sequence)

> [!important] EOS Token #关键定义
> **EOS (序列结束标记)**：指示序列结束的标记，在生成任务中尤为重要。

- **作用**：标记生成结束，控制生成长度
- **应用场景**：
  - 聊天对话的回复结束
  - 文本生成的自然终止
  - 摘要生成的长度控制

> [!warning] EOS 的重要性
> 没有 EOS，模型会持续生成，可能产生：
> - 无限循环（如重复一句话）
> - 意外终止（在中间位置截断）
> - 上下文泄漏（泄露后续输入）

---

### 1.2 掩码 Token

#### Masked Language Modeling

> [!important] Mask Token
> **掩码 Token**用于掩码语言建模（MLM），随机掩盖输入词，训练模型预测被掩盖的内容。

**BERT 训练流程**：
```
输入: "The cat sat on the mat"
掩盖: "The <mask> sat on the <mask>"
预测: "The cat sat on the mat"
```

> [!note] 掩码策略
> - **静态掩码**：15% 的词被掩盖
> - **80%**：替换为 `<mask>`
> - **10%**：替换为随机词
> - **10%**：保持原词

#### RoBERTa 的改进

RoBERTa 使用 **`<mask>`** 的改进策略：
- 更高的掩码比例（15% → 20%）
- 动态掩码（每次训练时随机）
- 预训练时不使用 `<mask>` 的未来信息

---

### 1.3 填充 Token

> [!important] Padding Token
> **Padding Token** 用于对齐不同长度的序列，确保批次数据形状一致。

#### 为什么需要 Padding？

```python
# 批次数据需要相同长度
[
    ["Hello", "world", "<pad>", "<pad>"],    # 长度 4
    ["How", "are", "you", "<pad>"],          # 长度 4
    ["Hi"]                                    # 长度 1 → 需要填充
]
```

#### Padding 策略

| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **后填充** | 保持自然阅读顺序 | 模型需要学习忽略填充 | 大多数模型 |
| **前填充** | 某些模型更有效 | 可能影响注意力计算 | GPT-2 等 |
| **双向填充** | 更好的对称性 | 实现复杂 | BERT 等 |

> [!tip] Padding Attention Mask
> 除了 attention mask，还需要 **padding mask** 来标记哪些位置是填充：
> - `1`：真实 token
> - `0`：填充 token

---

### 1.4 分类与连续 Token

#### BERT 的特殊 Token

> [!important] [CLS] 与 [SEP] #关键定义
> - **[CLS]**：Classification，用于分类任务的聚合表示
> - **[SEP]**：Separator，分隔不同句子/段落

**应用示例**：
```
[CLS] How old are you? [SEP] I am 25 years old. [SEP]
```

**聚合表示**：
- [CLS] 的输出作为整个序列的表示
- 用于情感分析、问答等任务

---

## 2. Chat 模板与 Token 对齐

### 2.1 ChatML 格式

ChatML 是 OpenAI 的对话模板格式：

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
]

# Token 化后
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
```

> [!important] ChatML 特点
> - 使用 `<|im_start|>` 和 `<|im_end|>` 标记对话边界
> - 角色明确标识（system/user/assistant）
- 支持多轮对话的完整历史

---

### 2.2 Llama 3 对话模板

Llama 3 使用更简洁的格式：

```python
# 系统提示
<s> [INST] <<SYS>>
You are a helpful assistant.
<</SYS>>
Hello, how are you? [/INST]

# 助手回复
Hello! I'm doing well, thank you for asking.

# 用户新消息
<s> [INST] What's the weather like today? [/INST]
```

> [!note] Llama 3 的改进
> - 使用 `[INST]` 和 `[/INST]` 标记用户输入
- 系统提示使用 `<<SYS>>` 和 `<</SYS>>` 包围
- 更紧凑的格式，减少 token 消耗

---

### 2.3 Token 对齐挑战与解决方案

#### 常见问题

| 问题 | 现象 | 原因 |
|------|------|------|
| **对齐失败** | 系统提示被截断 | token 限制过紧 |
| **角色混淆** | 助手回答了自己的问题 | 格式标记错误 |
| **历史丢失** | 早期对话被遗忘 | 上下文窗口限制 |

#### 解决方案

> [!important] Token 对齐策略
> **最佳实践**：确保系统提示、对话历史和用户输入的正确对齐

**1. 预留 Token 预算**
```python
# 总预算：4096 tokens
system_prompt_tokens = 100  # 预留系统提示
response_buffer = 500       # 预留回复缓冲
history_buffer = 2000       # 历史对话
user_input_buffer = 1296    # 用户输入
```

**2. 动态 Token 管理**
```python
def calculate_available_tokens():
    total = model.max_length
    used = len(tokenizer.encode(current_history))
    return total - used - response_buffer
```

**3. 智能截断策略**
- 保留最近的对话
- 保留系统提示完整
- 优先保留关键信息

---

## 3. 特殊 Token 在不同模型中的差异

### 3.1 BERT 家族

| 模型 | BOS | EOS | CLS | SEP | MASK |
|------|-----|-----|-----|-----|------|
| **BERT** | - | - | [CLS] | [SEP] | [MASK] |
| **RoBERTa** | - | - | <s> | </s> | <mask> |
| **DistilBERT** | - | - | [CLS] | [SEP] | [MASK] |

> [!note] BERT 的特殊性
> BERT 不使用 BOS/EOS，因为它是双向编码器，不需要序列边界标记。

---

### 3.2 GPT 家族

| 模型 | BOS | EOS | Prefix | Suffix | 专用 Token |
|------|-----|-----|--------|--------|-----------|
| **GPT-2** | <s> | </s> | - | - | - |
| **GPT-3** | - | - | - | - | <|endoftext|> |
| **GPT-4** | - | - | - | - | <|endoftext|> |

> [!important] GPT 的 Token 设计
> **GPT 系列**通常不使用 BOS/EOS，而是：
- 使用 `<|endoftext|>` 作为文档结束标记
- 通过生成时的停止条件控制序列结束

---

### 3.3 编码器-解码器模型

| 模型 | BOS | EOS | PAD | 特殊标记 |
|------|-----|-----|-----|----------|
| **T5** | <s> | </s> | <pad> | <extra_id_0> 等 |
| **BART** | <s> | </s> | <pad> | <mask> 等 |
| **Whisper** | - | - | <|notimestamps>| <|transcribe>| 等 |

> [!example] Whisper 的特殊标记
> Whisper 使用特殊标记控制任务类型：
> - `<|transcribe|>`：语音转文本
> - `<|translate|>`：语音翻译
> - `<|notimestamps|>`：不生成时间戳

---

## 4. 特殊 Token 的最佳实践

### 4.1 Token 预算管理

> [!important] Token 计算原则
> **不要超预算**：始终为回复预留足够的空间

**计算公式**：
```python
max_tokens = model.max_length
reserved = {
    'system': 100,      # 系统提示
    'response': 500,    # 回复长度
    'safety': 100,       # 安全边界
}
available = max_tokens - sum(reserved.values())
```

### 4.2 对话历史管理

> [!tip] 渐进式截断
> 当对话历史过长时：
> 1. 保留系统提示完整
> 2. 保留最近的 N 轮对话
> 3. 保留关键信息（如问题）

**实现示例**：
```python
def truncate_history(messages, max_tokens):
    # 保持系统消息
    system = [m for m in messages if m['role'] == 'system']

    # 按时间倒序保留对话
    conversation = [m for m in messages if m['role'] != 'system']
    conversation = conversation[-rounds_to_keep:]  # 保留最近轮次

    return system + conversation
```

### 4.3 错误处理

> [!warning] 常见错误
> **错误做法**：
> - 直接截断系统提示（导致模型行为改变）
> - 不计算特殊 Token（导致实际超出预算）
> - 忽略多字节字符（UTF-8 编码问题）

**正确做法**：
```python
# 使用 tokenizer 计算准确 token 数
encoded = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True
)
token_count = len(encoded)
```

---

## 5. 特殊 Token 的高级应用

### 5.1 动态系统提示

> [!tip] 上下文感知的系统提示
> 根据任务动态调整系统提示，优化模型表现：

```python
def get_system_prompt(task, domain=None):
    base = "You are a helpful AI assistant."

    if domain:
        base += f" You specialize in {domain}."

    if task == "creative":
        base += " Be creative and engaging."
    elif task == "technical":
        base += " Be precise and technical."

    return base
```

### 5.2 多语言支持

> [!important] 多语言特殊 Token
> 某些模型使用特殊 Token 控制语言：

| 模型 | 语言 Token | 用途 |
|------|------------|------|
| **mBERT** | [LANG:en], [LANG:zh] | 语言标识 |
| **XLM-R** | <langs:en>, <langs:zh> | 多语言混合 |
| **Whisper** | <|zh|>, <|en|> | 语音语言 |

**示例**：
```python
# 多语言提示
prompt = f"<|langs:{language}|> {text}"
```

### 5.3 控制生成

> [!example] 引导生成的特殊标记
> 使用特殊标记控制生成风格和格式：

```python
# 格式化输出
"<|json|> Return your answer in JSON format. <|json|>"
"<|table|> Create a table with these columns... <|table|>"
"<|code|> Write Python code for... <|code|>"
```

---

## 6. 主流模型的特殊 Token 总结

| 模型系列 | 系统 | 用户 | 助手 | 其他 | 特殊用途 |
|----------|------|------|------|------|----------|
| **ChatML** | `<|im_start|>system` | `<|im_start|>user` | `<|im_start|>assistant` | `<|im_end|>` | 结构化对话 |
| **Llama 3** | `<s> [INST] <<SYS>>` | `[INST]` | - | `[/INST]` | 紧凑格式 |
| **Claude** | `\n\nHuman:` | `\n\nHuman:` | `\n\nAssistant:` | - | 简单标记 |
| **Gemini** | `user:` | `user:` | `model:` | - | 简洁对话 |
| **Whisper** | - | - | - | `<|transcribe|>` | 语音任务 |

> [!tip] 模型选择建议
> **选择依据**：
> - ChatML：需要结构化对话、复杂提示
> - Llama 3：需要高效 token 使用
> - Claude/Gemini：简单对话、快速集成

---

## 相关链接

**所属模块**：[[索引_Tokenizer与分词]]

**前置知识**：
- [[01_BPE_WordPiece_Unigram]] — 理解 Token 的基本概念
-[[01_语言建模目标_MLE与交叉熵困惑度|语言建模目标]]] — 理解序列建模基础

**相关主题**：
- [[03_OOV_长度膨胀_多语言与代码]] — Token 数量管理
-[[02_自回归AR_自编码AE_序列到序列Seq2Seq对比|序列模型对比]]] — 不同架构的 Token 使用

**延伸阅读**：
- [[../04_Transformer核心结构/模型家族/索引_模型家族|模型家族]] — 了解不同模型的特殊设计
- [[04_Tokenizer的训练与构建流程]] — 自定义 Tokenizer 时添加特殊 Token
