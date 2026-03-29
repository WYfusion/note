---
tags:
  - LLM/语言建模
aliases:
  - OOV问题
  - 长度膨胀
  - 多语言分词
created: 2025-01-01
updated: 2026-03-28
---

# OOV 与长度膨胀：多语言与代码场景

> [!abstract] 摘要
> OOV（未登录词）和长度膨胀是分词器的核心挑战。本节深入分析 OOV 问题的成因与解决方案、Token 长度膨胀的影响、多语言和代码场景下的特殊挑战，以及相应的优化策略。

## 0. 统一概念：什么是 OOV 和长度膨胀？

### OOV (Out-of-Vocabulary)

> [!important] OOV 定义 #关键定义
> **OOV（未登录词）**：指词表中不存在的词，模型无法直接识别需要特殊处理。

**OOV 的典型表现**：
- 被替换为 `<unk>` 或 `[UNK]`
- 需要拆分为子词处理
- 可能影响模型性能

### 长度膨胀

> [!important] 长度膨胀定义 #关键定义
> **长度膨胀**：指文本转换为 Token 后，Token 数量比原始词数显著增加的现象。

**膨胀倍数示例**：
- 英文：平均膨胀 1.5-2 倍
- 中文：汉字通常 1:1，但词可能膨胀
- 日语：假名膨胀，汉字可能缩减
- 代码：严重膨胀，可达 5-10 倍

---

## 1. OOV 问题深度解析

### 1.1 OOV 产生的原因

| 原因 | 说明 | 示例 |
|------|------|------|
| **新词** | 新出现的专有名词、品牌名 | "ChatGPT", "TikTok" |
| **拼写错误** | 打字错误、 OCR 错误 | "accomodation" (应为 accommodation) |
| **专业术语** | 领域特定词汇 | "quantum entanglement", "blockchain" |
| **多语言混合** | 语码转换 | "这个很cool", "Thank you ありがとう" |
| **罕见词** | 低频但合法的词 | "serendipity", "ephemeral" |

> [!note] OOV 的频率分布
> 长尾理论：
> - 80% 的常见词在词表中
> - 15% 的词可以通过子词处理
> - 5% 的真正的 OOV 需要特殊处理

### 1.2 OOV 的处理策略

#### 1. 子词分词

> [!important] 子词分词
> 使用 BPE/WordPiece/Unigram 将 OOV 拆分为已知子词。

**示例**：
```
"unhappiness" → ["un", "##happi", "##ness"]  # WordPiece
"quantum" → ["quant", "##um"]               # BPE
```

> [!tip] 子词分词的局限性
> - **语义断裂**：子词可能失去完整语义
> - **边界模糊**：不同分词方式可能产生不同结果
> - **长词膨胀**：长专业词可能需要多个子词

#### 2. 未知词替换

> [!warning] 未知词替换策略
> 直接使用 `<unk>`，但会影响模型表现。

**OOV 率计算**：
$$\text{OOV Rate} = \frac{\text{Number of OOV tokens}}{\text{Total tokens}} \times 100\%$$

**目标 OOV 率**：
- 训练集：< 1%
- 验证集：< 3%
- 测试集：< 5%

#### 3. 字符级处理

> [!important] 字符级 Tokenization
> 使用字符作为最小单位，确保没有 OOV。

**优点**：
- 保证无 OOV
- 保留字形信息
- 适合低资源语言

**缺点**：
- 序列过长
- 语义信息弱
- 需要更多上下文

---

## 2. 长度膨胀问题

### 2.1 膨胀倍数分析

| 语言/场景 | 平均膨胀倍数 | 示例 |
|-----------|-------------|------|
| **英文** | 1.5-2.0 | "Hello world" → ["Hello", "world"] (1:2) |
| **中文** | 1.0-1.5 | "你好世界" → ["你", "好", "世", "界"] (1:1) |
| **日语** | 1.5-3.0 | "こんにちは" → ["こ", "ん", "に", "ち", "は"] (1:5) |
| **代码** | 5-10 | `def hello():` → ["def", "hello", "(", ")", ":"] (1:5) |
| **化学式** | 10-20 | "C6H12O6" → ["C", "6", "H", "12", "O", "6"] (1:6) |

> [!example] 代码膨胀实例
```python
# 原始代码（10 字符）
def add(a, b):
    return a + b

# Token 化后（18 tokens）
["def", "add", "(", "a", ",", "b", ")", ":",
 "return", "a", "+", "b"]
```

### 2.2 膨胀的影响

#### 1. 上下文窗口限制

> [!important] 上下文窗口影响
> 膨胀导致有效上下文减少，影响模型性能。

**计算示例**：
```
模型最大长度：4096 tokens
系统提示：100 tokens
回复预留：500 tokens
可用上下文：4096 - 100 - 500 = 3496 tokens

如果膨胀倍数 = 2
实际可处理文本长度：3496 / 2 = 1748 词
```

#### 2. 推理延迟

> [!tip] 推理延迟与膨胀
> Token 越多，推理时间越长：
> - KV Cache 占用增加
> - Attention 计算复杂度 O(n²)
> - 内存使用增加

**延迟估算**：
```
基础延迟：10ms per token
膨胀倍数：2.0
实际延迟：20ms per token
长文本（1000 词）：20,000ms = 20秒
```

#### 3. 成本增加

> [!important] 成本影响
> API 调用通常按 token 计费，膨胀直接影响成本。

**成本计算**：
```
单价：$0.002 per 1K tokens
输入：1000 词 × 2.0 膨胀 = 2000 tokens
输出：200 词 × 1.5 膨胀 = 300 tokens
总成本：(2000 + 300) × $0.002 / 1000 = $0.0046
```

---

## 3. 多语言场景的挑战

### 3.1 语言特性差异

| 语言特性 | 分词挑战 | 解决方案 |
|----------|----------|----------|
| **无空格语言** | 中文、日文需要分词 | Jieba, MeCab, spaCy |
| **复合词** | 德语、荷兰语长复合词 | 子词分词 + 词典 |
| **多字节字符** | UTF-8 编码问题 | Unicode 支持 |
| **方向性** | 阿拉伯语、希伯来语 RTL | 特殊处理 |
| **音节文字** | 日语假名膨胀 | 混合分词策略 |

> [!note] 中文分词的特殊性
> 中文需要词级别分词：
> - 字级别：["我", "爱", "你"]
> - 词级别：["我", "爱", "你"] (相同，但语义不同)
> - 未分词：["我", "爱", "你"] (机器理解困难)

### 3.2 多语言模型的支持

#### BERT 多语言版

> [!important] mBERT (Multilingual BERT)
> 支持 104 种语言，使用共享词表。

**语言覆盖**：
- 主要欧洲语言
- 中文、日文
- 部分亚洲语言
- **不覆盖**：阿拉伯语、希伯来语 RTL

**挑战**：
- 词表大小限制 (30K-120K)
- 语言间资源共享
- 低资源语言表现差

#### XLM-R

> [!tip] XLM-R 的改进
> 使用更大词表（250K），支持 100+ 语言：
- 更好的低资源语言支持
- 跨语言迁移能力
- 多语言预训练策略

### 3.3 代码分词的特殊挑战

#### 1. 语法敏感

> [!important] 代码语法
> 代码分词需要保留语法结构。

**错误示例**：
```python
# 错误分词
"def func(x):" → ["def", "func", "x", ":"]  # 失去语法信息
# 正确分词
"def func(x):" → ["def", "func", "(", "x", ")", ":"]
```

#### 2. 标识符处理

> [!warning] 标识符膨胀
> 变量名、函数名可能严重膨胀：

```
变量名："this_is_a_very_long_variable_name"
→ ["this", "_", "is", "_", "a", "_", "very", "_", "long", "_", "variable", "_", "name"]
膨胀倍数：1:12
```

#### 3. 语言混合

> [!important] 代码中的多语言
> 现代代码常包含：
- 注释（多语言）
- 字符串（多语言）
- 错误信息（多语言）

**处理策略**：
```python
# 保持字符串完整性
text = "Hello, 世界!"  # 作为一个 token
# 或使用特殊标记
text = "<|en|>Hello, <|zh|>世界<!|zh|>!<!|en|>"
```

---

## 4. 优化策略

### 4.1 词表优化

#### 1. 动态词表

> [!tip] 动态词表策略
> 根据领域动态调整词表：

```python
# 领域特定词表扩展
domain_tokens = {
    'medical': ['diabetes', 'hypertension', 'patient'],
    'tech': ['blockchain', 'neural_network', 'api'],
    'finance': ['stock', 'portfolio', 'derivative']
}
```

#### 2. 混合分词

> [!important] 混合分词
> 结合多种分词策略的优势：

```
方案：
- 常见词：完整词
- 专业词：子词分词
- 专有名词：字符级处理
- 代码：语法敏感分词
```

### 4.2 长度管理

#### 1. 智能截断

> [!tip] 上下文感知截断
> 保留关键信息，截断次要信息：

```python
def smart_truncate(text, max_tokens, keywords=None):
    if keywords:
        # 保留关键词附近内容
        segments = []
        for keyword in keywords:
            pos = text.find(keyword)
            if pos != -1:
                start = max(0, pos - 200)
                end = min(len(text), pos + 200)
                segments.append(text[start:end])
        return ' '.join(segments)

    # 默认：保留最近 70% 内容
    return text[-int(len(text) * 0.7):]
```

#### 2. 分层编码

> [!important] 分层编码策略
> 对不同重要性的内容使用不同编码策略：

```
重要性分级：
Level 1 (高)：系统提示、关键问题
Level 2 (中)：上下文、历史对话
Level 3 (低)：辅助信息、冗余内容
```

### 4.3 多语言优化

#### 1. 语言检测

> [!tip] 前置语言检测
> 自动检测语言，选择合适的分词策略：

```python
def tokenize_multilingual(text):
    lang = detect_language(text)

    if lang == 'zh':
        return jieba.cut(text)
    elif lang == 'ja':
        return mecab.parse(text)
    else:
        return text.split()
```

#### 2. 代码混合分词

> [!important] 代码专用分词器
> 为代码设计的专用分词策略：

```
GPT-Code 策略：
- 保留关键字完整：def, class, if, else
- 标识符智能分割：camelCase → ["camel", "Case"]
- 字符串完整：'"Hello, world"' → 一个 token
- 注释处理：按语言分词
```

---

## 5. 主流模型的 OOV 和长度管理

### 5.1 ChatGPT

| 策略 | 实现 | 效果 |
|------|------|------|
| **子词分词** | Byte-level BPE | 膨胀 1.5-2.0 倍 |
| **OOV 处理** | 自动拆分 | OOV率 < 0.1% |
| **上下文** | 4096-128K tokens | 智能截断 |
| **多语言** | 100+ 语言 | 自动检测 |

### 5.2 Claude

| 策略 | 实现 | 效果 |
|------|------|------|
| **子词分词** | SentencePiece Unigram | 膨胀 1.2-1.8 倍 |
| **OOV 处理** | 概率化处理 | 更平滑 |
| **上下文** | 100K-200K tokens | 长文本优化 |
| **多语言** | 偏好英文 | 中文较弱 |

### 5.3 Llama 3

| 策略 | 实现 | 效果 |
|------|------|------|
| **子词分词** | SentencePiece BPE | 膨胀 1.3-1.9 倍 |
| **OOV 处理** | 优化词表 | 低资源语言好 |
| **上下文** | 8K-128K | 渐进式扩展 |
| **多语言** | 8 种语言 | 代码强，语言中 |

---

## 6. 实战建议

### 6.1 选择分词策略

> [!important] 分词策略选择
根据场景选择合适的分词器：

```python
# 场景适配
strategies = {
    'chat': 'SentencePiece Unigram',  # 平衡膨胀和语义
    'code': 'Code-specific',           # 保留语法结构
    'multilingual': 'XLM-R',          # 多语言支持
    'medical': 'Domain-specific'      # 领域优化
}
```

### 6.2 监控与优化

> [!tip] 关键指标监控
建立监控体系：

```python
metrics = {
    'oov_rate': track_oov(text),
    'expansion_ratio': len(tokens) / len(words),
    'processing_time': measure_time(tokenize),
    'memory_usage': measure_memory(tokenizer)
}
```

### 6.3 未来趋势

> [!note] 分词技术演进
1. **语义导向**：基于语义的子词分词
2. **自适应**：动态调整分词策略
3. **多模态**：统一文本、代码、音频分词
4. **压缩优化**：更高效的编码方案

---

## 相关链接

**所属模块**：[[索引_Tokenizer与分词]]

**前置知识**：
- [[01_BPE_WordPiece_Unigram]] — 理解基础分词算法
- [[02_特殊Token_对齐与模板的影响]] — 特殊 Token 管理

**相关主题**：
- [[04_Tokenizer的训练与构建流程]] — 自定义分词器优化
- [[05_中文分词与多语言挑战]] — 中文分词专题
-[[02_自回归AR_自编码AE_序列到序列Seq2Seq对比|序列模型对比]]] — 不同模型的长度处理

**延伸阅读**：
- [[../04_Transformer核心结构/模型家族/索引_模型家族|模型家族]] — 了解各模型的分词特性
-[[03_音频Token_AudioCodec与Whisper|音频 Token 化]]] — 音频分词对比
