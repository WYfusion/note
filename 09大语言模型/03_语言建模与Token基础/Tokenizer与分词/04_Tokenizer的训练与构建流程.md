# Tokenizer 的训练与构建流程

Tokenizer 不是预定义的规则，而是通过在特定语料库上“训练”得到的。本章介绍从原始文本到生成词表（Vocabulary）的全流程。

## 1. 训练流程概览

训练 Tokenizer 的目标是找到一组最优的 Token 集合，使得在给定的词表大小限制下，能最有效地压缩文本。

1.  **准备语料 (Corpus Preparation)**: 收集具有代表性的文本数据（如 Wikipedia, Common Crawl）。
2.  **预分词 (Pre-tokenization)**: 基于规则将文本切分为基础单元（通常是按空格、标点切分）。
    *   *例子*: "Hello, world!" -> ["Hello", ",", "world", "!"]
3.  **模型训练 (Model Training)**: 使用 BPE / WordPiece / Unigram 算法统计频次，合并高频子词，生成词表。
4.  **后处理 (Post-processing)**: 添加特殊 Token（如 `[BOS]`, `[EOS]`, `[PAD]`）。

## 2. 关键算法回顾与实现细节

### 2.1 BPE (Byte-Pair Encoding) 训练
1.  初始化词表为所有出现的字符。
2.  统计所有相邻字符对 (Pair) 的频次。
3.  合并频次最高的对（如 'e', 's' -> 'es'），加入词表。
4.  重复步骤 2-3，直到词表达到预设大小（如 32k, 50k）。

### 2.2 Unigram 训练
1.  初始化一个巨大的词表（包含所有可能的子串）。
2.  使用 EM 算法计算每个子词的概率。
3.  计算如果移除某个子词，总 Loss 会增加多少。
4.  移除 Loss 增加最少的子词（即最不重要的词）。
5.  重复直到达到预设大小。

## 3. 词表大小 (Vocabulary Size) 的权衡

*   **词表太小**:
    *   每个词被切得太碎（如 "apple" -> "a", "p", "p", "l", "e"）。
    *   序列长度变长，推理变慢，且难以捕捉语义。
*   **词表太大**:
    *   Embedding 层参数量激增（$V \times d_{model}$）。
    *   稀疏词（Rare Words）难以训练充分。
    *   通常 LLM 选择 32k - 128k 之间（Llama 3 激进地使用了 128k）。

## 4. 字节级 BPE (Byte-Level BPE)

为了处理 Unicode 字符（如 Emoji、中文、生僻字）而不产生 `<UNK>`，现代 Tokenizer（如 GPT-2/3/4）直接在 **字节 (Byte)** 级别进行 BPE，而不是字符级别。

*   **原理**: 将文本先转为 UTF-8 字节序列，再进行 BPE 合并。
*   **优势**: 理论上可以编码任何字符串，彻底消灭 `<UNK>`。

## 5. 实战工具

*   **SentencePiece**: Google 开源，支持 BPE 和 Unigram，语言无关，直接处理原始句子（无需预分词）。
*   **Hugging Face Tokenizers**: Rust 实现，速度极快，支持所有主流算法。

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# 1. 初始化 BPE 模型
tokenizer = Tokenizer(models.BPE())

# 2. 预分词规则 (按空格)
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# 3. 训练器
trainer = trainers.BpeTrainer(vocab_size=30000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

# 4. 训练
tokenizer.train(["wiki.txt", "books.txt"], trainer)

# 5. 保存
tokenizer.save("my-tokenizer.json")
```
