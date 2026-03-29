---
tags:
  - LLM/语言建模
aliases:
  - Tokenizer训练
  - 分词器构建
  - 词表构建
created: 2025-01-01
updated: 2026-03-28
---

# Tokenizer 的训练与构建流程

> [!abstract] 摘要
> 从零开始训练和构建自定义 Tokenizer 是大语言模型开发的重要环节。本节详细讲解数据准备、算法选择、训练过程、验证测试的完整流程，以及主流工具的使用方法和最佳实践。

## 0. 统一概念：Tokenizer 训练在做什么？

> [!important] Tokenizer 训练的本质
> **Tokenizer 训练**的本质是**从数据中学习最佳的文本分割策略**，将原始文本映射为离散的 Token 序列，同时：
- 保留语义完整性
- 控制词表大小
- 平衡 OOV 和膨胀
- 适应特定领域

---

## 1. 训练流程概览

> [!important] 四步训练法
> Tokenizer 训练的四个关键步骤：

```
1.  数据准备 (Data Preparation)
    ↓
2.  算法选择 (Algorithm Selection)
    ↓
3.  模型训练 (Model Training)
    ↓
4.  验证部署 (Validation & Deployment)
```

### 1.1 训练流程详解

| 步骤 | 目标 | 输入 | 输出 |
|------|------|------|------|
| **数据准备** | 清洗、规范化数据 | 原始文本 | 高质量语料 |
| **预分词** | 基础分割 | 文本 | 基础单元序列 |
| **模型训练** | 学习最优分割 | 基础单元 | 词表+合并规则 |
| **后处理** | 添加特殊标记 | 词表 | 最终词表 |

---

## 2. 数据准备

### 2.1 数据类型与来源

| 数据类型 | 来源 | 特点 | 适用场景 |
|----------|------|------|----------|
| **网页文本** | Common Crawl, Wikipedia | 大规模、噪声多 | 通用语言模型 |
| **书籍** | Gutenberg, 图书馆 | 高质量、长文本 | 语言建模 |
| **代码** | GitHub, Stack Overflow | 结构化、语法敏感 | 代码模型 |
| **对话** | Reddit, 聊天记录 | 口语化、多轮 | 对话模型 |
| **专业文档** | 论文、技术文档 | 术语密集 | 领域模型 |

> [!note] 数据质量原则
> **GIGO 原则**：Garbage In, Garbage Out
> - 清洗低质量数据
> - 去除重复内容
> - 过滤敏感信息

### 2.2 数据预处理

```python
def preprocess_data(raw_texts, language='en'):
    processed = []

    for text in raw_texts:
        # 基础清洗
        text = clean_text(text)

        # 语言特定处理
        if language == 'zh':
            text = normalize_chinese(text)
        elif language == 'ja':
            text = normalize_japanese(text)

        # 文本规范化
        text = normalize_text(text)

        # 长度过滤
        if 10 <= len(text.split()) <= 1000:
            processed.append(text)

    return processed
```

#### 文本清洗函数

```python
def clean_text(text):
    # 移除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)

    # 移除多余空白
    text = re.sub(r'\s+', ' ', text).strip()

    # Unicode 规范化
    text = unicodedata.normalize('NFC', text)

    # 移除控制字符
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')

    return text
```

### 2.3 数据规模与分布

> [!important] 数据规模要求
> 不同规模模型需要的数据量：

| 模型规模 | 数据量级 | 文档数 | 词汇覆盖 |
|----------|----------|--------|----------|
| **小型 (1-3B)** | 1-10 GB | 10M-100M | 10K-50K |
| **中型 (7-13B)** | 50-200 GB | 100M-1B | 50K-100K |
| **大型 (70B+)** | 1-20 TB | 1B-10B | 100K-300K |

> [!tip] 数据分布策略
> **分层采样**：
> - 60% 通用文本
> - 20% 领域文本
> - 10% 代码
> - 10% 对话

---

## 3. 算法选择

### 3.1 三大算法对比

| 算法 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| **BPE** | 通用文本、代码 | 简单、快速 | 确定性、依赖预分割 |
| **WordPiece** | BERT 类模型 | 贪心匹配、稳定 | 覆盖依赖强 |
| **Unigram** | 多语言、专业领域 | 概率化、灵活 | 计算复杂、采样困难 |

> [!important] 算法选择原则
> **选择依据**：
- **通用用途**：SentencePiece BPE（GPT、Llama）
- **理解任务**：WordPiece（BERT）
- **多语言/专业**：Unigram（XLM-R、领域模型）

### 3.2 混合分词策略

```python
class HybridTokenizer:
    def __init__(self):
        self.word_tokenizer = WordTokenizer()      # 常见词
        self.subword_tokenizer = BPETokenizer()    # 子词
        self.char_tokenizer = CharTokenizer()     # 未知词

    def tokenize(self, text):
        # 优先尝试词级别
        if is_common_word(text):
            return self.word_tokenizer.tokenize(text)

        # 尝试子词
        subwords = self.subword_tokenizer.tokenize(text)
        if len(subwords) < 3:  # 合理的子词数量
            return subwords

        # 回退到字符
        return self.char_tokenizer.tokenize(text)
```

---

## 4. 关键算法回顾与实现细节

### 4.1 BPE (Byte-Pair Encoding) 训练

> [!important] BPE 四步法
> 1. 初始化词表为所有出现的字符
> 2. 统计所有相邻字符对的频次
> 3. 合并频次最高的对（如 'e', 's' -> 'es'）
> 4. 重复直到词表达到预设大小

```python
def train_bpe(corpus, vocab_size=32000):
    # 1. 初始化词表
    vocab = initialize_vocab(corpus)
    merges = []

    # 2. 统计词频
    word_freqs = get_word_frequencies(corpus)

    # 3. 迭代合并
    for i in range(vocab_size - len(vocab)):
        # 找到最频繁的相邻对
        best_pair = find_most_frequent_pair(word_freqs)

        if best_pair is None:
            break

        # 合并
        new_token = ''.join(best_pair)
        merges.append((best_pair, new_token))

        # 更新词频
        update_frequencies(word_freqs, best_pair, new_token)

    return vocab, merges
```

> [!example] BPE 训练示例
```
初始词表: ['h', 'e', 'l', 'o', 'w', 'r', 'd']
词频统计: {'hello': 100, 'world': 50}

迭代1: 最频繁对 'l'+'l' → 'll'
词表: ['h', 'e', 'll', 'o', 'w', 'r', 'd']

迭代2: 'e'+'ll' → 'ell'
词表: ['h', 'ell', 'o', 'w', 'r', 'd']

...
最终词表包含 'hello', 'world' 等完整词
```

### 4.2 WordPiece 训练

> [!important] WordPiece 训练
> 基于最大似然估计的贪心算法：

```python
def train_wordpiece(corpus, vocab_size=30000):
    # 1. 初始化词表
    vocab = set(get_all_chars(corpus))

    # 2. 添加常见词
    for word in get_common_words(corpus, min_freq=10):
        vocab.add(word)

    # 3. 迭代添加子词
    while len(vocab) < vocab_size:
        # 计算所有可能的子词组合
        candidates = generate_candidates(vocab)

        # 评估每个候选
        best_candidate = None
        best_score = float('-inf')

        for candidate in candidates:
            # 计算似然得分
            score = compute_likelihood(corpus, vocab | {candidate})

            if score > best_score:
                best_score = score
                best_candidate = candidate

        # 添加最佳候选
        if best_candidate:
            vocab.add(best_candidate)
        else:
            break

    return vocab
```

### 4.3 Unigram 训练

> [!important] Unigram 训练步骤
> 1. 生成候选子词
> 2. 使用 EM 算法计算概率
> 3. 迭代移除不重要的子词

```python
def train_unigram(corpus, vocab_size=32000):
    # 1. 生成候选子词
    candidates = generate_subword_candidates(corpus)

    # 2. 初始化概率分布
    probs = initialize_probabilities(candidates)

    # 3. 迭代剪枝
    current_size = len(candidates)
    while current_size > vocab_size:
        # 计算损失
        loss = compute_loss(corpus, probs)

        # 评估移除每个候选的影响
        scores = {}
        for candidate in candidates:
            new_probs = probs.copy()
            del new_probs[candidate]
            new_loss = compute_loss(corpus, new_probs)
            scores[candidate] = new_loss - loss

        # 移除影响最小的 10%
        to_remove = select_candidates_to_remove(scores, 0.1)
        candidates = [c for c in candidates if c not in to_remove]
        probs = {k: v for k, v in probs.items() if k not in to_remove}

        current_size = len(candidates)

    return candidates, probs
```

---

## 5. 词表大小与字节级 BPE

### 5.1 词表大小的权衡

> [!warning] 词表大小的影响
> - **词表太小**：
>   - 每个词被切得太碎（如 "apple" → "a", "p", "p", "l", "e"）
>   - 序列长度变长，推理变慢
>   - 难以捕捉语义
> - **词表太大**：
>   - Embedding 层参数量激增（$V \times d_{model}$）
>   - 稀疏词难以训练充分
>   - 内存占用增加

**主流模型词表大小**：
- GPT-2/3/4: 50K
- Llama 2: 32K
- Llama 3: 128K（激进选择）
- BERT: 30K
- XLM-R: 250K（多语言）

### 5.2 字节级 BPE (Byte-Level BPE)

> [!important] 字节级 BPE 原理
> 为了处理 Unicode 字符（如 Emoji、中文、生僻字）而不产生 `<UNK>`，现代 Tokenizer 直接在 **字节 (Byte)** 级别进行 BPE。

**处理流程**：
```
文本: "Hello, 世界!"
  ↓
UTF-8编码: [72, 101, 108, 108, 111, 44, 32, 228, 184, 150, 231, 149, 140, 33]
  ↓
字节级BPE: ["Hello", ",", " ", "世", "界", "!"]
```

**优势**：
- 理论上可以编码任何字符串
- 彻底消灭 `<UNK>`
- 处理多语言更加统一

---

## 6. 训练配置

### 6.1 核心参数设置

> [!important] 关键配置参数

#### 词表大小

```python
# 词表大小参考
vocab_sizes = {
    'chinese': 15000,    # 中文（汉字+词）
    'japanese': 32000,   # 日语（混合）
    'english': 32000,    # 英文
    'multilingual': 250000,  # 多语言（XLM-R）
    'code': 50000,      # 代码
    'medical': 40000     # 医疗
}
```

#### 特殊 Token 定义

```python
special_tokens = {
    # 必需的
    '<pad>': 0,          # 填充
    '<unk>': 1,          # 未知
    '<s>': 2,            # 序列开始
    '</s>': 3,           # 序列结束

    # 可选的
    '<mask>': 4,         # 掩码（BERT）
    '<cls>': 5,          # 分类（BERT）
    '<sep>': 6,          # 分隔（BERT）

    # 自定义
    '<|im_start|>': 100, # 对话开始
    '<|im_end|>': 101,   # 对话结束
    '<|json|>': 102,     # JSON 格式

    # 领域特定
    '<|medical|>': 200,   # 医疗领域
    '<|code|>': 201,     # 代码标记
}
```

### 6.2 SentencePiece 配置

```python
# SentencePiece 配置
spm_config = {
    # 基础设置
    'vocab_size': 32000,
    'model_type': 'bpe',  # 'bpe', 'unigram', 'word'
    'character_coverage': 0.9995,

    # 预处理
    'input_sentence_size': 10000000,  # 训练数据量
    'shuffle_input_sentence': True,
    'seed_sentencepiece_size': 1000000,

    # 正则化
    'normalization_rule_name': 'nmt_nfkc_cf',
    'remove_extra_whitespaces': True,
    'split_by_unicode_script': True,
    'split_by_number': True,
    'split_by_whitespace': True,

    # 子词设置
    'split_digits': True,
    'allow_whitespace_only_pieces': True,
    'control_symbols': '<pad>,<unk>,<s>,</s>',
    'user_defined_symbols': '<mask>,<sep>,<cls>',
    'required_chars': '',

    # 输出设置
    'model_prefix': 'tokenizer',
    'num_threads': 16,
    'num_sub_iterations': 2,
    'max_sentence_length': 4192,
    'max_sentencepiece_length': 16,
    'hard_vocab_limit': True,
    'use_all_vocab': False,

    # ID 分配
    'unk_id': 1,
    'bos_id': 2,
    'eos_id': 3,
    'pad_id': 0
}
```

### 6.3 Hugging Face Tokenizers 配置

```python
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

# 构建 BPE Tokenizer
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

# 配置规范化
tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFD(),
    normalizers.Lowercase(),
    normalizers.NFKC(),
    normalizers.Replace("``", '"'),
    normalizers.Replace("''", '"'),
])

# 配置预分词
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Whitespace(),
    pre_tokenizers.Digits(individual_digits=True),
    pre_tokenizers.Punctuation(),
])

# 配置训练器
trainer = trainers.BpeTrainer(
    vocab_size=32000,
    min_frequency=2,
    show_progress=True,
    special_tokens=[
        "<pad>", "<unk>", "<s>", "</s>", "<mask>"
    ],
    limit_alphabet=1000,
    initial_alphabet=["<", ">", "/", "|"],
    continuing_subword_prefix="##",
)

# 训练
tokenizer.train(["corpus.txt"], trainer)
```

---

## 7. 实战工具

### 7.1 SentencePiece

> [!important] SentencePiece 使用
> Google 开发的语言无关 Tokenizer 工具

#### 安装与基础使用

```bash
pip install sentencepiece
```

#### 训练脚本

```python
import sentencepiece as spm

# 训练命令
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='m',
    vocab_size=32000,
    model_type='bpe',  # bpe, unigram, word
    character_coverage=0.9995,
    split_by_unicode_script=True,
    split_by_number=True,
    split_by_whitespace=True,
    normalization_rule_name='nmt_nfkc_cf',
    remove_extra_whitespaces=True,
    input_sentence_size=1000000,
    shuffle_input_sentence=True,
    seed_sentencepiece_size=1000000,
    shrinking_factor=0.75,
    max_sentence_length=4192,
    num_threads=16,
    num_sub_iterations=2,
    max_sentencepiece_length=16,
    split_digits=True,
    control_symbols='<pad>,<unk>,<s>,</s>',
    user_defined_symbols='<mask>,<sep>,<cls>',
    required_chars='',
    byte_fallback=True,
    vocabulary_output_piece_score=True,
    hard_vocab_limit=True,
    use_all_vocab=False,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_id=0
)
```

#### 使用训练好的模型

```python
sp = spm.SentencePieceProcessor(model_file='m.model')

# 编码
tokens = sp.encode('Hello, world!', out_type=str)
print(tokens)  # ['▁Hello', ',', '▁world', '!']

# 解码
text = sp.decode(tokens)
print(text)  # 'Hello, world!'
```

### 7.2 Hugging Face Tokenizers

> [!tip] Hugging Face 集成
方便与 Transformers 生态集成

#### 快速开始

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

# 初始化
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = Whitespace()

# 训练
trainer = BpeTrainer(special_tokens=["<unk>", "<pad>", "<s>", "</s>"])
tokenizer.train(["corpus.txt"], trainer)

# 保存
tokenizer.save("tokenizer.json")
```

#### 与 Transformers 集成

```python
from transformers import AutoTokenizer

# 保存为 Hugging Face 格式
tokenizer.save_pretrained("./tokenizer_dir")

# 使用
tokenizer = AutoTokenizer.from_pretrained("./tokenizer_dir")
```

### 7.3 自定义分词器

> [!important] 自定义分词器
> 实现特定需求的分词逻辑

#### 代码分词器示例

```python
class CodeTokenizer:
    def __init__(self):
        # 关键字
        self.keywords = {
            'def', 'class', 'if', 'else', 'for', 'while',
            'import', 'from', 'return', 'yield'
        }

        # 操作符
        self.operators = {
            '+', '-', '*', '/', '=', '==', '!=', '<', '>',
            '<=', '>=', '&&', '||', '!', '&', '|', '^'
        }

        # 分隔符
        self.delimiters = {
            '(', ')', '[', ']', '{', '}', ';', ',', '.', ':'
        }

    def tokenize(self, code):
        tokens = []
        i = 0
        n = len(code)

        while i < n:
            if code[i].isspace():
                i += 1
                continue

            # 字符串
            if code[i] in ('"', "'"):
                tokens.append(self._parse_string(code, i))
                i = self._string_end
                continue

            # 注释
            if code[i:i+2] == '//':
                tokens.append(self._parse_comment(code, i))
                i = self._comment_end
                continue

            # 数字
            if code[i].isdigit():
                tokens.append(self._parse_number(code, i))
                i = self._number_end
                continue

            # 标识符
            if code[i].isalpha() or code[i] == '_':
                token = self._parse_identifier(code, i)
                tokens.append(token)
                i = self._identifier_end
                continue

            # 操作符
            if i + 1 < n and code[i:i+2] in self.operators:
                tokens.append(code[i:i+2])
                i += 2
                continue

            # 分隔符
            if code[i] in self.delimiters:
                tokens.append(code[i])
                i += 1
                continue

            # 未知字符
            tokens.append(f'<UNKNOWN:{code[i]}>')
            i += 1

        return tokens
```

---

## 8. 验证与测试

### 8.1 验证集构建

> [!important] 验证集策略

```python
def create_validation_set(raw_texts, validation_ratio=0.1):
    # 分割数据
    train_size = int(len(raw_texts) * (1 - validation_ratio))

    # 确保验证集多样性
    train_texts, val_texts = train_test_split(
        raw_texts,
        train_size=train_size,
        test_size=len(raw_texts) - train_size,
        stratify=stratify_by_domain(raw_texts)
    )

    return train_texts, val_texts
```

### 8.2 评估指标体系

```python
def evaluate_tokenizer(tokenizer, test_texts):
    metrics = {}

    # 1. 基础指标
    metrics['vocab_size'] = len(tokenizer.get_vocab())
    metrics['oov_rate'] = calculate_oov_rate(tokenizer, test_texts)
    metrics['coverage'] = calculate_coverage(tokenizer, test_texts)

    # 2. 膨胀率
    metrics['expansion_ratio'] = calculate_expansion_ratio(tokenizer, test_texts)

    # 3. 质量指标
    metrics['semantic_preservation'] = evaluate_semantics(tokenizer, test_texts)
    metrics['readability'] = evaluate_readability(tokenizer, test_texts)

    # 4. 性能指标
    metrics['encoding_speed'] = measure_encoding_speed(tokenizer)
    metrics['decoding_speed'] = measure_decoding_speed(tokenizer)

    return metrics

def calculate_oov_rate(tokenizer, texts):
    oov_count = 0
    total_count = 0

    for text in texts:
        tokens = tokenizer.tokenize(text)
        oov_count += sum(1 for t in tokens if t == '<unk>')
        total_count += len(tokens)

    return oov_count / total_count if total_count > 0 else 0
```

### 8.3 A/B 测试框架

```python
def run_ab_test(tokenizer_a, tokenizer_b, test_data, n_samples=1000):
    results = {
        'accuracy': {},
        'efficiency': {},
        'quality': {}
    }

    # 准备测试样本
    test_samples = sample_test_data(test_data, n_samples)

    # 准确性测试
    results['accuracy']['a'] = evaluate_accuracy(tokenizer_a, test_samples)
    results['accuracy']['b'] = evaluate_accuracy(tokenizer_b, test_samples)

    # 效率测试
    results['efficiency']['a'] = measure_efficiency(tokenizer_a)
    results['efficiency']['b'] = measure_efficiency(tokenizer_b)

    # 质量评估
    results['quality']['a'] = human_evaluation(tokenizer_a, test_samples)
    results['quality']['b'] = human_evaluation(tokenizer_b, test_samples)

    # 统计显著性检验
    p_value = statistical_significance_test(
        results['accuracy']['a'],
        results['accuracy']['b']
    )

    return results, p_value
```

---

## 9. 部署与优化

### 9.1 模型导出格式

> [!important] 导出格式选择

#### SentencePiece 格式

```python
# 保存
tokenizer.save('tokenizer.model')
tokenizer.save('tokenizer.vocab')

# 加载
sp = spm.SentencePieceProcessor(model_file='tokenizer.model')
```

#### Hugging Face 格式

```python
# 保存
tokenizer.save('tokenizer.json')
tokenizer.save_pretrained('./tokenizer_dir')

# 加载
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('./tokenizer_dir')
```

#### 自定义格式

```python
def save_custom_format(tokenizer, filepath):
    # 保存词表
    vocab = tokenizer.get_vocab()
    with open(filepath, 'w', encoding='utf-8') as f:
        for token, idx in vocab.items():
            f.write(f"{token}\t{idx}\n")

def load_custom_format(filepath):
    vocab = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            token, idx = line.strip().split('\t')
            vocab[token] = int(idx)
    return vocab
```

### 9.2 性能优化

> [!tip] 生产环境优化

#### 内存优化

```python
class MemoryEfficientTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # 使用更紧凑的数据结构
        self.vocab = compress_vocab(tokenizer.get_vocab())
        # 构建前缀树
        self.trie = build_trie(self.vocab)

    def tokenize(self, text):
        # 使用前缀树加速
        return self._trie_tokenize(text, self.trie)

    def compress_vocab(self, vocab):
        # 使用更节省内存的数据类型
        return {
            token: np.uint32(idx)
            for token, idx in vocab.items()
        }
```

#### 推理优化

```python
class OptimizedTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # 构建缓存
        self.cache = LRUCache(maxsize=10000)

    def tokenize(self, text):
        # 检查缓存
        if text in self.cache:
            return self.cache[text]

        # 编码
        tokens = self.tokenizer.tokenize(text)

        # 存入缓存
        self.cache[text] = tokens
        return tokens
```

### 9.3 版本管理

> [!important] 版本控制策略

```python
class TokenizerVersionManager:
    def __init__(self, base_path):
        self.base_path = base_path
        self.version_file = os.path.join(base_path, 'versions.json')

    def create_version(self, tokenizer, version_name, description):
        # 创建版本目录
        version_dir = os.path.join(self.base_path, version_name)
        os.makedirs(version_dir, exist_ok=True)

        # 保存模型
        tokenizer.save(os.path.join(version_dir, 'tokenizer.json'))

        # 保存配置
        config = {
            'version': version_name,
            'description': description,
            'vocab_size': len(tokenizer.get_vocab()),
            'timestamp': datetime.now().isoformat(),
            'special_tokens': tokenizer.special_tokens
        }

        with open(os.path.join(version_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        # 更新版本记录
        self._update_version_record(version_name, config)

        return version_dir
```

---

## 10. 实战案例

### 10.1 中文医疗分词器

```python
class MedicalChineseTokenizer:
    def __init__(self):
        # 加载医疗词典
        self.medical_dict = load_medical_dict()
        # 加载停用词
        self.stop_words = load_stop_words()

    def tokenize(self, text):
        # 1. 基础分词
        tokens = jieba.cut(text, cut_all=False)

        # 2. 医疗术语识别
        medical_tokens = []
        for token in tokens:
            if token in self.medical_dict:
                medical_tokens.append(f'<MED>{token}</MED>')
            elif token not in self.stop_words:
                medical_tokens.append(token)

        return medical_tokens
```

### 10.2 多语言代码分词器

```python
class MultilingualCodeTokenizer:
    def __init__(self):
        # 语言特定配置
        self.lang_configs = {
            'python': {
                'keywords': ['def', 'class', 'if', 'else'],
                'string_prefixes': ['r', 'u', 'f', 'b'],
                'indentation': True
            },
            'javascript': {
                'keywords': ['function', 'const', 'let', 'var'],
                'string_prefixes': [],
                'indentation': True
            },
            'java': {
                'keywords': ['public', 'private', 'class', 'void'],
                'string_prefixes': [],
                'indentation': True
            }
        }

    def tokenize(self, code, language='python'):
        # 检测语言
        config = self.lang_configs.get(language, self.lang_configs['python'])

        # 保留结构
        tokens = []

        # 1. 处理注释
        if code.startswith('#'):
            tokens.append('<COMMENT>')
            code = code[1:]

        # 2. 保留字符串
        if '"' in code or "'" in code:
            tokens.extend(self._extract_strings(code))

        # 3. 按语言规则分词
        tokens.extend(self._language_tokenize(code, config))

        return tokens
```

---

## 相关链接

**所属模块**：[[索引_Tokenizer与分词]]

**前置知识**：
- [[01_BPE_WordPiece_Unigram]] — 理解基础算法
- [[02_特殊Token_对齐与模板的影响]] — 特殊 Token 使用
- [[03_OOV_长度膨胀_多语言与代码]] — 理解需求和挑战

**相关主题**：
- [[05_中文分词与多语言挑战]] — 中文和多语言实践
- [[../04_Transformer核心结构/模型家族/索引_模型家族|模型家族]] — 不同模型的 Tokenizer 需求

**延伸阅读**：
- [SentencePiece 官方文档](https://github.com/google/sentencepiece)
- [Hugging Face Tokenizers 教程](https://huggingface.co/docs/tokenizers/training)
- [自定义 Tokenizer 最佳实践](https://huggingface.co/docs/transformers/custom_datasets)
