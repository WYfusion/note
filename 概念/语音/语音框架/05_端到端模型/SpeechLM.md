# 语音语言模型 (Speech Language Model)

## 作用
将语言模型范式扩展到语音领域，统一语音理解和生成。

## 核心思想
- 将语音离散化为token序列
- 用语言模型建模token序列
- 支持语音续写、对话等任务

## 代表模型

### GSLM (Generative Spoken Language Model)
Meta提出，基于HuBERT token

### AudioLM (Google)
语义token + 声学token的层级建模

### VALL-E (Microsoft)
TTS作为语言模型任务

### Moshi (Kyutai)
实时双向语音对话

### GPT-4o
OpenAI的原生多模态模型

## 优势
- 统一架构
- 涌现能力
- 零样本泛化
