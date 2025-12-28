# RAG 总体架构与数据流

## 1. RAG (Retrieval-Augmented Generation) 概述
RAG 是一种结合了**检索系统**（Retrieval System）和**生成模型**（Generative Model）的技术架构。它通过从外部知识库中检索相关信息，增强 LLM 的上下文，从而解决幻觉问题、知识过时问题和私有数据访问问题。

### 1.1 标准数据流
1.  **Indexing (索引阶段)**：
    -   文档加载 -> 切分 (Chunking) -> Embedding -> 存入向量数据库 (Vector DB)。
2.  **Retrieval (检索阶段)**：
    -   用户 Query -> Embedding -> 在 Vector DB 中查找 Top-K 相似片段。
3.  **Generation (生成阶段)**：
    -   Prompt = System Prompt + Retrieved Context + User Query。
    -   LLM 生成回答。

## 2. 核心组件
-   **Retriever (检索器)**：负责从海量数据中快速找到相关信息。常用算法包括 Dense Retrieval (向量检索) 和 Sparse Retrieval (关键词检索)。
-   **Generator (生成器)**：负责根据检索到的信息生成自然语言回答。
-   **Vector Database (向量数据库)**：存储高维向量，支持 ANN (Approximate Nearest Neighbor) 搜索。

## 3. 语音大模型中的 RAG (Audio RAG)
随着多模态技术的发展，RAG 不再局限于纯文本。Audio RAG 涉及音频数据的检索与生成，架构更加复杂。

### 3.1 跨模态检索范式
Audio RAG 根据 Query 和 Document 的模态不同，可以分为以下几种场景：

| 场景 | Query 模态 | Document 模态 | 典型应用 | 技术难点 |
| :--- | :--- | :--- | :--- | :--- |
| **T2A (Text-to-Audio)** | 文本 | 音频 | 播客搜索、音效库检索 | 文本与音频的语义对齐 (CLAP) |
| **A2T (Audio-to-Text)** | 音频 | 文本 | 语音助手、会议记录查询 | ASR 错误传播、口语理解 |
| **A2A (Audio-to-Audio)** | 音频 | 音频 | 哼唱识曲、相似语音推荐 | 音频特征提取、风格/内容解耦 |

### 3.2 Audio RAG 数据流架构

#### 流程 A：基于 ASR 的文本化检索 (ASR-based Retrieval)
这是目前最成熟的方案，将音频问题转化为文本问题处理。
1.  **Indexing**：
    -   音频文档 -> **ASR 转录** -> 文本切分 -> Text Embedding -> Vector DB。
    -   *保留原始音频的时间戳映射*。
2.  **Retrieval**：
    -   用户语音 Query -> **ASR 转录** -> Text Embedding -> 检索文本片段。
3.  **Generation**：
    -   LLM 根据检索到的文本生成回答。
    -   (可选) TTS 将回答转为语音。

#### 流程 B：基于音频向量的原生检索 (Native Audio Retrieval)
不依赖 ASR，直接在音频语义空间进行检索，适合非语音音频（如音乐、环境音）或包含情感/语气的语音。
1.  **Indexing**：
    -   音频文档 -> Audio Encoder (如 CLAP, Wav2Vec) -> **Audio Embedding** -> Vector DB。
2.  **Retrieval**：
    -   用户 Query (文本/音频) -> Encoder -> Query Embedding -> 检索音频片段。
3.  **Generation**：
    -   Audio LLM (如 Qwen-Audio) 直接接收检索到的音频片段作为 Context，进行端到端回答。

### 3.3 关键挑战
1.  **模态对齐**：如何确保 "悲伤的钢琴曲" 这段文本和对应的音频在向量空间中距离相近？需要使用对比学习预训练模型 (如 CLAP)。
2.  **粒度问题**：音频是连续流，如何切分？按时间（每30秒）、按静音段、还是按说话人（Speaker Diarization）？
3.  **信息丢失**：ASR 会丢失语气、情感和背景音信息；而纯 Audio Embedding 可能丢失精确的关键词信息。**混合检索 (Hybrid Retrieval)** 是常见解法。
