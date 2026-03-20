# 09_RAG (检索增强生成)

本模块详细解析了 RAG 系统的全流程架构，从数据索引、混合检索到生成评估。特别针对 **Audio RAG**（语音检索增强生成）这一前沿领域，补充了跨模态检索、音频切分及声学特征对齐等核心技术。

## 核心内容

### 1. 架构总览
- **[01_RAG总体架构_数据流.md](./01_RAG总体架构_数据流.md)**
  - 标准 RAG 流程：Indexing -> Retrieval -> Generation。
  - **Audio RAG 范式**：
    - **ASR-based**：音频转文本检索，成熟度高。
    - **Native Audio**：基于 CLAP 的跨模态检索，支持搜声音、搜情感。

### 2. 索引与向量库
- **[索引与向量库](./索引与向量库/索引_索引与向量库.md)**
  - **[01_Embedding模型](./索引与向量库/01_Embedding模型选择与对齐.md)**：BGE/E5 文本模型，**CLAP/ImageBind** 语音跨模态模型。
  - **[02_Chunking策略](./索引与向量库/02_Chunking策略_滑窗_语义分块.md)**：文本语义分块，**语音 VAD/Speaker 切分**策略。

### 3. 检索与重排
- **[检索与重排](./检索与重排/索引_检索与重排.md)**
  - **[01_召回策略](01_召回_BM25_Dense_Hybrid.md)**：BM25 + Vector 混合检索，语音场景下的 ASR 文本路 + 音频路融合。
  - **[02_Reranker](./检索与重排/02_Reranker_CrossEncoder.md)**：Cross-Encoder 原理，**跨模态 Reranker** 在 Audio RAG 中的纠错作用。

### 4. 评估与调优
- **[评估与调优](./评估与调优/索引_RAG评估与调优.md)**
  - **[01_RAG评估](./评估与调优/01_RAG评估_命中率_忠实度_幻觉.md)**：RAGAS 三元组指标，语音 RAG 的 **WER 鲁棒性**与**声学相关性**评估。

## 学习路径建议
1.  先掌握 **RAG 标准架构**，理解 Dense Retrieval 和 Hybrid Search 的必要性。
2.  深入 **Audio RAG**，重点关注 **CLAP 模型** 和 **VAD 切分**，这是处理非结构化音频数据的关键。
3.  关注 **Reranker**，它是提升 RAG 准确率性价比最高的组件。
