# RAG 评估与调优

本章节介绍了如何科学地评估 RAG 系统的性能，建立从检索到生成的全链路监控体系。

## 目录

### [01_RAG评估_命中率_忠实度_幻觉.md](./01_RAG评估_命中率_忠实度_幻觉.md)
- **检索指标**：Hit Rate, MRR, NDCG。
- **生成指标**：RAGAS 三元组 (Context Relevance, Groundedness, Answer Relevance)。
- **语音 RAG 评估**：
  - **WER 鲁棒性**：ASR 错误对检索的影响。
  - **声学相关性**：CLAP Score 评估音频风格匹配度。
  - **端到端体验**：Latency 与 MOS 打分。
