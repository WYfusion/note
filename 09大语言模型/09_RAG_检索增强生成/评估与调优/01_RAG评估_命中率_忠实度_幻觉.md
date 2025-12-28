# RAG 评估：命中率、忠实度与幻觉

## 1. RAG 评估的挑战
RAG 系统包含检索和生成两个模块，评估也需要分段进行。
-   **检索评估**：找得对不对？
-   **生成评估**：答得好不好？

## 2. 检索指标 (Retrieval Metrics)
关注召回的文档是否包含正确答案。
-   **Hit Rate (HR@K)**：Top-K 结果中至少有一个相关文档的比例。
-   **MRR (Mean Reciprocal Rank)**：第一个相关文档排名的倒数均值。
-   **NDCG (Normalized Discounted Cumulative Gain)**：考虑了相关文档的排名位置（排在前面得分更高）。

## 3. 生成指标 (Generation Metrics)
关注生成的回答质量。
-   **RAGAS (Retrieval Augmented Generation Assessment)** 框架提出了 RAG 三元组指标：
    1.  **Context Relevance (上下文相关性)**：检索到的 Context 是否与 Query 相关？（评估检索器）
    2.  **Groundedness / Faithfulness (忠实度)**：生成的 Answer 是否完全基于 Context？（评估幻觉）
    3.  **Answer Relevance (答案相关性)**：生成的 Answer 是否回答了 User Query？（评估生成器）

## 4. 评估方法
-   **LLM-as-a-Judge**：使用 GPT-4 等强模型作为裁判，对上述指标打分（1-5分）或进行二分类。
-   **Golden Dataset**：人工构建 (Query, Context, Answer) 数据集进行自动化测试。

## 5. 语音 RAG 的特有评估 (Audio RAG Evaluation)

### 5.1 ASR 错误传播评估
-   **WER (Word Error Rate)**：ASR 转录的错误率。
-   **Retrieval Robustness**：评估 WER 增加时，检索 Hit Rate 下降的幅度。
    -   测试方法：向 Query 或 Document 中人为注入 ASR 噪声，观察指标变化。

### 5.2 音频相关性评估 (Audio Relevance)
在 Audio-to-Audio 或 Text-to-Audio 检索中，"相关性" 难以用关键词匹配衡量。
-   **语义相关性**：内容是否一致（如 Query="关于AI的讨论"，Doc="AI会议录音"）。
-   **声学相关性**：风格/情感是否一致（如 Query="悲伤的音乐"，Doc="葬礼进行曲"）。
-   **评估工具**：使用 CLAP Score 计算 Query 和 Retrieved Audio 的余弦相似度。

### 5.3 端到端体验评估
-   **Latency**：从说话结束到听到第一个音频回答的延迟 (Voice-to-Audio Latency)。
-   **MOS (Mean Opinion Score)**：人工对回答的语音质量、自然度、有用性进行打分。

## 6. 幻觉检测 (Hallucination Detection)
在 RAG 中，幻觉主要表现为：
1.  **无中生有**：Context 中没有提及，模型自己编造。
2.  **张冠李戴**：Context 中提到了 A，模型说是 B。

**检测方法**：
-   **SelfCheckGPT**：采样多次回答，检查一致性。
-   **NLI (Natural Language Inference)**：使用蕴含模型判断 Context 是否蕴含 Answer。
