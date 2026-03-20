# Reranker 与 Cross-Encoder

## 1. 为什么要重排 (Reranking)？
召回阶段为了速度，通常使用双塔模型 (Bi-Encoder) 计算向量相似度，或者简单的 BM25。这些方法牺牲了一定的精度。
-   **Bi-Encoder**：Query 和 Doc 独立编码，交互仅发生在最后一步（点积）。无法捕捉 Query 和 Doc 之间的细粒度交互。
-   **Reranker**：对召回的 Top-K（如 50个）文档进行精细打分，筛选出 Top-N（如 5个）给 LLM。

## 2. Cross-Encoder 架构
Cross-Encoder 是目前最主流的 Reranker 架构。
-   **原理**：将 Query 和 Document 拼接在一起，送入 BERT 等模型进行全注意力交互 (Full Self-Attention)。
    $$ \text{Score} = \text{BERT}(\text{[CLS] } Q \text{ [SEP] } D) $$
-   **优势**：精度极高，能捕捉复杂的逻辑关系和细节。
-   **劣势**：计算量大，速度慢，无法预先计算 Document 向量。只能用于重排阶段。

## 3. 常用 Reranker 模型
-   **BGE-Reranker**：基于 XML-RoBERTa 或 LLaMA 训练，支持多语言。
-   **Cohere Rerank**：商业闭源 API，性能强劲。
-   **ColBERT (Contextualized Late Interaction over BERT)**：一种介于 Bi-Encoder 和 Cross-Encoder 之间的折中方案，保留了 Token 级别的交互，速度比 Cross-Encoder 快。

## 4. 语音 RAG 中的重排 (Audio Reranking)

### 4.1 文本化重排
如果 Document 是 ASR 转录文本，直接使用文本 Cross-Encoder 即可。
-   **注意**：ASR 文本通常没有标点，或者包含口语词，建议使用在口语数据上微调过的 Reranker。

### 4.2 跨模态重排 (Cross-modal Reranking)
当 Query 是文本，Document 是音频（或反之）时，需要跨模态 Reranker。
-   **架构**：类似于 VisualBERT 或 LayoutLM，将 Text Token 和 Audio Frame Token 拼接，送入多模态 Transformer。
-   **输入**：
    -   Text: "A dog barking in the distance"
    -   Audio: [Audio Spectrogram Patches]
-   **交互**：Attention 层允许文本 Token 关注音频的特定时间片。
-   **打分**：输出两者匹配度的 Score。

### 4.3 为什么 Audio RAG 特别需要 Reranker？
1.  **ASR 噪声**：召回阶段可能因为 ASR 错误召回了不相关的片段。Reranker 可以结合上下文语义进行纠错和过滤。
2.  **多模态互补**：在 Hybrid Retrieval 中，可能召回了"文本相关但声音不对"或"声音相关但内容不对"的片段。跨模态 Reranker 可以综合判断。

### 4.4 训练数据构建
-   **正样本**：(Query, Matched Audio Segment)。
-   **负样本**：(Query, Hard Negative Audio Segment)。Hard Negative 可以是同一段长音频中相邻的、但语义不符的片段。
