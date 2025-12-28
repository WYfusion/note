# 召回策略：BM25, Dense 与 Hybrid Search

## 1. 召回 (Retrieval) 概述
召回阶段的目标是从海量知识库（百万/千万级）中快速筛选出 Top-K（如 100个）最相关的候选文档。要求速度快，精度可以稍低。

## 2. 稀疏检索 (Sparse Retrieval)
基于关键词匹配的传统检索方法。
### 2.1 BM25 (Best Matching 25)
TF-IDF 的改进版，是目前最主流的稀疏检索算法。
-   **原理**：计算 Query 中的词在 Document 中的词频 (TF) 和逆文档频率 (IDF)，并考虑文档长度归一化。
-   **公式**：
    $$ \text{Score}(Q, D) = \sum_{i=1}^{n} IDF(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})} $$
-   **优势**：精确匹配能力强，对专有名词（人名、产品型号）敏感。
-   **劣势**：无法解决词义匹配问题（如 "手机" 和 "电话"）。

## 3. 稠密检索 (Dense Retrieval)
基于向量相似度的语义检索方法。
-   **原理**：使用 Embedding 模型将 Query 和 Document 映射为向量，计算 Cosine 相似度。
-   **优势**：捕捉语义相关性，解决同义词、多语言问题。
-   **劣势**：对精确匹配（如数字、罕见词）表现不如 BM25。

## 4. 混合检索 (Hybrid Search)
结合稀疏检索和稠密检索的优势。
-   **流程**：
    1.  **BM25 路**：召回 Top-N 文档。
    2.  **Vector 路**：召回 Top-N 文档。
    3.  **融合 (Fusion)**：将两路结果合并。
-   **RRF (Reciprocal Rank Fusion)**：一种常用的无参数融合算法。
    $$ \text{RRF\_Score}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}(d, r)} $$
    其中 $rank(d, r)$ 是文档 $d$ 在结果集 $r$ 中的排名。

## 5. 语音大模型中的召回策略

### 5.1 文本化召回 (ASR + Text Retrieval)
将音频转录为文本后，直接套用文本 RAG 的 Hybrid Search。
-   **BM25**：匹配 ASR 转录中的关键词。
-   **Dense**：匹配 ASR 转录的语义。
-   **挑战**：ASR 错误（如 "I scream" vs "Ice cream"）会严重影响 BM25 的效果。Dense Retrieval 对 ASR 噪声有一定鲁棒性。

### 5.2 跨模态混合召回 (Cross-modal Hybrid Retrieval)
同时利用音频的声学特征和转录的文本特征。
-   **路 1 (Text)**：Query Text $\leftrightarrow$ Document ASR Text (BM25/Dense)。
-   **路 2 (Audio)**：Query Audio/Text $\leftrightarrow$ Document Audio (CLAP Embedding)。
-   **融合**：使用 RRF 或加权求和合并结果。
-   **优势**：
    -   如果 ASR 错了，CLAP 可能还能匹配上（基于声音相似度）。
    -   如果 CLAP 没对齐好，ASR 文本能提供精确信息。

### 5.3 案例：音乐搜索
用户说："我想听那首很悲伤的、有小提琴的歌"。
-   **Text 路**：检索歌词、元数据（标签：Sad, Violin）。
-   **Audio 路**：检索音频的情感特征和乐器音色特征。
-   **Hybrid**：结合两者找到最匹配的曲目。
