# Embedding 模型选择与对齐

## 1. Embedding 模型基础
Embedding 是将离散的数据（文本、音频、图像）映射到连续的低维稠密向量空间的过程。
$$ f: X \rightarrow \mathbb{R}^d $$
其中 $d$ 通常为 768, 1024, 1536 等。

### 1.1 评价指标
-   **MTEB (Massive Text Embedding Benchmark)**：评估 Embedding 模型在分类、聚类、检索等任务上的综合能力。
-   **语义相似度**：余弦相似度 (Cosine Similarity) 是最常用的度量方式：
    $$ \text{sim}(u, v) = \frac{u \cdot v}{\|u\| \|v\|} $$

## 2. 文本 Embedding 模型
-   **BERT 系列**：基于 Encoder 架构，适合提取句向量。
-   **BGE (BAAI General Embedding)**：目前开源界最强的中文/英文 Embedding 之一，支持长文本。
-   **E5 (EmbEddings from bidirEctional Encoder rEpresentations)**：通过弱监督对比学习训练，检索性能优异。

## 3. 语音 Embedding 与跨模态对齐 (Audio Embedding)
在 Audio RAG 中，我们需要将音频映射到与文本相同的向量空间，或者独立的音频语义空间。

### 3.1 CLAP (Contrastive Language-Audio Pretraining)
CLAP 借鉴了 CLIP 的思想，通过对比学习将音频和文本映射到共享的潜在空间。
-   **双塔架构**：
    -   **Audio Encoder**：处理音频输入（如 HTSAT, CNN14）。
    -   **Text Encoder**：处理文本描述（如 BERT, RoBERTa）。
-   **训练目标**：最大化成对的 (Audio, Text) 的余弦相似度，最小化非成对的相似度。
-   **应用**：
    -   **Text-to-Audio Retrieval**：用文本 Query 检索音频。
    -   **Zero-shot Audio Classification**：用类别名称作为文本 Query 进行分类。

### 3.2 ImageBind
Meta 提出的多模态对齐模型，以图像为中心，对齐了文本、音频、深度图、热力图等 6 种模态。
-   **优势**：实现了 Audio $\leftrightarrow$ Text 的对齐，即使没有直接的 Audio-Text 训练数据。

### 3.3 语音专用 Embedding
-   **Wav2Vec 2.0 / HuBERT**：自监督预训练模型，提取的特征包含丰富的语音学信息（音素、发音），但语义信息较弱。
-   **Speaker Embedding (如 d-vector, x-vector)**：专门用于区分说话人身份，用于声纹识别和说话人分离。

## 4. 模型选择策略
| 任务需求 | 推荐模型类型 | 示例模型 |
| :--- | :--- | :--- |
| **语音内容检索** (搜内容) | ASR + Text Embedding | Whisper + BGE-M3 |
| **语音语义/风格检索** (搜声音) | Cross-modal Audio Embedding | CLAP, LAION-CLAP |
| **说话人检索** (搜人) | Speaker Verification Model | ECAPA-TDNN, ResNet34-SE |
| **混合检索** | 多路召回 | ASR文本路 + CLAP音频路 |

## 5. 对齐微调 (Alignment Fine-tuning)
如果通用的 CLAP 模型在特定领域（如医学音频、方言）表现不佳，需要进行微调。
-   **数据准备**：收集 (Audio, Text Description) 对。
-   **Loss 函数**：InfoNCE Loss。
    $$ \mathcal{L} = -\sum_{i} \log \frac{\exp(\text{sim}(A_i, T_i)/\tau)}{\sum_{j} \exp(\text{sim}(A_i, T_j)/\tau)} $$
