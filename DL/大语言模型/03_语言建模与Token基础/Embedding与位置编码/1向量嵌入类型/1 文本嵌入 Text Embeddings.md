文本嵌入是当前最主流、应用最广的嵌入类型，将文本（词、句、段、文档）映射为固定长度的向量。

<aside>
💡

**通俗理解**：给一段文字生成一个"语义指纹"，指纹越相似的文字，含义越接近。

</aside>

---

## 七大子类

### 静态词向量

Word2Vec / GloVe / FastText。颗粒度为**词**，上下文无关，轻量经典。

### Dense / Bi-Encoder 句向量

SBERT、E5、BGE、OpenAI `text-embedding-3-*`。**当前最主流**，适合检索、RAG、聚类、分类。

### 长上下文文本嵌入

Nomic、Jina v2/v3/v4。适合长文、手册、论文、代码库、PDF。

### 查询/文档区分嵌入（DPR 风格）

query 和 document 用不同方式编码，专门优化搜索场景。

### 稀疏学习嵌入（SPLADE）

保留词项可解释性，适合首阶段召回，常与 dense 混合使用。

### 多向量 / Late Interaction（ColBERT）

每段文本生成**多个** token 向量而非 1 个，检索质量更高，但索引和存储更重。

### 上下文化 Chunk 嵌入

Late Chunking / Contextual Retrieval。先利用全文上下文再生成 chunk 向量，是当前长文检索的重要升级点。

---

> 每个子类的详细技术解析、实现方法与 Python 代码示例见下方子页面。
> 

[1.1.2 Dense / Bi-Encoder 句向量 (SBERT / E5 / BGE / OpenAI)](2%20Dense%20Bi-Encoder%20句向量%20(SBERT%20E5%20BGE%20OpenAI).md)

[1.1.4 查询/文档区分嵌入 (DPR / E5 / Voyage input_type)](4%20查询%20文档区分嵌入%20(DPR%20E5%20Voyage%20input_type).md)

[1.1.1 静态词向量 Word2Vec / GloVe / FastText](1%20%20静态词向量%20Word2Vec%20GloVe%20FastText.md)

[1.1.3 长上下文文本嵌入 (Nomic / Jina v2-v4)](3%20长上下文文本嵌入%20(Nomic%20Jina%20v2-v4).md)

[1.1.5 稀疏学习嵌入 SPLADE](5%20稀疏学习嵌入%20SPLADE.md)

[1.1.7 上下文化 Chunk 嵌入 (Late Chunking / Contextual Retrieval)](1%201%207%20%E4%B8%8A%E4%B8%8B%E6%96%87%E5%8C%96%20Chunk%20%E5%B5%8C%E5%85%A5%20(Late%20Chunking%20Contextual%20Retr%20fcb2a77fb76341bfa39681f40bb86649.md)

[1.1.6 多向量 / Late Interaction (ColBERT / ColBERTv2)](6%20多向量%20Late%20Interaction%20(ColBERT%20ColBERTv2).md)