## 一句话概括

Learning-to-Rank（LTR）是用机器学习模型从标注数据中自动学习最优的融合权重和排序策略。效果最好，但需要标注数据和更复杂的实现。

## 核心原理

### 基本思路

RRF 和 Weighted Sum 的融合策略都是人工设定的。LTR 的思路是：**让模型从数据中自动学习最优的融合和排序方式**。

### 输入特征

对每个 (query, doc) 对，提取一组特征：

- BM25 分数
- Dense 相似度
- BM25 排名
- Dense 排名
- 文档长度
- 查询-文档的词重叠数
- 其他任何可用信号...

`特征向量 x = [bm25_score, dense_score, bm25_rank, dense_rank, doc_len, ...]`

### 三种学习范式

**Pointwise**
- 将每个 (query, doc) 对独立处理
- 模型预测单个文档的相关性分数
- 损失函数：MSE / Cross-Entropy
- 类似回归或分类任务
- 简单，但没有考虑文档间的相对关系

**Pairwise**
- 比较两个文档的相对顺序
- 模型学习：相关文档应该排在不相关文档前面
- 损失函数：Hinge Loss / Logistic Loss
- 代表：RankNet、LambdaMART

**Listwise**
- 直接优化整个排序列表的质量
- 损失函数直接优化排序指标（如 NDCG）
- 代表：LambdaMART、ListNet
- 效果最好，但实现最复杂

### LambdaMART——最常用的 LTR 算法
LambdaMART = Lambda + MART（Multiple Additive Regression Trees）

- 基于梯度提升决策树（GBDT）
- 用 Lambda 梯度直接优化 NDCG 等排序指标
- 在实践中效果稳定、高效
- 是搜索引擎行业的事实标准

## 与其他融合方法对比

|特性|RRF|Weighted Sum|Learning-to-Rank|是否需要训练数据|❌|❌（但调参需要）|✅ 需要标注|
|---|---|---|---|---|---|---|---|
|特征利用|仅排名|分数|任意特征|自动优化|❌|❌|✅|
|效果天花板|中|中~高|最高|实现复杂度|极低|低|中~高|

## 优势
- **效果最好**：可以自动发现最优的融合策略
- **特征灵活**：可以加入任意额外特征（文档新鲜度、点击率等）
- **自动优化**：无需人工调参
- **可处理非线性关系**：决策树能捕捉复杂的特征交互

## 局限

- **需要标注数据**：需要人工标注的 query-doc 相关性判断
- **实现复杂**：特征工程 + 训练 + 服务部署
- **可能过拟合**：数据量不够或分布偏移时
- **维护成本**：数据分布变化时需要重新训练

## 适用场景

- 有充足标注数据的大规模搜索系统
- 对排序质量有极高要求的场景
- 有多种检索信号需要融合的复杂系统
- 企业级搜索引擎

## 工程实现

- **XGBoost / LightGBM**：内置 LambdaMART，最常用
- **CatBoost**：支持 YetiRank 等 LTR 损失
- **Elasticsearch LTR 插件**：与 ES 集成
- **TF-Ranking**：TensorFlow 生态的 LTR 库
- **Allrank**：PyTorch 生态的 LTR 库
