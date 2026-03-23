## 一句话概括

直接使用 GPT-4、Claude 等大语言模型来对候选文档进行相关性排序。最灵活、最能理解复杂查询意图，但成本最高、延迟最大，通常作为其他 Reranker 效果不足时的兜底方案。

## 核心原理

### 基本思路

将排序任务转化为 LLM 的 Prompt 任务，让 LLM 利用其强大的语言理解和推理能力来判断文档与查询的相关性。

### 常见方式

**方式一：逐个打分（Pointwise）**

给 LLM 一个 query + 一个 doc，让它输出相关性分数。

`Prompt: 请判断以下文档与查询的相关性（1-10分）`

`Query: {query}`

`Document: {document}`

`相关性分数：`

- 每个文档独立打分
- 按分数排序

**方式二：列表排序（Listwise）**

给 LLM 一批候选文档，让它直接输出排序。

`Prompt: 请将以下文档按与查询的相关性从高到低排序`

`Query: {query}`

`[1] {doc_1}`

`[2] {doc_2}`

`...`

`排序结果：`

- 一次性处理多个文档
- 文档之间可以互相比较

**方式三：成对比较（Pairwise）**

给 LLM 两个文档，让它判断哪个更相关。

`Prompt: 对于查询 "{query}"，以下哪个文档更相关？`

`A: {doc_A}`

`B: {doc_B}`

`更相关的是：`

### RankGPT

RankGPT 是 LLM-as-Reranker 的代表方法，采用**滑动窗口排序**策略：

1. 将候选文档分成若干窗口（如每次 20 个）
2. LLM 对每个窗口内的文档排序
3. 窗口滑动，逐步将最相关的文档「冒泡」到前面
4. 多轮迭代得到最终排序

### 与传统 Reranker 的区别

|对比项|Cross-Encoder / BGE-Reranker|LLM-as-Reranker|
|---|---|---|
|训练|在排序数据上专门训练|通常零样本，通过 Prompt 驱动|
|成本|低~中（自部署 GPU）|高（API 调用费 or 大 GPU）|
|定制性|需要微调|修改 Prompt 即可|

## 优势

- **零样本能力最强**：无需训练数据，通过 Prompt 即可工作
- **理解复杂查询**：对歧义、多约束、推理性查询的理解远超专用模型
- **极度灵活**：修改 Prompt 即可适配不同场景和排序标准
- **可以加入业务逻辑**：在 Prompt 中描述业务特有的排序标准
- **兜底方案**：其他 Reranker 搞不定的场景，LLM 通常能处理

## 局限

- **成本最高**：API 调用费或大模型 GPU 成本
- **延迟最大**：秒级响应，不适合实时场景
- **输出不稳定**：相同输入可能输出不同排序
- **上下文窗口限制**：候选文档过多时需要分批处理
- **难以量化比较**：打分标准可能不一致

## 实用建议

- 通常作为**最后的精排层**或**离线评测工具**
- 在其他 Reranker 效果不足的**高价值查询**上使用
- 可以用 LLM 排序结果作为**训练数据**来蒸馏小模型
- 建议先尝试 BGE-Reranker / Cross-Encoder，不满足再考虑 LLM

## 典型应用

- **Cohere Rerank API**：云端 Reranker 服务
- **OpenAI / Claude API**：自定义 Prompt 实现排序
- **RankGPT**：学术框架，实现了多种 LLM 排序策略
- **RankLLM**：专门的 LLM 排序工具库

---

**相关页面**：[Reranker（精排器）](https://www.notion.so/Reranker-c676e27a75fe4518bde38787ee24006e?pvs=21) · [Rerank-first Pipeline（先精排流水线）](https://www.notion.so/Rerank-first-Pipeline-9173ce479e5c4e0b8dcb1fa1dd337e63?pvs=21) · [Agentic / Planner RAG](https://www.notion.so/Agentic-Planner-RAG-d19c59728df0427081913718b3205931?pvs=21)