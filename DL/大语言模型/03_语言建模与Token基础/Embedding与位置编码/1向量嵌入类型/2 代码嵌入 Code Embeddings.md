代码嵌入将自然语言（NL）和代码映射到同一向量空间，实现**自然语言搜代码**和**代码搜代码**。

<aside>
💡

**通俗理解**：让机器能"看懂"代码的含义，而不只是匹配关键词。比如搜"排序算法"能找到 `bubble_sort()` 函数。

</aside>

---

## 核心概念

- **NL ↔ Code 联合嵌入**：把自然语言描述和代码片段编码到同一空间
- **代表模型**：CodeBERT、Jina Code Embeddings、UniXcoder
- **用途**：代码检索、问答、相似代码查找、文档到代码映射、跨编程语言匹配

---

## 当前趋势

- 通用文本 embedding 可做"代码库粗召回"
- 专用 code embedding 更适合"NL 搜代码 / 代码搜代码"精细场景
- 大型代码模型（如 StarCoder）的隐层表示也可用作 embedding

---

## Python 实战：使用 sentence-transformers 做代码检索

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# 加载代码嵌入模型
model = SentenceTransformer("jinaai/jina-embeddings-v2-base-code")

# 代码片段库
code_snippets = [
    "def bubble_sort(arr):\n    for i in range(len(arr)):\n        for j in range(len(arr)-1-i):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr",
    "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
]

# 自然语言查询
query = "排序算法实现"

# 编码
query_vec = model.encode(query)
code_vecs = model.encode(code_snippets)

# 计算余弦相似度
from numpy.linalg import norm
similarities = [np.dot(query_vec, cv) / (norm(query_vec) * norm(cv)) for cv in code_vecs]

# 排序输出
for idx in np.argsort(similarities)[::-1]:
    print(f"相似度: {similarities[idx]:.4f}")
    print(f"代码: {code_snippets[idx][:60]}...\n")
```

---

## 与通用文本嵌入对比

| 维度 | 通用文本嵌入 | 专用代码嵌入 |
| --- | --- | --- |
| 代码语义理解 | 一般 | 强 |
| 跨语言代码匹配 | 弱 | 强 |
| NL→Code 精度 | 中 | 高 |
| 模型选择 | E5、BGE 等 | CodeBERT、Jina Code |
| 适用场景 | 粗召回 | 精细检索 |