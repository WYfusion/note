---
tags:
  - LLM/检索技术
  - 算法库
aliases:
  - Facebook AI Similarity Search
  - Faiss
created: 2025-01-01
updated: 2026-03-28
---

# Faiss 算法库实战

> [!abstract] 模块定位
> Faiss 是向量相似度搜索算法库，不是数据库。它负责把 [[1 ANN 索引算法详解|ANN 算法]] 变成可运行的索引实现，适合离线评测、自建召回服务、GPU 加速和索引压缩实验。

## 它在这组笔记里的位置

> [!note] 四层关系
> [[1 ANN 索引算法详解|ANN]] 讲原理，[[5 Faiss 算法库实战|Faiss]] 讲可运行实现，[[4 pgvector 关系数据库向量扩展|PostgreSQL + pgvector]] 和 [[2 专用向量数据库实战|专用向量数据库]] 讲如何把索引变成数据库或服务。

- 适合：实验、离线评测、自建检索服务底座
- 不适合：直接承担事务、权限、多租户和业务数据管理
- 优势：控制力高、索引类型丰富、支持 CPU / GPU
- 代价：过滤、持久化、服务化、监控都要自己补齐

## Faiss 的基本心智模型

> [!quote] 官方入门文档的核心抽象
> Faiss 的索引在构建时需要知道向量维度；不少索引需要训练；最核心的两个操作是 `add` 和 `search`。
> 参考：[Faiss Getting Started](https://github.com/facebookresearch/faiss/wiki/Getting-started)

| 动作 | 含义 |
| --- | --- |
| `train` | 让 IVF / PQ 等索引学习向量分布 |
| `add` | 把向量写入索引 |
| `search` | 做 kNN 搜索 |
| `write_index` / `read_index` | 持久化与加载索引 |

## Flat：作为精确检索基线

```python
import faiss
import numpy as np

d = 768
xb = np.random.randn(100000, d).astype("float32")
xq = np.random.randn(10, d).astype("float32")

index = faiss.IndexFlatL2(d)
index.add(xb)

D, I = index.search(xq, 5)
print(index.is_trained, index.ntotal)
print(I[0], D[0])
```

## HNSW：通用 ANN 选项

```python
import faiss
import numpy as np

d = 768
xb = np.random.randn(100000, d).astype("float32")
xq = np.random.randn(10, d).astype("float32")

index = faiss.IndexHNSWFlat(d, 32)
index.hnsw.efConstruction = 200
index.hnsw.efSearch = 128
index.add(xb)

D, I = index.search(xq, 10)
print(I[0], D[0])
```

## `index_factory`：把索引结构写成字符串

```python
import faiss
import numpy as np

d = 768
xb = np.random.randn(500000, d).astype("float32")

index = faiss.index_factory(d, "IVF256,PQ96", faiss.METRIC_L2)
index.train(xb[:100000])
index.add(xb)

index.nprobe = 32
D, I = index.search(np.random.randn(1, d).astype("float32"), 10)
print(I[0])
```

> [!tip] 如何把 `index_factory` 读懂
> `IVF256,PQ96` 可以理解为“先用 IVF 缩小候选集，再用 PQ 压缩向量”。当你已经能看懂这些字符串，后续再看 Milvus、pgvector 或其他数据库里的索引参数就会轻松很多。

## GPU 加速与工程边界

```python
import faiss

res = faiss.StandardGpuResources()
cpu_index = faiss.IndexFlatL2(768)
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
```

| 需求 | Faiss 是否合适 | 说明 |
| --- | --- | --- |
| 离线评测 / 算法实验 | 很合适 | 控制最强 |
| GPU 向量检索 | 很合适 | 常作为高性能底座 |
| 直接做数据库 | 不合适 | 没有事务、过滤、权限模型 |
| 做自建检索服务 | 可以 | 但要自己补齐服务层 |

## 与其他路线的关系

| 路线 | 角色 |
| --- | --- |
| [[1 ANN 索引算法详解|ANN 索引算法]] | 解释为什么有这些索引 |
| [[4 pgvector 关系数据库向量扩展|PostgreSQL + pgvector]] | 把索引放进数据库系统里 |
| [[2 专用向量数据库实战|专用向量数据库]] | 把索引做成独立检索平台 |
| [[3 搜索引擎向量扩展|搜索引擎扩展]] | 把向量召回并到全文搜索链路 |

## 相关链接

**所属模块**：
- [[索引_向量数据库与检索引擎]]

**前置知识**：
- [[1 ANN 索引算法详解|ANN 索引算法详解]] — 理解 HNSW、IVF、PQ 的理论。

**相关主题**：
- [[2 专用向量数据库实战|专用向量数据库实战]] — 看算法库如何变成服务。
- [[4 pgvector 关系数据库向量扩展|PostgreSQL + pgvector]] — 看数据库如何封装 ANN。

## 参考资料

- [Faiss Getting Started](https://github.com/facebookresearch/faiss/wiki/Getting-started)

