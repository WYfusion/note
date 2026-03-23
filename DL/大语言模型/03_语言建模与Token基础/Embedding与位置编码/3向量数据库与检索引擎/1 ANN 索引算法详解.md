ANN（Approximate Nearest Neighbor）近似最近邻是向量数据库的核心技术，在牺牲少量精度的前提下实现海量向量的高速检索。

---

## 为什么需要 ANN？

- **精确搜索（Brute-force）**：遍历所有向量计算距离，O(N×D) 复杂度
- **1M 向量 × 768 维**：精确搜索约需 100ms+，无法满足生产需求
- **ANN**：通过建立索引结构，将搜索复杂度降到 O(log N) 量级

---

## HNSW（Hierarchical Navigable Small World）

**当前最常用的 ANN 索引**。

- **原理**：构建多层图结构，每层是一个小世界网络。搜索从顶层开始，逐层向下贪心寻找最近邻
- **关键参数**：
    - `M`：每个节点的最大邻居数（通常 16~64），越大质量越好但内存越高
    - `ef_construction`：建索引时的搜索宽度（通常 100~500）
    - `ef_search`：查询时的搜索宽度（通常 50~500），越大越准但越慢
- **优点**：高召回率、低延迟、支持增量插入
- **缺点**：内存占用较高（需存储图结构）

```python
import hnswlib
import numpy as np

# 生成示例数据
dim = 768
num_elements = 100000
data = np.random.randn(num_elements, dim).astype(np.float32)

# 创建 HNSW 索引
index = hnswlib.Index(space="cosine", dim=dim)
index.init_index(
    max_elements=num_elements,
    M=16,                # 每个节点最大邻居数
    ef_construction=200  # 建索引时的搜索宽度
)

# 添加数据
index.add_items(data, ids=np.arange(num_elements))

# 搜索
index.set_ef(100)  # 查询时的搜索宽度
query = np.random.randn(1, dim).astype(np.float32)
labels, distances = index.knn_query(query, k=10)

print(f"最近邻 IDs: {labels[0]}")
print(f"距离: {distances[0]}")
```

---

## IVF（Inverted File Index）

- **原理**：先用 K-Means 将向量聚为 N 个簇（Voronoi cells），查询时只搜索最近的几个簇
- **关键参数**：
    - `nlist`：簇的数量（通常 sqrt(N) 到 4×sqrt(N)）
    - `nprobe`：查询时搜索的簇数（越大越准但越慢）
- **优点**：内存效率好，适合大规模数据
- **缺点**：需要训练（聚类），增量更新较麻烦

```python
import faiss
import numpy as np

dim = 768
n = 100000
data = np.random.randn(n, dim).astype(np.float32)

# IVF Flat 索引
nlist = 256  # 聚类数
quantizer = faiss.IndexFlatL2(dim)  # 精确量化器
index = faiss.IndexIVFFlat(quantizer, dim, nlist)

# 训练（聚类）
index.train(data)
index.add(data)

# 搜索
index.nprobe = 16  # 搜索 16 个最近的簇
query = np.random.randn(1, dim).astype(np.float32)
D, I = index.search(query, k=10)
print(f"最近邻 IDs: {I[0]}")
print(f"L2 距离: {D[0]}")
```

---

## PQ（Product Quantization）

- **原理**：将向量切分为若干子段，每段独立量化为一个码字，大幅压缩存储
- **关键参数**：
    - `m`：子段数量（通常 8~96）
    - `nbits`：每个子段的量化位数（通常 8）
- **优点**：**极大压缩存储**（768维 × 4字节 → 96字节），适合十亿级数据
- **缺点**：有精度损失
- **常与 IVF 组合**：IVF_PQ 兼具速度和存储优势

```python
import faiss
import numpy as np

dim = 768
n = 100000
data = np.random.randn(n, dim).astype(np.float32)

# IVF + PQ 索引
nlist = 256
m = 96  # 子段数（dim 必须能被 m 整除）
quantizer = faiss.IndexFlatL2(dim)
index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)  # 8 bits per sub-quantizer

index.train(data)
index.add(data)

# 内存对比
raw_size = n * dim * 4 / 1024 / 1024  # MB
pq_size = n * m / 1024 / 1024  # MB (每个向量 m 字节)
print(f"原始数据: {raw_size:.1f} MB")
print(f"PQ 压缩后: {pq_size:.1f} MB")
print(f"压缩比: {raw_size/pq_size:.1f}x")

# 搜索
index.nprobe = 16
D, I = index.search(np.random.randn(1, dim).astype(np.float32), k=10)
print(f"最近邻 IDs: {I[0]}")
```

---

## 索引类型对比

| 索引 | 召回率 | 延迟 | 内存 | 适用场景 |
| --- | --- | --- | --- | --- |
| Flat | 100% | 高 | 最高 | 小规模/精排基线 |
| HNSW | 很高 | 很低 | 高 | **最常用**，通用场景 |
| IVF | 高 | 低 | 中 | 大规模，可调精度-速度 |
| PQ | 中-高 | 低 | **很低** | 十亿级，内存受限 |
| IVF_PQ | 中-高 | 很低 | 低 | 大规模+压缩 |

---

## 调参指南

<aside>
🔑

**HNSW 调参**：从 M=16, ef_construction=200 开始。Recall 不够就加大 ef_search；内存紧就减小 M。

</aside>

<aside>
🔑

**IVF 调参**：nlist ≈ sqrt(N)。nprobe 从 nlist 的 5% 开始，Recall 不够就增加。

</aside>

<aside>
🔑

**PQ 调参**：m 越大精度越高但压缩率越低。先用 IVFFlat 测基线 Recall，再看 PQ 损失多少决定是否可接受。

</aside>