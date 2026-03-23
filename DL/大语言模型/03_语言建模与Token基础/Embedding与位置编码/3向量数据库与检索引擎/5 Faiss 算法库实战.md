Faiss（Facebook AI Similarity Search）是 Meta 开源的向量相似度搜索库，是许多向量数据库的底层基础。

---

## 定位

- **不是数据库**，而是一个**算法库**
- 提供各种 ANN 索引的高效实现
- 适合：实验、离线评测、自建服务底座、嵌入到自有系统
- 支持 CPU 和 GPU 加速

---

## 安装

```bash
# CPU 版本
pip install faiss-cpu

# GPU 版本（需要 CUDA）
pip install faiss-gpu
```

---

## 基本用法：Flat 精确搜索

```python
import faiss
import numpy as np

dim = 768
n = 100000  # 数据库大小
nq = 10     # 查询数量

# 生成数据
np.random.seed(42)
xb = np.random.randn(n, dim).astype(np.float32)  # 数据库向量
xq = np.random.randn(nq, dim).astype(np.float32)  # 查询向量

# 创建 Flat 索引（精确搜索，L2 距离）
index = faiss.IndexFlatL2(dim)
print(f"训练前是否已训练: {index.is_trained}")

# 添加数据
index.add(xb)
print(f"索引中的向量数: {index.ntotal}")

# 搜索 top-5
k = 5
D, I = index.search(xq, k)  # D: 距离, I: 索引
print(f"\n第一个查询的 top-{k} 结果:")
print(f"  IDs: {I[0]}")
print(f"  L2 距离: {D[0]}")
```

---

## HNSW 索引

```python
import faiss
import numpy as np
import time

dim = 768
n = 100000

xb = np.random.randn(n, dim).astype(np.float32)
xq = np.random.randn(10, dim).astype(np.float32)

# HNSW 索引
M = 32  # 邻居数
index = faiss.IndexHNSWFlat(dim, M)
index.hnsw.efConstruction = 200  # 建索引搜索宽度
index.hnsw.efSearch = 128        # 查询搜索宽度

# 建索引（自动训练）
start = time.time()
index.add(xb)
print(f"HNSW 建索引耗时: {time.time()-start:.2f}s")

# 搜索
start = time.time()
D, I = index.search(xq, 10)
print(f"搜索耗时: {(time.time()-start)*1000:.1f}ms")

# 与精确搜索对比
flat_index = faiss.IndexFlatL2(dim)
flat_index.add(xb)
D_gt, I_gt = flat_index.search(xq, 10)

# 计算 Recall@10
recall = np.mean([len(set(I[i]) & set(I_gt[i])) / 10 for i in range(len(xq))])
print(f"Recall@10: {recall:.4f}")
```

---

## Index Factory：一行创建复杂索引

```python
import faiss
import numpy as np

dim = 768
n = 500000
xb = np.random.randn(n, dim).astype(np.float32)

# index_factory: 用字符串描述索引结构
# "IVF256,PQ96" = IVF (256 个 cluster) + PQ (96 个子量化器)
index = faiss.index_factory(dim, "IVF256,PQ96", faiss.METRIC_L2)

# 训练（IVF+PQ 需要训练）
print("训练中...")
index.train(xb[:100000])  # 用部分数据训练
print("添加数据...")
index.add(xb)

# 搜索
index.nprobe = 32  # 搜索 32 个最近的 cluster
D, I = index.search(np.random.randn(1, dim).astype(np.float32), 10)
print(f"结果 IDs: {I[0]}")

# 其他常用 index_factory 字符串：
# "Flat"                    - 精确搜索
# "HNSW32"                  - HNSW, M=32
# "IVF256,Flat"              - IVF + 精确距离
# "IVF1024,PQ64"             - IVF + PQ 压缩
# "OPQ96_768,IVF1024,PQ96"   - OPQ 预处理 + IVF + PQ
```

---

## GPU 加速

```python
import faiss
import numpy as np

dim = 768
n = 1000000
xb = np.random.randn(n, dim).astype(np.float32)

# CPU 索引
cpu_index = faiss.IndexFlatL2(dim)

# 转到 GPU
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)  # GPU 0

gpu_index.add(xb)

# GPU 搜索（速度提升 5-50x）
xq = np.random.randn(100, dim).astype(np.float32)
D, I = gpu_index.search(xq, 10)
print(f"GPU 搜索完成, 结果形状: {I.shape}")
```

---

## 索引保存与加载

```python
import faiss

# 保存
faiss.write_index(index, "my_index.faiss")

# 加载
loaded_index = faiss.read_index("my_index.faiss")
print(f"加载完成, 向量数: {loaded_index.ntotal}")
```

---

## 实用场景

| 场景 | 推荐索引 |
| --- | --- |
| 实验/评测基线 | Flat |
| 通用生产 | HNSW |
| 大数据+省内存 | IVF + PQ |
| GPU 环境 | Flat (GPU) 或 IVF (GPU) |
| 嵌入到自有服务 | 按需选择 + faiss.index_factory |