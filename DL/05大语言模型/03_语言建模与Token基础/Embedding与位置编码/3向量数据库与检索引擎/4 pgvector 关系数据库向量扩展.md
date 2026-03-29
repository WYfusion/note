---
tags:
  - LLM/数据库
  - LLM/检索技术
  - PostgreSQL
aliases:
  - PostgreSQL 向量检索
  - PostgreSQL + pgvector
  - pgvector
created: 2025-01-01
updated: 2026-03-28
---

# PostgreSQL + pgvector：关系数据库中的向量检索

> [!abstract] 模块定位
> `pgvector` 让 PostgreSQL 在保留事务、`JOIN`、过滤、备份、复制与全文检索能力的同时，支持精确与近似向量检索。这条路线并不追求“最纯粹的向量数据库”，而是追求“一套数据库把业务数据和检索数据放在同一个一致性边界内”。

## 为什么 PostgreSQL 这条路线越来越常见

> [!tip] PostgreSQL 的价值不只是“也能存向量”
> 真正有价值的是：业务主数据、权限、租户、审计、元数据过滤、全文检索和向量检索，都能放在同一套 SQL 语义和事务边界里。

- 业务表与 embedding 表天然可 `JOIN`
- 向量写入与业务写入可在同一事务内完成
- metadata filter 直接写成 `WHERE`
- 可以结合 `tsvector` / `tsquery` 做 hybrid retrieval
- DevOps 延续既有 PostgreSQL 体系，不需要额外引入新的主存系统

> [!quote] 官方能力摘要
> `pgvector` 官方文档明确支持 exact / approximate nearest neighbor search，并强调 PostgreSQL 原生的 ACID、point-in-time recovery 与 `JOIN` 等能力。
> 参考：[pgvector README](https://github.com/pgvector/pgvector)

## 先把边界讲清楚

| 维度 | PostgreSQL + pgvector 的表现 |
| --- | --- |
| 最强优势 | SQL、事务、过滤、`JOIN`、全文检索、一体化运维 |
| 典型规模 | 中小到中大型业务检索；尤其适合与业务表强耦合的知识库 |
| 主要短板 | 超大规模分布式能力、独立检索平台能力不如 [[2 专用向量数据库实战|专用向量数据库]] |
| 常见误区 | 把 `pgvector` 当成“默认比专用向量数据库慢很多”，忽略了强过滤与强事务场景下的系统整体收益 |

> [!warning] 不要把它和分布式向量平台等同
> `pgvector` 是 PostgreSQL 扩展，不是分布式向量数据库。它非常适合“一体化业务检索”，但不是所有海量检索场景的默认最优解。

## 核心能力拆解

### 1. 数据类型与距离函数

| 项目 | 说明 |
| --- | --- |
| `vector(n)` | 最常用的单精度向量类型 |
| `halfvec(n)` | 半精度向量，适合更小索引体积 |
| `sparsevec` / `bit` | 适合稀疏或二值化场景 |
| `<->` | L2 距离 |
| `<#>` | 负内积 |
| `<=>` | cosine distance |
| `<+>` | L1 距离 |

### 2. 精确检索 vs 近似检索

| 模式 | 如何触发 | 特点 | 什么时候用 |
| --- | --- | --- | --- |
| 精确检索 | 不建近似索引，直接 `ORDER BY embedding <=> ...` | 完全召回，最直接 | 数据量不大、过滤很强、需要严格结果 |
| 近似检索 | 建 `hnsw` 或 `ivfflat` 索引 | 以 recall 换速度 | 数据量大、延迟要求高 |

> [!note] 一个容易忽略的判断
> 如果过滤条件已经把候选集压到很小，精确扫描小候选集并不一定比 ANN 慢。不要一开始就默认 “向量检索 = 必须上 HNSW”。

### 3. HNSW vs IVFFlat

| 索引 | 优势 | 代价 | 关键参数 |
| --- | --- | --- | --- |
| `hnsw` | 查询性能通常更好，速度 / recall 折中优秀 | 建索引更慢、内存占用更高 | `m`、`ef_construction`、`hnsw.ef_search` |
| `ivfflat` | 建索引更快、更省内存 | recall 与查询性能通常不如 HNSW | `lists`、`ivfflat.probes` |

> [!quote] pgvector 官方对两类索引的说明
> HNSW 通常有更好的 speed-recall tradeoff，但建索引更慢、更吃内存；IVFFlat 更省内存、建索引更快，但查询表现通常弱一些。
> 参考：[pgvector README](https://github.com/pgvector/pgvector)

## 最小建模：业务字段、全文检索和向量字段放在一起

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE kb_chunks (
    id BIGSERIAL PRIMARY KEY,
    tenant_id BIGINT NOT NULL,
    doc_id BIGINT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    category TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    content_tsv TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('simple', coalesce(title, '') || ' ' || coalesce(content, ''))
    ) STORED,
    embedding VECTOR(768) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX kb_chunks_tsv_idx
ON kb_chunks
USING gin (content_tsv);

CREATE INDEX kb_chunks_embedding_hnsw_idx
ON kb_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);
```

## 最小查询：向量召回 + SQL 过滤

```sql
SELECT
    id,
    title,
    1 - (embedding <=> $1::vector) AS semantic_score
FROM kb_chunks
WHERE tenant_id = $2
  AND category = 'tech'
ORDER BY embedding <=> $1::vector
LIMIT 10;
```

## Hybrid Retrieval：全文检索和向量检索放在同一条 SQL 链里

```sql
WITH lexical AS MATERIALIZED (
    SELECT
        id,
        ts_rank_cd(content_tsv, websearch_to_tsquery('simple', $1)) AS lexical_score
    FROM kb_chunks
    WHERE tenant_id = $2
      AND content_tsv @@ websearch_to_tsquery('simple', $1)
    ORDER BY lexical_score DESC
    LIMIT 100
),
semantic AS MATERIALIZED (
    SELECT
        id,
        1 - (embedding <=> $3::vector) AS semantic_score
    FROM kb_chunks
    WHERE tenant_id = $2
    ORDER BY embedding <=> $3::vector
    LIMIT 100
)
SELECT
    c.id,
    c.title,
    coalesce(l.lexical_score, 0) AS lexical_score,
    coalesce(s.semantic_score, 0) AS semantic_score,
    0.3 * coalesce(l.lexical_score, 0) + 0.7 * coalesce(s.semantic_score, 0) AS final_score
FROM kb_chunks c
LEFT JOIN lexical l USING (id)
LEFT JOIN semantic s USING (id)
WHERE l.id IS NOT NULL OR s.id IS NOT NULL
ORDER BY final_score DESC
LIMIT 10;
```

> [!note] 这就是 PostgreSQL 路线最核心的工程价值
> `tsvector`、`jsonb`、业务字段过滤、租户隔离和向量近邻搜索，都能在同一个 SQL 查询计划里表达。

## Python 最小示例

```python
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector

conn = psycopg2.connect("postgresql://user:pass@localhost/mydb")
register_vector(conn)

query_vec = np.random.randn(768).astype("float32")
tenant_id = 7

with conn, conn.cursor() as cur:
    cur.execute(
        """
        SELECT id, title, 1 - (embedding <=> %s::vector) AS score
        FROM kb_chunks
        WHERE tenant_id = %s
          AND metadata->>'lang' = 'zh'
        ORDER BY embedding <=> %s::vector
        LIMIT 5
        """,
        (query_vec.tolist(), tenant_id, query_vec.tolist()),
    )
    for row in cur.fetchall():
        print(row)
```

## 实战建议

1. 第一版系统优先上精确检索 + SQL 过滤，只有当延迟成为问题时再补 HNSW / IVFFlat。
2. 多租户、高频过滤场景，优先考虑普通索引、部分索引、分区和更强的 `WHERE` 约束，不要只盯 ANN 参数。
3. 如果 approximate index 在强过滤下返回结果不够，优先调 `hnsw.ef_search` 或 `ivfflat.probes`。
4. 当单库扩展性开始成为瓶颈，再评估迁移到 [[2 专用向量数据库实战|专用向量数据库]]，而不是一开始就引入额外系统。

## 相关链接

**所属模块**：
- [[索引_向量数据库与检索引擎]]

**前置知识**：
- [[1 ANN 索引算法详解|ANN 索引算法详解]] — 理解 HNSW / IVFFlat 的算法基础。
-[[1_表示层与距离度量|表示层与距离度量]]] — 理解距离函数与相似度。

**相关主题**：
- [[2 专用向量数据库实战|专用向量数据库实战]] — 当检索平台需要独立扩缩容时再对照看。
- [[3 搜索引擎向量扩展|搜索引擎向量扩展]] — 如果系统已经建在搜索引擎栈上。
- [[../../../09_RAG_检索增强生成/RAG方案/9Metadata Filtering(元数据过滤)|Metadata Filtering]] — 过滤在 RAG 中的作用。

## 参考资料

- [pgvector README](https://github.com/pgvector/pgvector)
- [PostgreSQL Full Text Search](https://www.postgresql.org/docs/current/textsearch-intro.html)

