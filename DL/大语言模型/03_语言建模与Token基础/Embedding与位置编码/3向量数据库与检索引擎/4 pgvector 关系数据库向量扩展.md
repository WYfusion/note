pgvector 是 PostgreSQL 的向量扩展，让你在同一个数据库中同时管理事务数据和向量检索。

---

## 核心优势

- **一套数据库管所有**：业务数据（用户、订单……）和向量数据在同一个库
- **SQL 原生操作**：用熟悉的 SQL 进行向量搜索和 metadata 过滤
- **ACID 事务**：向量数据和业务数据的一致性
- **快速落地**：不需要额外部署向量数据库

---

## 适用场景

- 中小规模（百万级以内）
- 业务一体化、快速落地
- 已有 PostgreSQL 基础设施

---

## 局限

- 超大规模（十亿+）性能不如专用引擎
- HNSW 索引内存占用较高
- 缺少分布式分片能力

---

## 安装与设置

```sql
-- 安装 pgvector 扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- 创建表
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT,
    category TEXT,
    embedding vector(768),  -- 768 维向量
    created_at TIMESTAMP DEFAULT NOW()
);

-- 创建 HNSW 索引（推荐）
CREATE INDEX ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- 或者 IVFFlat 索引（更省内存）
-- CREATE INDEX ON documents
-- USING ivfflat (embedding vector_cosine_ops)
-- WITH (lists = 100);
```

---

## Python 实战

```python
import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector

# 连接数据库
conn = psycopg2.connect("postgresql://user:pass@localhost/mydb")
register_vector(conn)
cur = conn.cursor()

# 创建扩展和表
cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        title TEXT,
        content TEXT,
        category TEXT,
        embedding vector(768)
    )
""")

# 插入数据
for i in range(100):
    vec = np.random.randn(768).astype(np.float32)
    cur.execute(
        "INSERT INTO documents (title, content, category, embedding) VALUES (%s, %s, %s, %s)",
        (f"Doc {i}", f"Content of doc {i}", "tech", vec.tolist())
    )
conn.commit()

# 创建 HNSW 索引
cur.execute("""
    CREATE INDEX IF NOT EXISTS docs_embedding_idx
    ON documents USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200)
""")
conn.commit()

# --- 向量搜索 ---
query_vec = np.random.randn(768).astype(np.float32)

# 基本向量搜索（Cosine 距离）
cur.execute("""
    SELECT id, title, 1 - (embedding <=> %s::vector) AS similarity
    FROM documents
    ORDER BY embedding <=> %s::vector
    LIMIT 5
""", (query_vec.tolist(), query_vec.tolist()))

print("向量搜索结果:")
for row in cur.fetchall():
    print(f"  ID: {row[0]}, Title: {row[1]}, Similarity: {row[2]:.4f}")

# --- 向量搜索 + metadata 过滤 ---
cur.execute("""
    SELECT id, title, 1 - (embedding <=> %s::vector) AS similarity
    FROM documents
    WHERE category = 'tech'
    ORDER BY embedding <=> %s::vector
    LIMIT 5
""", (query_vec.tolist(), query_vec.tolist()))

print("\n带过滤的搜索结果:")
for row in cur.fetchall():
    print(f"  ID: {row[0]}, Title: {row[1]}, Similarity: {row[2]:.4f}")

cur.close()
conn.close()
```

---

## 距离运算符

| 运算符 | 含义 | SQL 示例 |
| --- | --- | --- |
| `<=>` | Cosine 距离 | `ORDER BY embedding <=> query` |
| `<->` | L2 距离 | `ORDER BY embedding <-> query` |
| `<#>` | 内积（取负） | `ORDER BY embedding <#> query` |

---

## 与 SQLAlchemy 集成

```python
from sqlalchemy import create_engine, Column, Integer, String, text
from sqlalchemy.orm import declarative_base, Session
from pgvector.sqlalchemy import Vector
import numpy as np

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents_sa"
    id = Column(Integer, primary_key=True)
    title = Column(String)
    category = Column(String)
    embedding = Column(Vector(768))  # pgvector 类型

engine = create_engine("postgresql://user:pass@localhost/mydb")
Base.metadata.create_all(engine)

with Session(engine) as session:
    # 插入
    doc = Document(
        title="Test Document",
        category="tech",
        embedding=np.random.randn(768).tolist()
    )
    session.add(doc)
    session.commit()
    
    # 搜索
    query_vec = np.random.randn(768).tolist()
    results = (
        session.query(Document)
        .order_by(Document.embedding.cosine_distance(query_vec))
        .limit(5)
        .all()
    )
    for doc in results:
        print(f"ID: {doc.id}, Title: {doc.title}")
```