# FastAPI 快速入门

## 概述

FastAPI 是一个现代、高性能的Python Web框架，用于构建API。它基于Python类型提示，提供自动文档生成、异步支持等特性。

## 环境准备

```bash
pip install fastapi uvicorn
```

## 第一个应用

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}
```

## 运行服务

```bash
uvicorn main:app --reload
```

访问 http://127.0.0.1:8000 查看结果

## 自动文档

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

## 核心概念

| 概念 | 说明 |
|------|------|
| App | FastAPI实例 |
| Route | API端点 |
| Dependency | 依赖注入 |
| Middleware | 中间件 |

## 下一步

- [[02路由和参数]]
- [[03请求体和响应]]
