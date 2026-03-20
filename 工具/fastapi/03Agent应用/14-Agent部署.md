# Agent服务部署

## 概述

将FastAPI Agent服务部署到生产环境。

## 使用Gunicorn

```bash
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## Docker部署

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
```

## 生产检查清单

1. 环境变量管理 (API Keys)
2. 日志配置
3. 健康检查端点
4. 限流和认证
5. 监控和指标

## 健康检查

```python
@app.get("/health")
def health_check():
    return {"status": "healthy"}
```
