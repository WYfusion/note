# Agent架构基础

## 概述

构建基于FastAPI的Agent服务框架。Agent核心是接收用户输入、调用LLM、返回响应的循环。

## Agent核心架构

```
User Input → Router → LLM → Tool Selector → Tool Executor → Response
```

## 基础Agent实现

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    tools: Optional[List[Dict]] = None

class ChatResponse(BaseModel):
    message: Message
    tool_calls: Optional[List[Dict]] = None

app = FastAPI()

async def call_llm(messages: List[Dict]) -> Dict:
    # 这里接入LLM API
    return {"role": "assistant", "content": "Response from LLM"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    messages = [msg.dict() for msg in request.messages]
    response = await call_llm(messages)
    return ChatResponse(**response)
```

## 下一步

- [[10-Tool定义和调用]]
- [[11-流式响应]]
