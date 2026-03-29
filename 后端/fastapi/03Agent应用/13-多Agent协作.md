# 多Agent协作

## 概述

多个Agent协同工作，处理复杂任务。常见模式包括：Router Agent、Specialist Agents、Supervisor。

## 架构模式

```
                    ┌─────────────┐
                    │   Router    │
                    └──────┬──────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
      ┌──────────┐   ┌──────────┐   ┌──────────┐
      │Researcher│   │Coder     │   │Writer    │
      └──────────┘   └──────────┘   └──────────┘
            │              │              │
            └──────────────┼──────────────┘
                           ▼
                    ┌─────────────┐
                    │  Supervisor │
                    └─────────────┘
```

## Agent通信协议

```python
from enum import Enum

class AgentType(str, Enum):
    ROUTER = "router"
    RESEARCHER = "researcher"
    CODER = "coder"
    WRITER = "writer"

class AgentMessage(BaseModel):
    sender: AgentType
    receiver: AgentType
    content: str
    metadata: Dict = {}
```

## 下一步

- [[14-Agent部署]]
