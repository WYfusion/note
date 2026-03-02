# Tool定义和调用

## 概述

Agent的核心能力之一是调用外部工具。定义Tool schema并执行工具调用。

## Tool Schema (OpenAI格式)

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    }
                },
                "required": ["city"]
            }
        }
    }
]
```

## 工具注册和执行

```python
from typing import Dict, Any, Callable

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
    
    def register(self, name: str, func: Callable):
        self.tools[name] = func
    
    async def execute(self, name: str, arguments: Dict[str, Any]):
        if name not in self.tools:
            raise ValueError(f"Tool {name} not found")
        return await self.tools[name](**arguments)

registry = ToolRegistry()

@registry.register("get_weather")
async def get_weather(city: str):
    return {"city": city, "weather": "sunny", "temperature": 25}
```

## 下一步

- [[11-流式响应]]
- [[12-MCP集成]]
