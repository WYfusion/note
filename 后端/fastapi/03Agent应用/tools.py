from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Callable, Optional, List
import json
import asyncio

app = FastAPI()


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.schemas: Dict[str, Dict] = {}
    
    def register(self, name: str, schema: Dict, func: Callable):
        self.tools[name] = func
        self.schemas[name] = schema
    
    async def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
        if name not in self.tools:
            raise ValueError(f"Tool {name} not found")
        return await self.tools[name](**arguments)
    
    def get_schemas(self) -> List[Dict]:
        return list(self.schemas.values())


registry = ToolRegistry()


async def get_weather(city: str) -> Dict[str, Any]:
    await asyncio.sleep(0.5)
    weathers = {"beijing": "sunny", "shanghai": "cloudy", "guangzhou": "rainy"}
    weather = weathers.get(city.lower(), "unknown")
    return {"city": city, "weather": weather, "temperature": 25}


async def calculate(expression: str) -> Dict[str, Any]:
    try:
        result = eval(expression)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


registry.register(
    "get_weather",
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取城市天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"}
                },
                "required": ["city"]
            }
        }
    },
    get_weather
)

registry.register(
    "calculate",
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "数学计算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"}
                },
                "required": ["expression"]
            }
        }
    },
    calculate
)


class ToolCallRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]


@app.get("/tools")
def list_tools():
    return {"tools": registry.get_schemas()}


@app.post("/tools/call")
async def call_tool(request: ToolCallRequest):
    try:
        result = await registry.execute(request.tool_name, request.arguments)
        return {"result": result}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/")
def root():
    return {"message": "Tool Registry Ready"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
