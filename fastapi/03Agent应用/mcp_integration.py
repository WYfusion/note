from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import asyncio

app = FastAPI()


class MCPTool(BaseModel):
    name: str
    description: str
    inputSchema: Dict[str, Any]


class MCPToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]


class MCPContent(BaseModel):
    type: str = "text"
    text: str


class MCPToolResult(BaseModel):
    content: List[MCPContent]
    isError: bool = False


class MCPResource(BaseModel):
    uri: str
    name: str
    mimeType: str
    text: str


mcp_tools = {
    "get_weather": MCPTool(
        name="get_weather",
        description="获取指定城市的天气信息",
        inputSchema={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"}
            },
            "required": ["city"]
        }
    ),
    "search_web": MCPTool(
        name="search_web",
        description="搜索网络获取信息",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"}
            },
            "required": ["query"]
        }
    )
}

mcp_resources = {
    "config://app": MCPResource(
        uri="config://app",
        name="app_config",
        mimeType="application/json",
        text='{"version": "1.0", "debug": true}'
    )
}


@app.get("/mcp/tools/list")
def list_tools():
    return {"tools": list(mcp_tools.values())}


@app.post("/mcp/tools/call")
async def call_tool(call: MCPToolCall) -> MCPToolResult:
    if call.name == "get_weather":
        city = call.arguments.get("city", "unknown")
        result = f"City {city}: 22°C, Sunny"
        return MCPToolResult(content=[MCPContent(text=result)])
    
    elif call.name == "search_web":
        query = call.arguments.get("query", "")
        result = f"Search results for: {query}"
        return MCPToolResult(content=[MCPContent(text=result)])
    
    raise HTTPException(status_code=404, detail=f"Tool {call.name} not found")


@app.get("/mcp/resources/list")
def list_resources():
    return {"resources": list(mcp_resources.values())}


@app.get("/mcp/resources/{uri}")
def get_resource(uri: str):
    if uri in mcp_resources:
        return mcp_resources[uri]
    raise HTTPException(status_code=404, detail=f"Resource {uri} not found")


@app.get("/")
def root():
    return {"message": "MCP Server Ready"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
