from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from enum import Enum

app = FastAPI()


class Role(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class Message(BaseModel):
    role: Role
    content: str


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: Dict[str, Any]


class ChatRequest(BaseModel):
    messages: List[Message]
    tools: Optional[List[Dict[str, Any]]] = None
    stream: bool = False


class ChatResponse(BaseModel):
    message: Message
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: Optional[str] = None


async def call_llm(messages: List[Dict], tools: Optional[List[Dict]] = None) -> Dict:
    last_message = messages[-1]["content"] if messages else ""
    
    if "weather" in last_message.lower():
        return {
            "role": "assistant",
            "content": "I'll check the weather for you.",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "name": "get_weather",
                "arguments": {"city": "Beijing"}
            }]
        }
    
    return {
        "role": "assistant",
        "content": f"You said: {last_message}. This is a response from the agent."
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    messages = [msg.model_dump() for msg in request.messages]
    
    response = await call_llm(messages, request.tools)
    
    tool_calls = None
    if "tool_calls" in response:
        tool_calls = [ToolCall(**tc) for tc in response["tool_calls"]]
    
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "agent-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": response["role"],
                "content": response.get("content", "")
            },
            "finish_reason": "tool_calls" if tool_calls else "stop"
        }]
    }


@app.get("/")
def root():
    return {"message": "Agent API Ready"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
