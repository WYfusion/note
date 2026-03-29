from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import json
import asyncio
import time

app = FastAPI()


async def generate_sse(content: str):
    for word in content.split():
        data = {"content": word + " "}
        yield f"data: {json.dumps(data)}\n\n"
        await asyncio.sleep(0.1)
    yield "data: [DONE]\n\n"


async def generate_openai_format(content: str, chat_id: str = "chatcmpl-123"):
    for i, word in enumerate(content.split()):
        delta = {"content": word + " "}
        chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "agent-model",
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.05)
    
    final_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "agent-model",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


class ChatRequest(BaseModel):
    message: str
    stream: bool = True


@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.stream:
        return {"content": f"Echo: {request.message}"}
    
    return StreamingResponse(
        generate_openai_format(f"Echo: {request.message}"),
        media_type="text/event-stream"
    )


@app.get("/sse")
async def sse_demo():
    content = "This is a streaming response demonstration"
    return StreamingResponse(
        generate_sse(content),
        media_type="text/event-stream"
    )


@app.get("/")
def root():
    return {"message": "Streaming Demo Ready"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
