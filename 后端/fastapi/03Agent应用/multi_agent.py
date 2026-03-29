from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from enum import Enum
import uuid

app = FastAPI()


class AgentType(str, Enum):
    ROUTER = "router"
    RESEARCHER = "researcher"
    CODER = "coder"
    WRITER = "writer"


class AgentMessage(BaseModel):
    sender: AgentType
    receiver: AgentType
    content: str
    metadata: Dict[str, Any] = {}


class TaskRequest(BaseModel):
    message: str
    agent_type: Optional[AgentType] = None


class TaskResponse(BaseModel):
    task_id: str
    agent_type: AgentType
    result: str
    status: str


message_history: List[AgentMessage] = []


async def process_by_agent(agent_type: AgentType, message: str) -> str:
    if agent_type == AgentType.ROUTER:
        if "code" in message.lower() or "implement" in message.lower():
            return "CODER"
        elif "research" in message.lower() or "find" in message.lower():
            return "RESEARCHER"
        elif "write" in message.lower() or "document" in message.lower():
            return "WRITER"
        return "RESEARCHER"
    
    elif agent_type == AgentType.RESEARCHER:
        return f"Research findings for: {message}"
    
    elif agent_type == AgentType.CODER:
        return f"Code implementation for: {message}"
    
    elif agent_type == AgentType.WRITER:
        return f"Documentation for: {message}"
    
    return "Task completed"


@app.post("/task", response_model=TaskResponse)
async def process_task(request: TaskRequest):
    task_id = str(uuid.uuid4())
    
    if request.agent_type:
        result = await process_by_agent(request.agent_type, request.message)
        return TaskResponse(
            task_id=task_id,
            agent_type=request.agent_type,
            result=result,
            status="completed"
        )
    
    agent_type = AgentType.ROUTER
    next_agent = await process_by_agent(agent_type, request.message)
    
    if isinstance(next_agent, str) and next_agent in [a.value for a in AgentType]:
        agent_type = AgentType(next_agent)
        result = await process_by_agent(agent_type, request.message)
    else:
        result = next_agent
    
    return TaskResponse(
        task_id=task_id,
        agent_type=agent_type,
        result=result,
        status="completed"
    )


@app.get("/agents")
def list_agents():
    return {
        "agents": [
            {"type": a.value, "description": a.name} 
            for a in AgentType
        ]
    }


@app.get("/history")
def get_history():
    return {"messages": message_history}


@app.get("/")
def root():
    return {"message": "Multi-Agent System Ready"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
