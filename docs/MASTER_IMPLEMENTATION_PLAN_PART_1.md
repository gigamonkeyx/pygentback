# A2A + DGM Master Implementation Plan - Part 1: Foundation & Setup

## Overview

This is the complete, step-by-step implementation plan for integrating Google's A2A Protocol and Sakana AI's Darwin Gödel Machine (DGM) into PyGent Factory. Each step is numbered and explicit for exact execution.

## Phase 1: Foundation Setup (Weeks 1-2)

### 1.1 Environment Preparation

**Step 1:** Create development branch
```bash
git checkout -b feature/a2a-dgm-integration
git push -u origin feature/a2a-dgm-integration
```

**Step 2:** Install additional dependencies
```bash
# Add to requirements.txt
echo "sse-starlette==1.6.5" >> requirements.txt
echo "pydantic[email]==2.4.2" >> requirements.txt
echo "httpx==0.25.1" >> requirements.txt
echo "python-multipart==0.0.6" >> requirements.txt
pip install -r requirements.txt
```

**Step 3:** Create directory structure
```bash
mkdir -p src/protocols/a2a
mkdir -p src/dgm/core
mkdir -p src/dgm/validation
mkdir -p src/dgm/archive
mkdir -p tests/protocols/a2a
mkdir -p tests/dgm
```

**Step 4:** Create base configuration files
- Create `src/protocols/a2a/__init__.py`
- Create `src/dgm/__init__.py`
- Create `tests/protocols/__init__.py`
- Create `tests/dgm/__init__.py`

### 1.2 A2A Protocol Foundation

**Step 5:** Create A2A data models file
File: `src/protocols/a2a/models.py`
```python
from typing import Dict, List, Optional, Any, Literal, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class TaskState(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    CANCELED = "canceled"
    FAILED = "failed"
    REJECTED = "rejected"
    AUTH_REQUIRED = "auth-required"
    UNKNOWN = "unknown"

class TaskStatus(BaseModel):
    state: TaskState
    message: Optional[str] = None
    progress: Optional[float] = Field(None, ge=0.0, le=1.0)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

class TextPart(BaseModel):
    kind: Literal["text"] = "text"
    text: str
    metadata: Optional[Dict[str, Any]] = None

class FilePart(BaseModel):
    kind: Literal["file"] = "file"
    name: str
    mimeType: str
    data: Optional[str] = None  # Base64 encoded
    uri: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class DataPart(BaseModel):
    kind: Literal["data"] = "data"
    mimeType: str
    data: str  # Base64 encoded
    metadata: Optional[Dict[str, Any]] = None

Part = Union[TextPart, FilePart, DataPart]

class Message(BaseModel):
    role: Literal["user", "assistant"]
    parts: List[Part]
    metadata: Optional[Dict[str, Any]] = None

class Artifact(BaseModel):
    name: str
    mimeType: str
    data: Optional[str] = None  # Base64 encoded
    uri: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class Task(BaseModel):
    id: str
    contextId: str
    status: TaskStatus
    history: Optional[List[Message]] = None
    artifacts: Optional[List[Artifact]] = None
    metadata: Optional[Dict[str, Any]] = None
    kind: Literal["task"] = "task"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class MessageSendParams(BaseModel):
    contextId: str
    message: Message
    agent_id: Optional[str] = None

class TaskQueryParams(BaseModel):
    task_id: str

class AgentCard(BaseModel):
    apiVersion: str = "a2a/v1"
    kind: Literal["agent"] = "agent"
    metadata: Dict[str, Any]
    spec: Dict[str, Any]
```

**Step 6:** Create A2A protocol handler
File: `src/protocols/a2a/handler.py`
```python
import uuid
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import HTTPException
from sse_starlette import EventSourceResponse

from .models import (
    Task, TaskState, TaskStatus, MessageSendParams, 
    TaskQueryParams, AgentCard, Message, Part
)

logger = logging.getLogger(__name__)

class A2AProtocolHandler:
    """A2A Protocol v0.2.1 implementation"""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.active_streams: Dict[str, asyncio.Queue] = {}
        self.agent_cards: Dict[str, AgentCard] = {}
    
    async def handle_message_send(self, params: MessageSendParams) -> Task:
        """Implement message/send method"""
        task_id = str(uuid.uuid4())
        
        task = Task(
            id=task_id,
            contextId=params.contextId,
            status=TaskStatus(state=TaskState.SUBMITTED),
            history=[params.message] if params.message else None
        )
        
        self.tasks[task_id] = task
        
        # Process message asynchronously
        asyncio.create_task(self._process_message_async(task, params))
        
        return task
    
    async def handle_message_stream(self, params: MessageSendParams) -> EventSourceResponse:
        """Implement streaming with Server-Sent Events"""
        task_id = str(uuid.uuid4())
        stream_queue = asyncio.Queue()
        self.active_streams[task_id] = stream_queue
        
        # Start processing in background
        asyncio.create_task(self._process_stream_async(task_id, params, stream_queue))
        
        return EventSourceResponse(self._stream_generator(task_id, stream_queue))
    
    async def handle_tasks_get(self, params: TaskQueryParams) -> Task:
        """Implement task retrieval"""
        task = self.tasks.get(params.task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task {params.task_id} not found")
        return task
    
    async def _process_message_async(self, task: Task, params: MessageSendParams):
        """Process message asynchronously"""
        try:
            # Update task status
            task.status.state = TaskState.WORKING
            task.updated_at = datetime.utcnow()
            
            # Simulate processing
            await asyncio.sleep(0.1)
            
            # Complete task
            task.status.state = TaskState.COMPLETED
            task.status.progress = 1.0
            task.updated_at = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error processing task {task.id}: {e}")
            task.status.state = TaskState.FAILED
            task.status.message = str(e)
            task.updated_at = datetime.utcnow()
    
    async def _process_stream_async(self, task_id: str, params: MessageSendParams, queue: asyncio.Queue):
        """Process streaming message"""
        try:
            # Send initial status
            await queue.put({"type": "status", "data": {"state": "working"}})
            
            # Simulate streaming processing
            for i in range(5):
                await asyncio.sleep(0.2)
                await queue.put({
                    "type": "progress", 
                    "data": {"progress": (i + 1) / 5}
                })
            
            # Send completion
            await queue.put({"type": "status", "data": {"state": "completed"}})
            await queue.put({"type": "end"})
            
        except Exception as e:
            await queue.put({"type": "error", "data": {"message": str(e)}})
        finally:
            if task_id in self.active_streams:
                del self.active_streams[task_id]
    
    async def _stream_generator(self, task_id: str, queue: asyncio.Queue):
        """Generate SSE events"""
        try:
            while True:
                event = await queue.get()
                if event.get("type") == "end":
                    break
                yield event
        except Exception as e:
            logger.error(f"Stream error for task {task_id}: {e}")
            yield {"type": "error", "data": {"message": str(e)}}
```

**Step 7:** Create A2A FastAPI router
File: `src/protocols/a2a/router.py`
```python
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from sse_starlette import EventSourceResponse

from .handler import A2AProtocolHandler
from .models import MessageSendParams, TaskQueryParams, Task

router = APIRouter(prefix="/a2a/v1", tags=["A2A Protocol"])

# Global handler instance (will be dependency injected later)
a2a_handler = A2AProtocolHandler()

def get_a2a_handler() -> A2AProtocolHandler:
    return a2a_handler

@router.post("/message/send", response_model=Task)
async def message_send(
    params: MessageSendParams,
    handler: A2AProtocolHandler = Depends(get_a2a_handler)
):
    """Send message to agent"""
    return await handler.handle_message_send(params)

@router.post("/message/stream")
async def message_stream(
    params: MessageSendParams,
    handler: A2AProtocolHandler = Depends(get_a2a_handler)
):
    """Stream message to agent"""
    return await handler.handle_message_stream(params)

@router.get("/tasks/{task_id}", response_model=Task)
async def get_task(
    task_id: str,
    handler: A2AProtocolHandler = Depends(get_a2a_handler)
):
    """Get task by ID"""
    query_params = TaskQueryParams(task_id=task_id)
    return await handler.handle_tasks_get(query_params)

@router.get("/.well-known/agent.json")
async def get_agent_card():
    """Get public agent card"""
    # This will be implemented in Step 12
    return JSONResponse({
        "apiVersion": "a2a/v1",
        "kind": "agent",
        "metadata": {
            "name": "pygent-factory",
            "displayName": "PyGent Factory",
            "description": "Advanced AI Agent Factory",
            "version": "1.0.0"
        },
        "spec": {
            "endpoints": {
                "a2a": "/a2a/v1"
            },
            "capabilities": ["reasoning", "evolution", "multi-agent"]
        }
    })
```

## Next Steps

This completes Part 1 of the master implementation plan. The next parts will cover:

- **Part 2**: A2A Integration with Existing Systems (Steps 8-15)
- **Part 3**: DGM Core Engine Implementation (Steps 16-25)  
- **Part 4**: DGM Integration & Validation (Steps 26-35)
- **Part 5**: Testing & Deployment (Steps 36-45)

Each part maintains the explicit numbered step format for precise execution.
