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
