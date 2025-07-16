"""
A2A Protocol Message Models

Implements the core message structures for the Agent-to-Agent Protocol v0.2.1
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import uuid

class MessageRole(str, Enum):
    """Message role types"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"

class ContentType(str, Enum):
    """Content types supported in messages"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    JSON = "json"
    BINARY = "binary"

class TaskState(str, Enum):
    """Task lifecycle states"""
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    CANCELED = "canceled"
    FAILED = "failed"
    REJECTED = "rejected"
    AUTH_REQUIRED = "auth-required"
    UNKNOWN = "unknown"

class MessagePart(BaseModel):
    """Individual message content part"""
    kind: ContentType
    text: Optional[str] = None
    data: Optional[str] = None  # Base64 encoded for binary content
    mime_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class A2AMessage(BaseModel):
    """Core A2A message structure"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole
    content: List[MessagePart]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class TaskMessage(BaseModel):
    """Task-specific message"""
    task_id: str
    message: A2AMessage
    state: TaskState = TaskState.SUBMITTED
    priority: int = Field(default=5, ge=1, le=10)
    timeout: Optional[int] = None  # seconds

class AgentCapability(BaseModel):
    """Agent capability description"""
    name: str
    description: str
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None

class AgentSkill(BaseModel):
    """Individual agent skill"""
    name: str
    description: str
    category: str = "general"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

class AgentCapabilities(BaseModel):
    """Collection of agent capabilities and skills"""
    skills: List[AgentSkill] = []
    capabilities: List[AgentCapability] = []
    supported_formats: List[str] = ["text", "json"]
    max_context_length: Optional[int] = None

class AgentProvider(BaseModel):
    """Agent provider information"""
    name: str
    version: str = "1.0.0"
    description: str = ""
    endpoint: Optional[str] = None
    authentication: Optional[Dict[str, Any]] = None

class TaskStatus(BaseModel):
    """Current status of a task"""
    state: TaskState
    progress: Optional[float] = None  # 0.0 to 1.0
    estimated_completion: Optional[datetime] = None
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None

class AgentCard(BaseModel):
    """Agent discovery card"""
    agent_id: str
    name: str
    description: str
    version: str = "1.0.0"
    capabilities: List[AgentCapability] = []
    supported_protocols: List[str] = ["a2a-v0.2.1"]
    endpoint: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class Task(BaseModel):
    """A2A Task definition"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    messages: List[A2AMessage] = []
    state: TaskState = TaskState.SUBMITTED
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class TextPart(MessagePart):
    """Text content part"""
    kind: ContentType = ContentType.TEXT

    def __init__(self, text: str, **kwargs):
        super().__init__(kind=ContentType.TEXT, text=text, **kwargs)

class JSONPart(MessagePart):
    """JSON content part"""
    kind: ContentType = ContentType.JSON

    def __init__(self, data: Dict[str, Any], **kwargs):
        import json
        super().__init__(
            kind=ContentType.JSON,
            text=json.dumps(data),
            mime_type="application/json",
            **kwargs
        )

# Convenience functions for creating messages
def create_text_message(text: str, role: MessageRole = MessageRole.USER) -> A2AMessage:
    """Create a simple text message"""
    return A2AMessage(
        role=role,
        content=[TextPart(text)]
    )

def create_task_message(task_id: str, text: str, role: MessageRole = MessageRole.USER) -> TaskMessage:
    """Create a task message"""
    return TaskMessage(
        task_id=task_id,
        message=create_text_message(text, role)
    )

def create_agent_card(agent_id: str, name: str, description: str,
                     capabilities: List[AgentCapability] = None) -> AgentCard:
    """Create an agent card"""
    return AgentCard(
        agent_id=agent_id,
        name=name,
        description=description,
        capabilities=capabilities or []
    )