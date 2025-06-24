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
    role: MessageRole
    parts: List[MessagePart]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    context: Optional[Dict[str, Any]] = None

class TaskReference(BaseModel):
    """Reference to a task being processed"""
    task_id: str
    agent_id: str
    context_id: Optional[str] = None
    priority: int = 0
    metadata: Optional[Dict[str, Any]] = None

class TaskStatus(BaseModel):
    """Current status of a task"""
    state: TaskState
    progress: Optional[float] = None  # 0.0 to 1.0
    estimated_completion: Optional[datetime] = None
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None
    result_preview: Optional[str] = None

class TaskResult(BaseModel):
    """Result of a completed task"""
    task_id: str
    success: bool
    result_data: Optional[Any] = None
    error_details: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None
    artifacts: Optional[List[str]] = None  # URLs or references to artifacts
    completion_time: datetime = Field(default_factory=datetime.utcnow)

class A2ATask(BaseModel):
    """Complete A2A task representation"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    context_id: str
    message: A2AMessage
    status: TaskStatus
    result: Optional[TaskResult] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    assigned_agent: Optional[str] = None
    requester_agent: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class MessageSendParams(BaseModel):
    """Parameters for message/send endpoint"""
    context_id: str = Field(..., description="Conversation context identifier")
    message: A2AMessage = Field(..., description="Message to send")
    stream: bool = Field(default=False, description="Enable streaming response")
    priority: int = Field(default=0, description="Task priority (higher = more urgent)")
    timeout: Optional[int] = Field(default=None, description="Timeout in seconds")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

class TaskQueryParams(BaseModel):
    """Parameters for task queries"""
    task_id: Optional[str] = None
    context_id: Optional[str] = None
    agent_id: Optional[str] = None
    state: Optional[TaskState] = None
    limit: int = Field(default=10, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    since: Optional[datetime] = None

class A2AResponse(BaseModel):
    """Standard A2A API response"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class StreamingChunk(BaseModel):
    """Streaming response chunk for Server-Sent Events"""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    chunk_type: str  # "progress", "partial_result", "completion", "error"
    content: Any
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    is_final: bool = False

class AgentCapability(BaseModel):
    """Individual agent capability specification"""
    name: str
    description: str
    input_types: List[ContentType]
    output_types: List[ContentType]
    parameters: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, float]] = None

class PeerAgent(BaseModel):
    """Peer agent information for federation"""
    agent_id: str
    name: str
    endpoint_url: str
    capabilities: List[AgentCapability]
    trust_level: float = Field(ge=0.0, le=1.0, default=0.5)
    last_seen: datetime = Field(default_factory=datetime.utcnow)
    performance_history: Optional[List[Dict[str, Any]]] = None
    version: str = "0.2.1"

# Utility functions for message handling
def create_text_message(text: str, role: MessageRole = MessageRole.USER, context: Optional[Dict[str, Any]] = None) -> A2AMessage:
    """Create a simple text message"""
    return A2AMessage(
        role=role,
        parts=[MessagePart(kind=ContentType.TEXT, text=text)],
        context=context
    )

def create_task_from_message(context_id: str, message: A2AMessage, requester_agent: Optional[str] = None) -> A2ATask:
    """Create a task from a message"""
    return A2ATask(
        context_id=context_id,
        message=message,
        status=TaskStatus(state=TaskState.SUBMITTED),
        requester_agent=requester_agent
    )

def extract_text_content(message: A2AMessage) -> str:
    """Extract all text content from a message"""
    text_parts = [part.text for part in message.parts if part.kind == ContentType.TEXT and part.text]
    return " ".join(text_parts)
