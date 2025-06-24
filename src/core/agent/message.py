"""
Agent Message System

This module defines message classes and types for agent communication
following the Model Context Protocol (MCP) specification.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


class MessageType(Enum):
    """Message types following MCP specification"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    CAPABILITY_REQUEST = "capability_request"
    CAPABILITY_RESPONSE = "capability_response"


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


@dataclass
class AgentMessage:
    """
    Represents a message between agents or between agent and system.
    
    This class follows the MCP message specification and provides
    a standardized way to communicate within the PyGent Factory system.
    """
    
    # Core message fields
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.REQUEST
    sender: str = ""
    recipient: str = ""
    content: Dict[str, Any] = field(default_factory=dict)
    
    # Message metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    priority: MessagePriority = MessagePriority.NORMAL
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    
    # Message lifecycle
    created_at: datetime = field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # Error handling
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        """Post-initialization processing"""
        if self.correlation_id is None:
            self.correlation_id = self.id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization"""
        return {
            "id": self.id,
            "type": self.type.value,
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.value,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "metadata": self.metadata,
            "headers": self.headers,
            "created_at": self.created_at.isoformat(),
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """Create message from dictionary"""
        message = cls(
            id=data.get("id", str(uuid.uuid4())),
            type=MessageType(data.get("type", "request")),
            sender=data.get("sender", ""),
            recipient=data.get("recipient", ""),
            content=data.get("content", {}),
            priority=MessagePriority(data.get("priority", 2)),
            correlation_id=data.get("correlation_id"),
            reply_to=data.get("reply_to"),
            metadata=data.get("metadata", {}),
            headers=data.get("headers", {}),
            error_code=data.get("error_code"),
            error_message=data.get("error_message"),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3)
        )
        
        # Parse timestamps
        if "timestamp" in data:
            message.timestamp = datetime.fromisoformat(data["timestamp"])
        if "created_at" in data:
            message.created_at = datetime.fromisoformat(data["created_at"])
        if "processed_at" in data and data["processed_at"]:
            message.processed_at = datetime.fromisoformat(data["processed_at"])
        if "expires_at" in data and data["expires_at"]:
            message.expires_at = datetime.fromisoformat(data["expires_at"])
        
        return message
    
    def create_response(self, content: Dict[str, Any], 
                       message_type: MessageType = MessageType.RESPONSE) -> 'AgentMessage':
        """Create a response message to this message"""
        return AgentMessage(
            type=message_type,
            sender=self.recipient,
            recipient=self.sender,
            content=content,
            correlation_id=self.correlation_id,
            reply_to=self.id,
            priority=self.priority
        )
    
    def create_error_response(self, error_code: str, error_message: str) -> 'AgentMessage':
        """Create an error response message"""
        return AgentMessage(
            type=MessageType.ERROR,
            sender=self.recipient,
            recipient=self.sender,
            content={
                "error": {
                    "code": error_code,
                    "message": error_message
                }
            },
            correlation_id=self.correlation_id,
            reply_to=self.id,
            priority=MessagePriority.HIGH,
            error_code=error_code,
            error_message=error_message
        )
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def can_retry(self) -> bool:
        """Check if message can be retried"""
        return self.retry_count < self.max_retries
    
    def increment_retry(self) -> None:
        """Increment retry count"""
        self.retry_count += 1
    
    def mark_processed(self) -> None:
        """Mark message as processed"""
        self.processed_at = datetime.utcnow()
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the message"""
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value"""
        return self.metadata.get(key, default)
    
    def add_header(self, key: str, value: str) -> None:
        """Add header to the message"""
        self.headers[key] = value
    
    def get_header(self, key: str, default: str = None) -> str:
        """Get header value"""
        return self.headers.get(key, default)
    
    def __str__(self) -> str:
        """String representation of the message"""
        return f"AgentMessage(id={self.id}, type={self.type.value}, sender={self.sender}, recipient={self.recipient})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"AgentMessage(id={self.id}, type={self.type.value}, "
                f"sender={self.sender}, recipient={self.recipient}, "
                f"priority={self.priority.value}, timestamp={self.timestamp})")


# Message factory functions for common message types
def create_request_message(sender: str, recipient: str, content: Dict[str, Any],
                          priority: MessagePriority = MessagePriority.NORMAL) -> AgentMessage:
    """Create a request message"""
    return AgentMessage(
        type=MessageType.REQUEST,
        sender=sender,
        recipient=recipient,
        content=content,
        priority=priority
    )


def create_notification_message(sender: str, recipient: str, content: Dict[str, Any],
                               priority: MessagePriority = MessagePriority.NORMAL) -> AgentMessage:
    """Create a notification message"""
    return AgentMessage(
        type=MessageType.NOTIFICATION,
        sender=sender,
        recipient=recipient,
        content=content,
        priority=priority
    )


def create_tool_call_message(sender: str, recipient: str, tool_name: str, 
                            arguments: Dict[str, Any]) -> AgentMessage:
    """Create a tool call message"""
    return AgentMessage(
        type=MessageType.TOOL_CALL,
        sender=sender,
        recipient=recipient,
        content={
            "tool_name": tool_name,
            "arguments": arguments
        },
        priority=MessagePriority.HIGH
    )


def create_capability_request_message(sender: str, recipient: str, 
                                     capability: str, parameters: Dict[str, Any]) -> AgentMessage:
    """Create a capability request message"""
    return AgentMessage(
        type=MessageType.CAPABILITY_REQUEST,
        sender=sender,
        recipient=recipient,
        content={
            "capability": capability,
            "parameters": parameters
        },
        priority=MessagePriority.NORMAL
    )
