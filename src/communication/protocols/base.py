"""
Communication Protocol Base Classes

This module defines the core interfaces and data structures for communication protocols.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Protocol, runtime_checkable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import uuid


logger = logging.getLogger(__name__)


class ProtocolType(Enum):
    """Communication protocol types"""
    MCP = "mcp"                    # Model Context Protocol
    HTTP = "http"                  # HTTP REST API
    WEBSOCKET = "websocket"        # WebSocket real-time
    GRPC = "grpc"                  # gRPC high-performance
    INTERNAL = "internal"          # Internal message bus
    AGENT_TO_AGENT = "agent_to_agent"  # Direct agent communication


class MessageFormat(Enum):
    """Message format types"""
    JSON = "json"
    PROTOBUF = "protobuf"
    MSGPACK = "msgpack"
    PLAIN_TEXT = "plain_text"


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class DeliveryMode(Enum):
    """Message delivery modes"""
    FIRE_AND_FORGET = "fire_and_forget"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"
    REQUEST_RESPONSE = "request_response"


@dataclass
class ProtocolMessage:
    """Enhanced protocol message with comprehensive metadata"""
    # Basic identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    protocol: ProtocolType = ProtocolType.INTERNAL
    format: MessageFormat = MessageFormat.JSON
    
    # Routing information
    sender: str = ""
    recipient: str = ""
    message_type: str = ""
    
    # Content
    payload: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    
    # Timing and lifecycle
    timestamp: datetime = field(default_factory=datetime.utcnow)
    ttl: Optional[int] = None  # Time to live in seconds
    expires_at: Optional[datetime] = None
    
    # Message flow control
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    priority: MessagePriority = MessagePriority.NORMAL
    delivery_mode: DeliveryMode = DeliveryMode.FIRE_AND_FORGET
    
    # Retry and reliability
    retry_count: int = 0
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Tracing and monitoring
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing"""
        # Set expiration time if TTL is specified
        if self.ttl and not self.expires_at:
            self.expires_at = datetime.utcnow().timestamp() + self.ttl
        
        # Generate trace ID if not provided
        if not self.trace_id:
            self.trace_id = str(uuid.uuid4())
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        if self.expires_at:
            return datetime.utcnow().timestamp() > self.expires_at
        return False
    
    def can_retry(self) -> bool:
        """Check if message can be retried"""
        return self.retry_count < self.max_retries
    
    def increment_retry(self) -> None:
        """Increment retry count"""
        self.retry_count += 1
    
    def add_tag(self, key: str, value: str) -> None:
        """Add a tag for tracing/monitoring"""
        self.tags[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return {
            "id": self.id,
            "protocol": self.protocol.value,
            "format": self.format.value,
            "sender": self.sender,
            "recipient": self.recipient,
            "message_type": self.message_type,
            "payload": self.payload,
            "headers": self.headers,
            "timestamp": self.timestamp.isoformat(),
            "ttl": self.ttl,
            "expires_at": self.expires_at,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "priority": self.priority.value,
            "delivery_mode": self.delivery_mode.value,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProtocolMessage':
        """Create message from dictionary"""
        message = cls(
            id=data.get("id", str(uuid.uuid4())),
            protocol=ProtocolType(data.get("protocol", "internal")),
            format=MessageFormat(data.get("format", "json")),
            sender=data.get("sender", ""),
            recipient=data.get("recipient", ""),
            message_type=data.get("message_type", ""),
            payload=data.get("payload", {}),
            headers=data.get("headers", {}),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat())),
            ttl=data.get("ttl"),
            expires_at=data.get("expires_at"),
            correlation_id=data.get("correlation_id"),
            reply_to=data.get("reply_to"),
            priority=MessagePriority(data.get("priority", "normal")),
            delivery_mode=DeliveryMode(data.get("delivery_mode", "fire_and_forget")),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            retry_delay=data.get("retry_delay", 1.0),
            trace_id=data.get("trace_id"),
            span_id=data.get("span_id"),
            tags=data.get("tags", {})
        )
        return message


@dataclass
class ProtocolStats:
    """Statistics for protocol performance monitoring"""
    messages_sent: int = 0
    messages_received: int = 0
    messages_failed: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    avg_latency_ms: float = 0.0
    connection_count: int = 0
    last_activity: Optional[datetime] = None
    
    def update_sent(self, message_size: int, latency_ms: float) -> None:
        """Update sent message statistics"""
        self.messages_sent += 1
        self.bytes_sent += message_size
        
        # Update average latency
        if self.messages_sent == 1:
            self.avg_latency_ms = latency_ms
        else:
            self.avg_latency_ms = (
                (self.avg_latency_ms * (self.messages_sent - 1) + latency_ms) / self.messages_sent
            )
        
        self.last_activity = datetime.utcnow()
    
    def update_received(self, message_size: int) -> None:
        """Update received message statistics"""
        self.messages_received += 1
        self.bytes_received += message_size
        self.last_activity = datetime.utcnow()
    
    def update_failed(self) -> None:
        """Update failed message statistics"""
        self.messages_failed += 1
        self.last_activity = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "messages_failed": self.messages_failed,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "avg_latency_ms": self.avg_latency_ms,
            "connection_count": self.connection_count,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None
        }


@runtime_checkable
class CommunicationProtocol(Protocol):
    """Protocol defining the interface for communication protocols"""
    
    async def initialize(self) -> None:
        """Initialize the protocol"""
        ...
    
    async def shutdown(self) -> None:
        """Shutdown the protocol"""
        ...
    
    async def send_message(self, message: ProtocolMessage) -> bool:
        """Send a message through this protocol"""
        ...
    
    async def receive_message(self, timeout: float = 1.0) -> Optional[ProtocolMessage]:
        """Receive a message through this protocol"""
        ...
    
    def is_connected(self) -> bool:
        """Check if protocol is connected"""
        ...
    
    async def get_stats(self) -> ProtocolStats:
        """Get protocol statistics"""
        ...


class BaseCommunicationProtocol(ABC):
    """Abstract base class for communication protocols"""
    
    def __init__(self, name: str, protocol_type: ProtocolType):
        self.name = name
        self.protocol_type = protocol_type
        self.stats = ProtocolStats()
        self.message_handlers: Dict[str, List[Callable]] = {}
        self._running = False
        self._connection_id = str(uuid.uuid4())
    
    @abstractmethod
    async def _send_message_impl(self, message: ProtocolMessage) -> bool:
        """Implementation-specific message sending"""
        pass
    
    @abstractmethod
    async def _receive_message_impl(self, timeout: float) -> Optional[ProtocolMessage]:
        """Implementation-specific message receiving"""
        pass
    
    @abstractmethod
    async def _initialize_impl(self) -> None:
        """Implementation-specific initialization"""
        pass
    
    @abstractmethod
    async def _shutdown_impl(self) -> None:
        """Implementation-specific shutdown"""
        pass
    
    async def initialize(self) -> None:
        """Initialize the protocol"""
        try:
            await self._initialize_impl()
            self._running = True
            self.stats.connection_count += 1
            logger.info(f"Protocol {self.name} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize protocol {self.name}: {str(e)}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the protocol"""
        try:
            self._running = False
            await self._shutdown_impl()
            logger.info(f"Protocol {self.name} shutdown successfully")
        except Exception as e:
            logger.error(f"Error shutting down protocol {self.name}: {str(e)}")
    
    async def send_message(self, message: ProtocolMessage) -> bool:
        """Send a message with statistics tracking"""
        if not self._running:
            logger.warning(f"Protocol {self.name} not running, cannot send message")
            return False
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Set protocol type
            message.protocol = self.protocol_type
            
            # Send message
            success = await self._send_message_impl(message)
            
            # Update statistics
            end_time = asyncio.get_event_loop().time()
            latency_ms = (end_time - start_time) * 1000
            message_size = len(str(message.to_dict()))
            
            if success:
                self.stats.update_sent(message_size, latency_ms)
            else:
                self.stats.update_failed()
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to send message via {self.name}: {str(e)}")
            self.stats.update_failed()
            return False
    
    async def receive_message(self, timeout: float = 1.0) -> Optional[ProtocolMessage]:
        """Receive a message with statistics tracking"""
        if not self._running:
            return None
        
        try:
            message = await self._receive_message_impl(timeout)
            
            if message:
                message_size = len(str(message.to_dict()))
                self.stats.update_received(message_size)
                
                # Call message handlers
                await self._handle_message(message)
            
            return message
            
        except Exception as e:
            logger.error(f"Failed to receive message via {self.name}: {str(e)}")
            return None
    
    def is_connected(self) -> bool:
        """Check if protocol is connected"""
        return self._running
    
    async def get_stats(self) -> ProtocolStats:
        """Get protocol statistics"""
        return self.stats
    
    def add_message_handler(self, message_type: str, handler: Callable) -> None:
        """Add message handler for specific message type"""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)
    
    def remove_message_handler(self, message_type: str, handler: Callable) -> None:
        """Remove message handler"""
        if message_type in self.message_handlers:
            if handler in self.message_handlers[message_type]:
                self.message_handlers[message_type].remove(handler)
    
    async def _handle_message(self, message: ProtocolMessage) -> None:
        """Handle incoming message with registered handlers"""
        handlers = self.message_handlers.get(message.message_type, [])
        handlers.extend(self.message_handlers.get("*", []))  # Wildcard handlers
        
        for handler in handlers:
            try:
                await handler(message)
            except Exception as e:
                logger.error(f"Message handler failed: {str(e)}")


# Exception classes
class ProtocolError(Exception):
    """Base exception for protocol errors"""
    pass


class ProtocolTimeoutError(ProtocolError):
    """Exception raised when protocol operations timeout"""
    pass


class ProtocolConnectionError(ProtocolError):
    """Exception raised for connection-related errors"""
    pass


class ProtocolSerializationError(ProtocolError):
    """Exception raised for serialization/deserialization errors"""
    pass
