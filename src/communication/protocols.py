"""
Communication Protocols - Backward Compatibility Layer

This module provides backward compatibility for the modular communication protocol system.
All communication protocol functionality has been moved to the communication.protocols submodule
for better organization. This file maintains the original interface while
delegating to the new modular components.
"""

# Import all components from the modular communication protocol system
from .protocols import (
    ProtocolManager as ModularProtocolManager,
    ProtocolMessage as ModularProtocolMessage,
    ProtocolType as ModularProtocolType,
    MessageFormat as ModularMessageFormat,
    CommunicationProtocol as ModularCommunicationProtocol,
    BaseCommunicationProtocol,
    InternalProtocol as ModularInternalProtocol,
    MCPProtocol as ModularMCPProtocol,
    HTTPProtocol as ModularHTTPProtocol,
    WebSocketProtocol as ModularWebSocketProtocol
)

# Legacy imports for backward compatibility
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Protocol
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import json
import uuid

from ..core.agent import AgentMessage, MessageType, BaseAgent
from ..core.message_system import MessageBus
from ..config.settings import Settings


logger = logging.getLogger(__name__)


class ProtocolType(Enum):
    """Communication protocol types - Legacy compatibility wrapper"""
    MCP = "mcp"                    # Model Context Protocol
    HTTP = "http"                  # HTTP REST API
    WEBSOCKET = "websocket"        # WebSocket real-time
    GRPC = "grpc"                  # gRPC high-performance
    INTERNAL = "internal"          # Internal message bus
    AGENT_TO_AGENT = "agent_to_agent"  # Direct agent communication

    def to_modular_type(self) -> ModularProtocolType:
        """Convert to modular protocol type"""
        mapping = {
            self.MCP: ModularProtocolType.MCP,
            self.HTTP: ModularProtocolType.HTTP,
            self.WEBSOCKET: ModularProtocolType.WEBSOCKET,
            self.GRPC: ModularProtocolType.GRPC,
            self.INTERNAL: ModularProtocolType.INTERNAL,
            self.AGENT_TO_AGENT: ModularProtocolType.AGENT_TO_AGENT
        }
        return mapping.get(self, ModularProtocolType.INTERNAL)


class MessageFormat(Enum):
    """Message format types - Legacy compatibility wrapper"""
    JSON = "json"
    PROTOBUF = "protobuf"
    MSGPACK = "msgpack"
    PLAIN_TEXT = "plain_text"

    def to_modular_format(self) -> ModularMessageFormat:
        """Convert to modular message format"""
        mapping = {
            self.JSON: ModularMessageFormat.JSON,
            self.PROTOBUF: ModularMessageFormat.PROTOBUF,
            self.MSGPACK: ModularMessageFormat.MSGPACK,
            self.PLAIN_TEXT: ModularMessageFormat.PLAIN_TEXT
        }
        return mapping.get(self, ModularMessageFormat.JSON)


@dataclass
class ProtocolMessage:
    """Generic protocol message - Legacy compatibility wrapper"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    protocol: ProtocolType = ProtocolType.INTERNAL
    format: MessageFormat = MessageFormat.JSON
    sender: str = ""
    recipient: str = ""
    message_type: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    ttl: Optional[int] = None  # Time to live in seconds
    
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
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "ttl": self.ttl
        }

    def to_modular_message(self) -> ModularProtocolMessage:
        """Convert to modular message format"""
        return ModularProtocolMessage(
            id=self.id,
            protocol=self.protocol.to_modular_type(),
            format=self.format.to_modular_format(),
            sender=self.sender,
            recipient=self.recipient,
            message_type=self.message_type,
            payload=self.payload,
            headers=self.headers,
            timestamp=self.timestamp,
            correlation_id=self.correlation_id,
            reply_to=self.reply_to,
            ttl=self.ttl
        )

    @classmethod
    def from_modular_message(cls, modular_message: ModularProtocolMessage) -> 'ProtocolMessage':
        """Create legacy message from modular message"""
        # Convert modular types back to legacy
        legacy_protocol = ProtocolType.INTERNAL
        for protocol in ProtocolType:
            if protocol.to_modular_type() == modular_message.protocol:
                legacy_protocol = protocol
                break

        legacy_format = MessageFormat.JSON
        for format_type in MessageFormat:
            if format_type.to_modular_format() == modular_message.format:
                legacy_format = format_type
                break

        return cls(
            id=modular_message.id,
            protocol=legacy_protocol,
            format=legacy_format,
            sender=modular_message.sender,
            recipient=modular_message.recipient,
            message_type=modular_message.message_type,
            payload=modular_message.payload,
            headers=modular_message.headers,
            timestamp=modular_message.timestamp,
            correlation_id=modular_message.correlation_id,
            reply_to=modular_message.reply_to,
            ttl=modular_message.ttl
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProtocolMessage':
        """Create message from dictionary"""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            protocol=ProtocolType(data.get("protocol", "internal")),
            format=MessageFormat(data.get("format", "json")),
            sender=data.get("sender", ""),
            recipient=data.get("recipient", ""),
            message_type=data.get("message_type", ""),
            payload=data.get("payload", {}),
            headers=data.get("headers", {}),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat())),
            correlation_id=data.get("correlation_id"),
            reply_to=data.get("reply_to"),
            ttl=data.get("ttl")
        )


class CommunicationProtocol(ABC):
    """Abstract base class for communication protocols"""
    
    @abstractmethod
    async def send_message(self, message: ProtocolMessage) -> bool:
        """Send a message through this protocol"""
        pass
    
    @abstractmethod
    async def receive_message(self) -> Optional[ProtocolMessage]:
        """Receive a message through this protocol"""
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the protocol"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the protocol"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if protocol is connected"""
        pass


class InternalProtocol(CommunicationProtocol):
    """Internal message bus protocol"""
    
    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self.message_queue = asyncio.Queue()
        self.handlers: Dict[str, Callable] = {}
        self._running = False
    
    async def initialize(self) -> None:
        """Initialize internal protocol"""
        self._running = True
        
        # Register message handler with message bus
        self.message_bus.add_handler("*", self._handle_internal_message)
        
        logger.info("Internal protocol initialized")
    
    async def shutdown(self) -> None:
        """Shutdown internal protocol"""
        self._running = False
        logger.info("Internal protocol shutdown")
    
    def is_connected(self) -> bool:
        """Check if internal protocol is connected"""
        return self._running
    
    async def send_message(self, message: ProtocolMessage) -> bool:
        """Send message through internal message bus"""
        try:
            # Convert to AgentMessage
            agent_message = AgentMessage(
                id=message.id,
                type=MessageType.REQUEST,
                sender=message.sender,
                recipient=message.recipient,
                content=message.payload,
                metadata=message.headers,
                correlation_id=message.correlation_id
            )
            
            # Send through message bus
            await self.message_bus.send_message(agent_message)
            return True
            
        except Exception as e:
            logger.error(f"Failed to send internal message: {str(e)}")
            return False
    
    async def receive_message(self) -> Optional[ProtocolMessage]:
        """Receive message from internal queue"""
        try:
            message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
            return message
        except asyncio.TimeoutError:
            return None
    
    async def _handle_internal_message(self, agent_message: AgentMessage) -> None:
        """Handle incoming internal message"""
        try:
            # Convert AgentMessage to ProtocolMessage
            protocol_message = ProtocolMessage(
                id=agent_message.id,
                protocol=ProtocolType.INTERNAL,
                sender=agent_message.sender,
                recipient=agent_message.recipient,
                message_type=agent_message.type.value,
                payload=agent_message.content,
                headers=agent_message.metadata,
                timestamp=agent_message.timestamp,
                correlation_id=agent_message.correlation_id
            )
            
            # Add to queue
            await self.message_queue.put(protocol_message)
            
        except Exception as e:
            logger.error(f"Failed to handle internal message: {str(e)}")


class MCPProtocol(CommunicationProtocol):
    """MCP (Model Context Protocol) implementation"""
    
    def __init__(self, server_manager):
        self.server_manager = server_manager
        self._running = False
    
    async def initialize(self) -> None:
        """Initialize MCP protocol"""
        self._running = True
        logger.info("MCP protocol initialized")
    
    async def shutdown(self) -> None:
        """Shutdown MCP protocol"""
        self._running = False
        logger.info("MCP protocol shutdown")
    
    def is_connected(self) -> bool:
        """Check if MCP protocol is connected"""
        return self._running
    
    async def send_message(self, message: ProtocolMessage) -> bool:
        """Send message through MCP"""
        try:
            # Extract tool call information
            if message.message_type == "tool_call":
                tool_name = message.payload.get("tool_name")
                arguments = message.payload.get("arguments", {})
                
                if tool_name:
                    # Call MCP tool
                    result = await self.server_manager.call_tool(tool_name, arguments)
                    
                    # Send response back
                    response = ProtocolMessage(
                        protocol=ProtocolType.MCP,
                        sender="mcp_server",
                        recipient=message.sender,
                        message_type="tool_response",
                        payload={"result": result},
                        correlation_id=message.id
                    )
                    
                    # Send response back through the message sender
                    if hasattr(message, 'sender_id') and message.sender_id:
                        await self._send_response_to_sender(response, message.sender_id)
                    else:
                        logger.warning("No sender information available for tool response")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to send MCP message: {str(e)}")
            return False
    
    async def receive_message(self) -> Optional[ProtocolMessage]:
        """Receive message from MCP"""
        # MCP is typically request-response, so this might not be used
        return None


class HTTPProtocol(CommunicationProtocol):
    """HTTP REST API protocol"""
    
    def __init__(self, base_url: str, headers: Optional[Dict[str, str]] = None):
        self.base_url = base_url
        self.headers = headers or {}
        self.session = None
        self._running = False
    
    async def initialize(self) -> None:
        """Initialize HTTP protocol"""
        import aiohttp
        self.session = aiohttp.ClientSession(headers=self.headers)
        self._running = True
        logger.info(f"HTTP protocol initialized for {self.base_url}")
    
    async def shutdown(self) -> None:
        """Shutdown HTTP protocol"""
        if self.session:
            await self.session.close()
        self._running = False
        logger.info("HTTP protocol shutdown")
    
    def is_connected(self) -> bool:
        """Check if HTTP protocol is connected"""
        return self._running and self.session is not None
    
    async def send_message(self, message: ProtocolMessage) -> bool:
        """Send HTTP message"""
        try:
            if not self.session:
                return False
            
            # Determine HTTP method and endpoint
            method = message.headers.get("method", "POST")
            endpoint = message.headers.get("endpoint", "/api/message")
            url = f"{self.base_url}{endpoint}"
            
            # Prepare request data
            data = message.to_dict()
            
            # Send request
            async with self.session.request(method, url, json=data) as response:
                if response.status < 400:
                    return True
                else:
                    logger.error(f"HTTP request failed with status {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to send HTTP message: {str(e)}")
            return False
    
    async def receive_message(self) -> Optional[ProtocolMessage]:
        """Receive HTTP message (polling-based)"""
        # HTTP is typically request-response, so this might poll an endpoint
        return None


class WebSocketProtocol(CommunicationProtocol):
    """WebSocket real-time protocol"""
    
    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None):
        self.url = url
        self.headers = headers or {}
        self.websocket = None
        self.message_queue = asyncio.Queue()
        self._running = False
        self._receive_task = None
    
    async def initialize(self) -> None:
        """Initialize WebSocket protocol"""
        import websockets
        
        try:
            self.websocket = await websockets.connect(self.url, extra_headers=self.headers)
            self._running = True
            
            # Start receive task
            self._receive_task = asyncio.create_task(self._receive_loop())
            
            logger.info(f"WebSocket protocol initialized for {self.url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket: {str(e)}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown WebSocket protocol"""
        self._running = False
        
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        
        if self.websocket:
            await self.websocket.close()
        
        logger.info("WebSocket protocol shutdown")
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected"""
        return self._running and self.websocket is not None
    
    async def send_message(self, message: ProtocolMessage) -> bool:
        """Send WebSocket message"""
        try:
            if not self.websocket:
                return False
            
            # Serialize message
            data = json.dumps(message.to_dict())
            
            # Send through WebSocket
            await self.websocket.send(data)
            return True
            
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {str(e)}")
            return False
    
    async def receive_message(self) -> Optional[ProtocolMessage]:
        """Receive WebSocket message"""
        try:
            message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
            return message
        except asyncio.TimeoutError:
            return None
    
    async def _receive_loop(self) -> None:
        """WebSocket receive loop"""
        try:
            while self._running and self.websocket:
                try:
                    data = await self.websocket.recv()
                    
                    # Parse message
                    message_data = json.loads(data)
                    message = ProtocolMessage.from_dict(message_data)
                    
                    # Add to queue
                    await self.message_queue.put(message)
                    
                except Exception as e:
                    logger.error(f"Error in WebSocket receive loop: {str(e)}")
                    break
                    
        except Exception as e:
            logger.error(f"WebSocket receive loop failed: {str(e)}")


class ProtocolManager:
    """
    Protocol Manager - Legacy compatibility wrapper

    Provides backward compatibility while delegating to the modular protocol manager.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.protocols: Dict[ProtocolType, CommunicationProtocol] = {}
        self.message_handlers: Dict[str, List[Callable]] = {}
        self._running = False

        # Create modular protocol manager
        self._modular_manager = ModularProtocolManager(settings)
    
    async def initialize(self, message_bus: MessageBus, server_manager) -> None:
        """Initialize protocol manager"""
        # Initialize modular manager
        await self._modular_manager.initialize()

        # Initialize legacy protocols for backward compatibility
        internal_protocol = InternalProtocol(message_bus)
        await internal_protocol.initialize()
        self.protocols[ProtocolType.INTERNAL] = internal_protocol

        # Initialize MCP protocol
        mcp_protocol = MCPProtocol(server_manager)
        await mcp_protocol.initialize()
        self.protocols[ProtocolType.MCP] = mcp_protocol

        self._running = True
        logger.info("Protocol manager initialized")
    
    async def shutdown(self) -> None:
        """Shutdown protocol manager"""
        self._running = False
        
        # Shutdown all protocols
        for protocol in self.protocols.values():
            await protocol.shutdown()
        
        logger.info("Protocol manager shutdown")
    
    async def register_protocol(self, protocol_type: ProtocolType, 
                               protocol: CommunicationProtocol) -> None:
        """Register a new protocol"""
        await protocol.initialize()
        self.protocols[protocol_type] = protocol
        logger.info(f"Registered protocol: {protocol_type.value}")
    
    async def send_message(self, message: ProtocolMessage) -> bool:
        """Send message through appropriate protocol"""
        try:
            # Convert to modular message and delegate
            modular_message = message.to_modular_message()
            return await self._modular_manager.send_message(modular_message)
        except Exception as e:
            # Fallback to legacy protocol handling
            logger.warning(f"Modular send failed, falling back to legacy: {str(e)}")

            protocol = self.protocols.get(message.protocol)
            if not protocol:
                logger.error(f"Protocol not found: {message.protocol.value}")
                return False

            if not protocol.is_connected():
                logger.error(f"Protocol not connected: {message.protocol.value}")
                return False

            return await protocol.send_message(message)
    
    async def broadcast_message(self, message: ProtocolMessage, 
                               protocols: Optional[List[ProtocolType]] = None) -> Dict[ProtocolType, bool]:
        """Broadcast message to multiple protocols"""
        results = {}
        target_protocols = protocols or list(self.protocols.keys())
        
        for protocol_type in target_protocols:
            if protocol_type in self.protocols:
                message.protocol = protocol_type
                result = await self.send_message(message)
                results[protocol_type] = result
        
        return results
    
    def add_message_handler(self, message_type: str, handler: Callable) -> None:
        """Add message handler for specific message type"""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)
    
    def remove_message_handler(self, message_type: str, handler: Callable) -> None:
        """Remove message handler"""
        if message_type in self.message_handlers:
            try:
                self.message_handlers[message_type].remove(handler)
            except ValueError:
                pass
    
    async def handle_message(self, message: ProtocolMessage) -> None:
        """Handle incoming message"""
        handlers = self.message_handlers.get(message.message_type, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                logger.error(f"Message handler failed: {str(e)}")
    
    def get_protocol_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all protocols"""
        status = {}
        
        for protocol_type, protocol in self.protocols.items():
            status[protocol_type.value] = {
                "connected": protocol.is_connected(),
                "type": protocol.__class__.__name__
            }
        
        return status

    async def _send_response_to_sender(self, response: ProtocolMessage, sender_id: str) -> bool:
        """Send response back to the message sender."""
        try:
            # Use the protocol registry to find the appropriate protocol for the sender
            protocol_manager = getattr(self, 'protocol_manager', None)
            if protocol_manager:
                return await protocol_manager.send_message_to_agent(sender_id, response)
            else:
                logger.warning(f"No protocol manager available to send response to {sender_id}")
                return False
        except Exception as e:
            logger.error(f"Failed to send response to sender {sender_id}: {e}")
            return False
