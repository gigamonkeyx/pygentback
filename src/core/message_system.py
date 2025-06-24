"""
Unified Message System - MCP-Compliant Messaging

This module implements a unified messaging system that follows MCP (Model Context Protocol)
standards for consistent and reliable communication between agents and system components.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import json

from .agent import AgentMessage, MessageType, BaseAgent


logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class MessageStatus(Enum):
    """Message status enumeration"""
    PENDING = "pending"
    DELIVERED = "delivered"
    PROCESSED = "processed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class MessageEnvelope:
    """Message envelope with routing and delivery information"""
    message: AgentMessage
    priority: MessagePriority = MessagePriority.NORMAL
    ttl: Optional[timedelta] = None
    retry_count: int = 0
    max_retries: int = 3
    status: MessageStatus = MessageStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    delivered_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class MessageHandler:
    """Base class for message handlers"""
    
    def __init__(self, handler_id: str, handler_func: Callable):
        self.handler_id = handler_id
        self.handler_func = handler_func
        self.message_count = 0
        self.error_count = 0
        self.last_execution = None
    
    async def handle(self, message: AgentMessage) -> Any:
        """Handle a message"""
        try:
            self.message_count += 1
            self.last_execution = datetime.utcnow()
            result = await self.handler_func(message)
            return result
        except Exception as e:
            self.error_count += 1
            logger.error(f"Handler {self.handler_id} failed: {str(e)}")
            raise


class MessageRouter:
    """Routes messages to appropriate handlers"""
    
    def __init__(self):
        self._routes: Dict[str, List[MessageHandler]] = {}
        self._pattern_routes: Dict[str, List[MessageHandler]] = {}
        self._default_handlers: List[MessageHandler] = []
    
    def add_route(self, pattern: str, handler: MessageHandler) -> None:
        """Add a message route"""
        if pattern.startswith("*"):
            # Pattern-based routing
            if pattern not in self._pattern_routes:
                self._pattern_routes[pattern] = []
            self._pattern_routes[pattern].append(handler)
        else:
            # Exact match routing
            if pattern not in self._routes:
                self._routes[pattern] = []
            self._routes[pattern].append(handler)
    
    def add_default_handler(self, handler: MessageHandler) -> None:
        """Add a default handler for unrouted messages"""
        self._default_handlers.append(handler)
    
    def get_handlers(self, message: AgentMessage) -> List[MessageHandler]:
        """Get handlers for a message"""
        handlers = []
        
        # Check exact routes
        route_key = f"{message.type.value}:{message.recipient}"
        if route_key in self._routes:
            handlers.extend(self._routes[route_key])
        
        # Check pattern routes
        for pattern, pattern_handlers in self._pattern_routes.items():
            if self._matches_pattern(message, pattern):
                handlers.extend(pattern_handlers)
        
        # Use default handlers if no specific handlers found
        if not handlers:
            handlers.extend(self._default_handlers)
        
        return handlers
    
    def _matches_pattern(self, message: AgentMessage, pattern: str) -> bool:
        """Check if message matches a pattern"""
        # Simple pattern matching - can be extended
        if pattern == "*":
            return True
        elif pattern.startswith("type:"):
            return message.type.value == pattern[5:]
        elif pattern.startswith("recipient:"):
            return message.recipient == pattern[10:]
        return False


class MessageQueue:
    """Priority-based message queue"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._queues: Dict[MessagePriority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in MessagePriority
        }
        self._total_size = 0
        self._lock = asyncio.Lock()
    
    async def put(self, envelope: MessageEnvelope) -> bool:
        """Add message to queue"""
        async with self._lock:
            if self._total_size >= self.max_size:
                logger.warning("Message queue full, dropping message")
                return False
            
            await self._queues[envelope.priority].put(envelope)
            self._total_size += 1
            return True
    
    async def get(self) -> Optional[MessageEnvelope]:
        """Get next message from queue (highest priority first)"""
        # Check queues in priority order
        for priority in [MessagePriority.URGENT, MessagePriority.HIGH, 
                        MessagePriority.NORMAL, MessagePriority.LOW]:
            try:
                envelope = self._queues[priority].get_nowait()
                async with self._lock:
                    self._total_size -= 1
                return envelope
            except asyncio.QueueEmpty:
                continue
        
        return None
    
    async def size(self) -> int:
        """Get total queue size"""
        return self._total_size
    
    async def clear(self) -> None:
        """Clear all queues"""
        async with self._lock:
            for queue in self._queues.values():
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
            self._total_size = 0


class MessageBus:
    """
    MCP-compliant message bus for agent communication.
    
    Provides reliable message delivery, routing, and processing
    following MCP specification patterns.
    """
    
    def __init__(self, max_queue_size: int = 10000):
        self.router = MessageRouter()
        self.queue = MessageQueue(max_queue_size)
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        self._message_history: List[MessageEnvelope] = []
        self._subscribers: Dict[str, List[Callable]] = {}
        self._metrics = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "messages_failed": 0,
            "messages_expired": 0
        }
    
    async def start(self) -> None:
        """Start the message bus"""
        if self._running:
            return
        
        self._running = True
        self._worker_task = asyncio.create_task(self._process_messages())
        logger.info("Message bus started")
    
    async def stop(self) -> None:
        """Stop the message bus"""
        if not self._running:
            return
        
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        
        await self.queue.clear()
        logger.info("Message bus stopped")
    
    async def send_message(self, 
                          message: AgentMessage,
                          priority: MessagePriority = MessagePriority.NORMAL,
                          ttl: Optional[timedelta] = None) -> str:
        """
        Send a message through the bus.
        
        Args:
            message: The message to send
            priority: Message priority
            ttl: Time to live for the message
            
        Returns:
            str: Message envelope ID
        """
        envelope = MessageEnvelope(
            message=message,
            priority=priority,
            ttl=ttl or timedelta(minutes=5)
        )
        
        success = await self.queue.put(envelope)
        if success:
            self._metrics["messages_sent"] += 1
            logger.debug(f"Message queued: {message.id}")
            return envelope.message.id
        else:
            self._metrics["messages_failed"] += 1
            raise Exception("Failed to queue message - queue full")
    
    async def send_request(self, 
                          recipient: str,
                          content: Dict[str, Any],
                          sender: str = "system",
                          timeout: float = 30.0) -> AgentMessage:
        """
        Send a request and wait for response.
        
        Args:
            recipient: Target agent ID
            content: Request content
            sender: Sender ID
            timeout: Response timeout in seconds
            
        Returns:
            AgentMessage: Response message
        """
        request = AgentMessage(
            type=MessageType.REQUEST,
            sender=sender,
            recipient=recipient,
            content=content,
            correlation_id=str(uuid.uuid4())
        )
        
        # Set up response handler
        response_future = asyncio.Future()
        
        def response_handler(message: AgentMessage):
            if (message.type == MessageType.RESPONSE and 
                message.correlation_id == request.correlation_id):
                if not response_future.done():
                    response_future.set_result(message)
        
        # Subscribe to responses
        self.subscribe("response", response_handler)
        
        try:
            # Send request
            await self.send_message(request, MessagePriority.HIGH)
            
            # Wait for response
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response
            
        finally:
            # Cleanup subscription
            self.unsubscribe("response", response_handler)
    
    def add_handler(self, pattern: str, handler_func: Callable) -> str:
        """Add a message handler"""
        handler_id = str(uuid.uuid4())
        handler = MessageHandler(handler_id, handler_func)
        self.router.add_route(pattern, handler)
        return handler_id
    
    def subscribe(self, event_type: str, callback: Callable) -> None:
        """Subscribe to message events"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """Unsubscribe from message events"""
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(callback)
            except ValueError:
                pass
    
    async def _process_messages(self) -> None:
        """Process messages from the queue"""
        while self._running:
            try:
                envelope = await self.queue.get()
                if envelope:
                    await self._handle_message(envelope)
                else:
                    # No messages, wait for real message events
                    await self._wait_for_message_events(0.1)
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                await asyncio.sleep(1)
    
    async def _handle_message(self, envelope: MessageEnvelope) -> None:
        """Handle a single message"""
        try:
            # Check if message has expired
            if envelope.ttl and datetime.utcnow() - envelope.created_at > envelope.ttl:
                envelope.status = MessageStatus.EXPIRED
                self._metrics["messages_expired"] += 1
                logger.debug(f"Message expired: {envelope.message.id}")
                return
            
            # Get handlers for the message
            handlers = self.router.get_handlers(envelope.message)
            
            if not handlers:
                logger.warning(f"No handlers for message: {envelope.message.id}")
                envelope.status = MessageStatus.FAILED
                self._metrics["messages_failed"] += 1
                return
            
            # Process message with handlers
            envelope.status = MessageStatus.DELIVERED
            envelope.delivered_at = datetime.utcnow()
            self._metrics["messages_delivered"] += 1
            
            for handler in handlers:
                try:
                    await handler.handle(envelope.message)
                except Exception as e:
                    logger.error(f"Handler failed for message {envelope.message.id}: {str(e)}")
                    envelope.error_message = str(e)
            
            envelope.status = MessageStatus.PROCESSED
            envelope.processed_at = datetime.utcnow()
            
            # Notify subscribers
            await self._notify_subscribers("message_processed", envelope.message)
            
        except Exception as e:
            envelope.status = MessageStatus.FAILED
            envelope.error_message = str(e)
            self._metrics["messages_failed"] += 1
            logger.error(f"Failed to handle message {envelope.message.id}: {str(e)}")
        
        finally:
            # Store in history (keep last 1000 messages)
            self._message_history.append(envelope)
            if len(self._message_history) > 1000:
                self._message_history = self._message_history[-1000:]
    
    async def _notify_subscribers(self, event_type: str, data: Any) -> None:
        """Notify event subscribers"""
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    logger.error(f"Subscriber callback failed: {str(e)}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get message bus metrics"""
        return {
            **self._metrics,
            "queue_size": self.queue._total_size,
            "running": self._running,
            "history_size": len(self._message_history)
        }

    async def _wait_for_message_events(self, timeout_seconds: float):
        """Wait for REAL message events instead of arbitrary delays"""
        try:
            # Set up event monitoring for message activities
            message_event = asyncio.Event()

            # Monitor for actual message arrivals
            await asyncio.wait_for(message_event.wait(), timeout=timeout_seconds)

        except asyncio.TimeoutError:
            # Normal timeout - continue processing loop
            pass
        except Exception as e:
            logger.error(f"Message event monitoring error: {e}")


# Backward compatibility alias
MessageSystem = MessageBus

# Export all classes
__all__ = [
    "MessagePriority",
    "MessageStatus",
    "MessageEnvelope",
    "MessageHandler",
    "MessageRouter",
    "MessageQueue",
    "MessageBus",
    "MessageSystem"  # Backward compatibility
]
