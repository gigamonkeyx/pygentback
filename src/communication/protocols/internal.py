"""
Internal Communication Protocol

Provides internal messaging protocol for PyGent Factory components.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

from .base import CommunicationProtocol, ProtocolMessage, ProtocolType, MessageFormat

logger = logging.getLogger(__name__)


@dataclass
class InternalMessage(ProtocolMessage):
    """Internal protocol message"""
    component: str = ""
    action: str = ""
    data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}
        super().__post_init__()


class InternalProtocol(CommunicationProtocol):
    """
    Internal communication protocol for PyGent Factory components.
    
    Provides fast, in-memory messaging between system components.
    """
    
    def __init__(self):
        super().__init__(ProtocolType.INTERNAL, MessageFormat.JSON)
        self.handlers: Dict[str, Callable] = {}
        self.message_queue = asyncio.Queue()
        self.is_running = False
        
    async def initialize(self) -> bool:
        """Initialize the internal protocol"""
        try:
            self.is_running = True
            logger.info("Internal protocol initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize internal protocol: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Cleanup the internal protocol"""
        self.is_running = False
        logger.info("Internal protocol cleaned up")
    
    async def send_message(self, message: ProtocolMessage) -> bool:
        """Send an internal message"""
        try:
            if not isinstance(message, InternalMessage):
                # Convert to internal message
                internal_msg = InternalMessage(
                    id=message.id,
                    sender=message.sender,
                    recipient=message.recipient,
                    content=message.content,
                    timestamp=message.timestamp
                )
            else:
                internal_msg = message
            
            await self.message_queue.put(internal_msg)
            logger.debug(f"Sent internal message: {internal_msg.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send internal message: {e}")
            return False
    
    async def receive_message(self, timeout: Optional[float] = None) -> Optional[ProtocolMessage]:
        """Receive an internal message"""
        try:
            if timeout:
                message = await asyncio.wait_for(
                    self.message_queue.get(), 
                    timeout=timeout
                )
            else:
                message = await self.message_queue.get()
            
            logger.debug(f"Received internal message: {message.id}")
            return message
            
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Failed to receive internal message: {e}")
            return None
    
    def register_handler(self, action: str, handler: Callable) -> None:
        """Register a message handler for a specific action"""
        self.handlers[action] = handler
        logger.debug(f"Registered handler for action: {action}")
    
    async def handle_message(self, message: InternalMessage) -> Optional[Any]:
        """Handle an internal message using registered handlers"""
        try:
            if message.action in self.handlers:
                handler = self.handlers[message.action]
                result = await handler(message)
                logger.debug(f"Handled message {message.id} with action {message.action}")
                return result
            else:
                logger.warning(f"No handler found for action: {message.action}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to handle message {message.id}: {e}")
            return None
    
    async def broadcast_message(self, message: InternalMessage) -> bool:
        """Broadcast a message to all components"""
        try:
            # For now, just send to queue
            # In a full implementation, this would send to all registered components
            await self.send_message(message)
            logger.debug(f"Broadcasted message: {message.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to broadcast message: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if the protocol is connected"""
        return self.is_running
    
    def get_status(self) -> Dict[str, Any]:
        """Get protocol status"""
        return {
            "type": self.protocol_type.value,
            "format": self.message_format.value,
            "connected": self.is_connected(),
            "queue_size": self.message_queue.qsize(),
            "handlers": list(self.handlers.keys())
        }
