"""
WebSocket Communication Protocol

Provides WebSocket protocol implementation for PyGent Factory.
"""

import asyncio
import websockets
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .base import CommunicationProtocol, ProtocolMessage, ProtocolType, MessageFormat

logger = logging.getLogger(__name__)


@dataclass
class WebSocketMessage(ProtocolMessage):
    """WebSocket protocol message"""
    event_type: str = ""
    
    def __post_init__(self):
        super().__post_init__()


class WebSocketProtocol(CommunicationProtocol):
    """
    WebSocket communication protocol.
    
    Provides real-time bidirectional communication for PyGent Factory.
    """
    
    def __init__(self):
        super().__init__(ProtocolType.WEBSOCKET, MessageFormat.JSON)
        self.websocket = None
        self.is_server = False
        
    async def initialize(self) -> bool:
        """Initialize the WebSocket protocol"""
        try:
            logger.info("WebSocket protocol initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket protocol: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Cleanup the WebSocket protocol"""
        if self.websocket:
            await self.websocket.close()
        logger.info("WebSocket protocol cleaned up")
    
    async def send_message(self, message: ProtocolMessage) -> bool:
        """Send a WebSocket message"""
        try:
            if not self.websocket:
                return False
                
            ws_data = {
                "id": message.id,
                "sender": message.sender,
                "recipient": message.recipient,
                "content": message.content,
                "timestamp": message.timestamp.isoformat()
            }
            
            await self.websocket.send(json.dumps(ws_data))
            logger.debug(f"Sent WebSocket message: {message.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
            return False
    
    async def receive_message(self, timeout: Optional[float] = None) -> Optional[ProtocolMessage]:
        """Receive a WebSocket message"""
        try:
            if not self.websocket:
                return None
                
            if timeout:
                data = await asyncio.wait_for(
                    self.websocket.recv(),
                    timeout=timeout
                )
            else:
                data = await self.websocket.recv()
            
            message_data = json.loads(data)
            message = WebSocketMessage(
                id=message_data.get("id", ""),
                sender=message_data.get("sender", ""),
                recipient=message_data.get("recipient", ""),
                content=message_data.get("content", {})
            )
            
            logger.debug(f"Received WebSocket message: {message.id}")
            return message
            
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Failed to receive WebSocket message: {e}")
            return None
    
    def is_connected(self) -> bool:
        """Check if the protocol is connected"""
        return self.websocket is not None and not self.websocket.closed
    
    def get_status(self) -> Dict[str, Any]:
        """Get protocol status"""
        return {
            "type": self.protocol_type.value,
            "format": self.message_format.value,
            "connected": self.is_connected(),
            "is_server": self.is_server
        }
