"""
HTTP Communication Protocol

Provides HTTP protocol implementation for PyGent Factory.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .base import CommunicationProtocol, ProtocolMessage, ProtocolType, MessageFormat

logger = logging.getLogger(__name__)


@dataclass
class HTTPMessage(ProtocolMessage):
    """HTTP protocol message"""
    method: str = "GET"
    url: str = ""
    headers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}
        super().__post_init__()


class HTTPProtocol(CommunicationProtocol):
    """
    HTTP communication protocol.
    
    Provides HTTP client/server functionality for PyGent Factory.
    """
    
    def __init__(self):
        super().__init__(ProtocolType.HTTP, MessageFormat.JSON)
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self) -> bool:
        """Initialize the HTTP protocol"""
        try:
            self.session = aiohttp.ClientSession()
            logger.info("HTTP protocol initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize HTTP protocol: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Cleanup the HTTP protocol"""
        if self.session:
            await self.session.close()
        logger.info("HTTP protocol cleaned up")
    
    async def send_message(self, message: ProtocolMessage) -> bool:
        """Send an HTTP message"""
        try:
            if not self.session:
                return False
                
            http_msg = message if isinstance(message, HTTPMessage) else HTTPMessage(
                id=message.id,
                sender=message.sender,
                recipient=message.recipient,
                content=message.content
            )
            
            async with self.session.request(
                http_msg.method,
                http_msg.url,
                json=http_msg.content,
                headers=http_msg.headers
            ) as response:
                logger.debug(f"Sent HTTP message: {message.id}, status: {response.status}")
                return response.status < 400
                
        except Exception as e:
            logger.error(f"Failed to send HTTP message: {e}")
            return False
    
    async def receive_message(self, timeout: Optional[float] = None) -> Optional[ProtocolMessage]:
        """Receive an HTTP message"""
        try:
            # Placeholder - HTTP is typically request/response
            logger.debug("HTTP receive not implemented for client mode")
            return None
            
        except Exception as e:
            logger.error(f"Failed to receive HTTP message: {e}")
            return None
    
    def is_connected(self) -> bool:
        """Check if the protocol is connected"""
        return self.session is not None and not self.session.closed
    
    def get_status(self) -> Dict[str, Any]:
        """Get protocol status"""
        return {
            "type": self.protocol_type.value,
            "format": self.message_format.value,
            "connected": self.is_connected()
        }
