"""
Communication Protocols Module

This module provides modular communication protocol implementations,
including MCP, HTTP, WebSocket, and internal messaging protocols.
"""

# Import all protocol components
from .base import (
    CommunicationProtocol, ProtocolMessage, ProtocolType, MessageFormat,
    ProtocolError, ProtocolTimeoutError, ProtocolConnectionError
)
from .internal import InternalProtocol
from .mcp import MCPProtocol
from .http import HTTPProtocol
from .websocket import WebSocketProtocol
from .manager import ProtocolManager

# Re-export for easy importing
__all__ = [
    # Core interfaces and types
    "CommunicationProtocol",
    "ProtocolMessage", 
    "ProtocolType",
    "MessageFormat",
    "ProtocolError",
    "ProtocolTimeoutError",
    "ProtocolConnectionError",
    
    # Protocol implementations
    "InternalProtocol",
    "MCPProtocol",
    "HTTPProtocol",
    "WebSocketProtocol",
    
    # Management
    "ProtocolManager"
]
