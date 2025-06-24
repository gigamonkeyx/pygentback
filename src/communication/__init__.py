"""
Communication Module - Modular Architecture

This module provides modular communication infrastructure for PyGent Factory,
including protocol management, message routing, and external integrations.
The system has been modularized for better organization and maintainability
while maintaining full backward compatibility.

Modular Structure:
- protocols/: Protocol implementations and management
- middleware/: Communication middleware and interceptors
- serialization/: Message serialization and deserialization

Legacy compatibility is maintained through wrapper classes.
"""

# Import modular components
from .protocols import (
    ProtocolManager as ModularProtocolManager
)

# Import legacy compatibility layer
from .protocols import (
    ProtocolManager,
    ProtocolMessage, 
    ProtocolType,
    MessageFormat,
    CommunicationProtocol,
    InternalProtocol,
    MCPProtocol,
    HTTPProtocol, 
    WebSocketProtocol
)

# Export both modular and legacy interfaces
__all__ = [
    # Legacy compatibility (default imports)
    "ProtocolManager",
    "ProtocolMessage", 
    "ProtocolType",
    "MessageFormat",
    "CommunicationProtocol",
    "InternalProtocol",
    "MCPProtocol",
    "HTTPProtocol", 
    "WebSocketProtocol",
    
    # Modular components (for direct access)
    "ModularProtocolManager"
]
