"""
MCP (Model Context Protocol) Integration - Modular System

This module provides modular MCP server management and tool integration capabilities
for PyGent Factory. The system has been modularized for better organization and
maintainability while maintaining full backward compatibility.

Modular Structure:
- server/: Server management, registry, lifecycle, and configuration
- tools/: Tool registry, discovery, and execution
- client/: MCP client implementations and communication

Legacy compatibility is maintained through wrapper classes.
"""

# Import modular components
from .server import (
    MCPServerManager as ModularMCPServerManager,
    MCPServerConfig as ModularMCPServerConfig,
    MCPServerRegistry,
    MCPServerLifecycle
)

from .tools import (
    MCPToolRegistry
)

# Import legacy compatibility layer
from .server_registry import (
    MCPServerManager,
    MCPServerConfig,
    ServerStatus,
    MCPServerInstance
)

# Export both modular and legacy interfaces
__all__ = [
    # Legacy compatibility (default imports)
    "MCPServerManager",
    "MCPServerConfig", 
    "ServerStatus",
    "MCPServerInstance",
    
    # Modular components (for direct access)
    "ModularMCPServerManager",
    "ModularMCPServerConfig",
    "MCPServerRegistry",
    "MCPServerLifecycle",
    
    # Tool management
    "MCPToolRegistry"
]
