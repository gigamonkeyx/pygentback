"""
MCP Server Module

This module provides modular MCP server management functionality,
including server registration, lifecycle management, and configuration.
"""

# Import all MCP server components
from .registry import MCPServerRegistry
from .manager import MCPServerManager
from .config import MCPServerConfig, MCPServerType
from .lifecycle import MCPServerLifecycle

# Re-export for backward compatibility
__all__ = [
    # Core components
    "MCPServerRegistry",
    "MCPServerManager", 
    "MCPServerConfig",
    "MCPServerType",
    "MCPServerLifecycle"
]
