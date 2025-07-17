"""
MCP Server Registry and Management - Backward Compatibility Layer

This module provides backward compatibility for the modular MCP server system.
All MCP server functionality has been moved to the mcp.server submodule
for better organization. This file maintains the original interface while
delegating to the new modular components.
"""

# Import all components from the modular MCP server system
from .server.config import (
    MCPServerConfig as ModularMCPServerConfig,
    MCPServerStatus as ModularMCPServerStatus,
    MCPServerType,
    MCPTransportType
)
from .server.registry import MCPServerRegistry as ModularMCPServerRegistry
from .server.manager import MCPServerManager as ModularMCPServerManager

# Legacy imports for backward compatibility
import asyncio
import logging
import subprocess
import json
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.types import Tool, Resource, Prompt
    MCP_AVAILABLE = True
except ImportError:
    # Fallback if mcp is not available
    ClientSession = Any
    StdioServerParameters = Any
    Tool = Any
    Resource = Any
    Prompt = Any
    MCP_AVAILABLE = False

# Use absolute import with fallback
try:
    from ..config.settings import Settings
except ImportError:
    # Fallback to absolute import
    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        from src.config.settings import Settings
    except ImportError:
        # Final fallback - create minimal Settings class
        class Settings:
            def __init__(self):
                pass
            def get(self, key, default=None):
                return default


logger = logging.getLogger(__name__)


class ServerStatus(Enum):
    """MCP Server status enumeration - Legacy compatibility"""
    INACTIVE = "inactive"
    STARTING = "starting"
    ACTIVE = "active"
    ERROR = "error"
    STOPPING = "stopping"

    @classmethod
    def from_modular_status(cls, modular_status: ModularMCPServerStatus) -> 'ServerStatus':
        """Convert from modular status to legacy status"""
        mapping = {
            ModularMCPServerStatus.STOPPED: cls.INACTIVE,
            ModularMCPServerStatus.STARTING: cls.STARTING,
            ModularMCPServerStatus.RUNNING: cls.ACTIVE,
            ModularMCPServerStatus.STOPPING: cls.STOPPING,
            ModularMCPServerStatus.ERROR: cls.ERROR,
            ModularMCPServerStatus.MAINTENANCE: cls.INACTIVE
        }
        return mapping.get(modular_status, cls.INACTIVE)


@dataclass
class MCPServerConfig:
    """MCP Server configuration - Legacy compatibility wrapper"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    command: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    transport: str = "stdio"
    config: Dict[str, Any] = field(default_factory=dict)
    auto_start: bool = True
    restart_on_failure: bool = True
    max_restarts: int = 3
    timeout: int = 30

    def to_modular_config(self) -> ModularMCPServerConfig:
        """Convert to modular config format"""
        # Map transport string to enum
        transport_mapping = {
            "stdio": MCPTransportType.STDIO,
            "http": MCPTransportType.HTTP,
            "websocket": MCPTransportType.WEBSOCKET,
            "tcp": MCPTransportType.TCP
        }

        return ModularMCPServerConfig(
            id=self.id,
            name=self.name,
            command=self.command,
            capabilities=self.capabilities,
            transport=transport_mapping.get(self.transport, MCPTransportType.STDIO),
            custom_config=self.config,
            auto_start=self.auto_start,
            restart_on_failure=self.restart_on_failure,
            max_restarts=self.max_restarts,
            timeout=self.timeout
        )

    @classmethod
    def from_modular_config(cls, modular_config: ModularMCPServerConfig) -> 'MCPServerConfig':
        """Create legacy config from modular config"""
        return cls(
            id=modular_config.id,
            name=modular_config.name,
            command=modular_config.command,
            capabilities=modular_config.capabilities,
            transport=modular_config.transport.value,
            config=modular_config.custom_config,
            auto_start=modular_config.auto_start,
            restart_on_failure=modular_config.restart_on_failure,
            max_restarts=modular_config.max_restarts,
            timeout=modular_config.timeout
        )


@dataclass
class MCPServerInstance:
    """Running MCP Server instance - Legacy compatibility wrapper"""
    config: MCPServerConfig
    status: ServerStatus = ServerStatus.INACTIVE
    session: Optional[ClientSession] = None
    process: Optional[subprocess.Popen] = None
    tools: Dict[str, Tool] = field(default_factory=dict)
    resources: Dict[str, Resource] = field(default_factory=dict)
    prompts: Dict[str, Prompt] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    last_error: Optional[str] = None
    restart_count: int = 0

    def is_running(self) -> bool:
        """Check if server is running"""
        return self.status == ServerStatus.ACTIVE and self.session is not None


class MCPServerManager:
    """
    MCP Server Manager - Legacy compatibility wrapper

    Provides backward compatibility while delegating to the modular MCP server manager.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.servers: Dict[str, MCPServerInstance] = {}
        self.tool_registry: Dict[str, str] = {}  # tool_name -> server_id
        self.resource_registry: Dict[str, str] = {}  # resource_name -> server_id
        self.prompt_registry: Dict[str, str] = {}  # prompt_name -> server_id
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False

        # Create modular manager
        self._modular_manager = ModularMCPServerManager(settings)
    
    async def start(self) -> None:
        """Start the MCP server manager"""
        if self._running:
            return

        self._running = True

        # Initialize modular manager
        await self._modular_manager.initialize()

        # Load default server configurations (legacy)
        await self._load_default_servers()

        # Start monitoring task (legacy)
        self._monitor_task = asyncio.create_task(self._monitor_servers())

        logger.info("MCP Server Manager started")

    async def stop(self) -> None:
        """Stop the MCP server manager"""
        if not self._running:
            return

        self._running = False

        # Stop monitoring task
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        # Shutdown modular manager
        await self._modular_manager.shutdown()

        # Clear legacy state
        self.servers.clear()
        self.tool_registry.clear()
        self.resource_registry.clear()
        self.prompt_registry.clear()

        logger.info("MCP Server Manager stopped")
    
    async def register_server(self, config: MCPServerConfig) -> str:
        """
        Register a new MCP server.

        Args:
            config: Server configuration

        Returns:
            str: Server ID
        """
        # Register with modular manager
        modular_config = config.to_modular_config()
        server_id = await self._modular_manager.register_server(modular_config)

        # Create legacy instance for backward compatibility
        server_instance = MCPServerInstance(config=config)
        self.servers[config.id] = server_instance

        logger.info(f"Registered MCP server: {config.name} ({config.id})")

        return config.id
    
    async def unregister_server(self, server_id: str) -> bool:
        """
        Unregister an MCP server.

        Args:
            server_id: Server ID

        Returns:
            bool: True if successful
        """
        # Delegate to modular manager
        success = await self._modular_manager.unregister_server(server_id)

        # Clean up legacy state
        if server_id in self.servers:
            server = self.servers[server_id]
            self._unregister_server_capabilities(server)
            del self.servers[server_id]

        logger.info(f"Unregistered MCP server: {server_id}")
        return success

    async def start_server(self, server_id: str) -> bool:
        """
        Start an MCP server.

        Args:
            server_id: Server ID

        Returns:
            bool: True if successful
        """
        # Delegate to modular manager
        success = await self._modular_manager.start_server(server_id)

        # Update legacy state
        if server_id in self.servers:
            server = self.servers[server_id]
            if success:
                server.status = ServerStatus.ACTIVE
                server.start_time = datetime.utcnow()
                server.last_error = None
            else:
                server.status = ServerStatus.ERROR
                server.last_error = "Failed to start server"

        return success
    
    async def stop_server(self, server_id: str) -> bool:
        """
        Stop an MCP server.

        Args:
            server_id: Server ID

        Returns:
            bool: True if successful
        """
        # Delegate to modular manager
        success = await self._modular_manager.stop_server(server_id)

        # Update legacy state
        if server_id in self.servers:
            server = self.servers[server_id]
            if success:
                server.status = ServerStatus.INACTIVE
                server.session = None
                server.process = None
                self._unregister_server_capabilities(server)
            else:
                server.status = ServerStatus.ERROR
                server.last_error = "Failed to stop server"

        return success

    async def restart_server(self, server_id: str) -> bool:
        """
        Restart an MCP server.

        Args:
            server_id: Server ID

        Returns:
            bool: True if successful
        """
        # Delegate to modular manager
        return await self._modular_manager.restart_server(server_id)
    
    async def get_server_status(self, server_id: str) -> Optional[Dict[str, Any]]:
        """Get server status information"""
        # Delegate to modular manager
        modular_status = await self._modular_manager.get_server_status(server_id)
        if not modular_status:
            return None

        # Convert to legacy format
        return {
            "id": server_id,
            "name": modular_status.get("config", {}).get("name", ""),
            "status": ServerStatus.from_modular_status(
                ModularMCPServerStatus(modular_status.get("status", "stopped"))
            ).value,
            "tools": [],  # Would need to query tool registry
            "resources": [],
            "prompts": [],
            "start_time": modular_status.get("registered_at"),
            "last_error": modular_status.get("last_error"),
            "restart_count": modular_status.get("restart_count", 0)
        }

    async def list_servers(self) -> List[Dict[str, Any]]:
        """List all registered servers"""
        # Delegate to modular manager
        modular_servers = await self._modular_manager.list_servers()

        # Convert to legacy format
        servers = []
        for server_info in modular_servers:
            legacy_status = {
                "id": server_info.get("config", {}).get("id", ""),
                "name": server_info.get("config", {}).get("name", ""),
                "status": ServerStatus.from_modular_status(
                    ModularMCPServerStatus(server_info.get("status", "stopped"))
                ).value,
                "tools": [],
                "resources": [],
                "prompts": [],
                "start_time": server_info.get("registered_at"),
                "last_error": server_info.get("last_error"),
                "restart_count": server_info.get("restart_count", 0)
            }
            servers.append(legacy_status)

        return servers

    async def find_tool_server(self, tool_name: str) -> Optional[str]:
        """Find which server provides a specific tool"""
        # Delegate to modular manager
        return await self._modular_manager.find_tool_server(tool_name)

    async def get_tool(self, tool_name: str) -> Optional[Tool]:
        """Get a tool by name"""
        # Delegate to modular manager
        tool_info = await self._modular_manager.get_tool(tool_name)
        if tool_info and MCP_AVAILABLE:
            # Convert to MCP Tool format if available
            # This is a simplified conversion
            return tool_info
        return None

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call an MCP tool.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        # Delegate to modular manager
        return await self._modular_manager.call_tool(tool_name, arguments)

    async def get_connected_servers_count(self) -> int:
        """Get count of connected servers"""
        # Delegate to modular manager
        return await self._modular_manager.get_connected_servers_count()
    
    async def _load_default_servers(self) -> None:
        """Load default MCP server configurations (legacy compatibility)"""
        # This is now handled by the modular manager during initialization
        # We keep this method for backward compatibility but it's mostly a no-op
        logger.info("Default server loading is now handled by modular manager")
    
    async def _discover_server_capabilities(self, server: MCPServerInstance) -> None:
        """Discover and register server capabilities (legacy compatibility)"""
        # This is now handled by the modular manager
        logger.debug(f"Capability discovery for {server.config.name} handled by modular manager")

    def _unregister_server_capabilities(self, server: MCPServerInstance) -> None:
        """Unregister server capabilities from registries (legacy compatibility)"""
        # Clear legacy state
        server.tools.clear()
        server.resources.clear()
        server.prompts.clear()

    async def _monitor_servers(self) -> None:
        """Monitor server health and restart failed servers (legacy compatibility)"""
        # The modular manager handles monitoring, but we keep this for compatibility
        while self._running:
            try:
                # Basic monitoring - the real work is done by modular manager
                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in legacy server monitoring: {str(e)}")
                await asyncio.sleep(30)


# Re-export modular components for direct access
MCPServerManager = MCPServerManager  # Legacy wrapper
ModularMCPServerManager = ModularMCPServerManager  # Direct access to modular manager

# Export all for backward compatibility
__all__ = [
    # Legacy classes
    "ServerStatus",
    "MCPServerConfig",
    "MCPServerInstance",
    "MCPServerManager",

    # Modular classes for direct access
    "ModularMCPServerConfig",
    "ModularMCPServerStatus",
    "ModularMCPServerRegistry",
    "ModularMCPServerManager",
    "MCPServerType",
    "MCPTransportType"
]
