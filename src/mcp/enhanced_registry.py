"""
Enhanced MCP Server Registry with Tool Discovery

This module extends the existing MCP server registry to implement proper tool discovery
according to the MCP specification. It ensures that after servers are registered and started,
their tools are discovered via tools/list and stored for agent access.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    # Use proper type stubs for when MCP is not available
    ClientSession = type(None)
    StdioServerParameters = type(None)
    Tool = type(None)
    TextContent = type(None)

from .server.registry import MCPServerRegistry, MCPServerRegistration
from .server.config import MCPServerConfig, MCPServerStatus


logger = logging.getLogger(__name__)


@dataclass
class MCPToolDefinition:
    """Represents a discovered MCP tool"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    annotations: Optional[Dict[str, Any]] = None
    server_id: str = ""
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "annotations": self.annotations,
            "server_id": self.server_id,
            "discovered_at": self.discovered_at.isoformat()
        }    @classmethod
    def from_mcp_tool(cls, tool: Any, server_id: str) -> 'MCPToolDefinition':
        """Create from MCP Tool object"""
        return cls(
            name=tool.name,
            description=tool.description or "",
            input_schema=tool.inputSchema or {},
            annotations=getattr(tool, 'annotations', None),
            server_id=server_id
        )


@dataclass
class MCPServerCapabilities:
    """Discovered capabilities of an MCP server"""
    server_id: str
    tools: List[MCPToolDefinition] = field(default_factory=list)
    resources: List[Dict[str, Any]] = field(default_factory=list)
    prompts: List[Dict[str, Any]] = field(default_factory=list)
    supports_tools: bool = False
    supports_resources: bool = False
    supports_prompts: bool = False
    list_changed_notifications: bool = False
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def update_tools(self, tools: List[MCPToolDefinition]) -> None:
        """Update the tools list"""
        self.tools = tools
        self.last_updated = datetime.utcnow()
    
    def get_tool_count(self) -> int:
        """Get number of available tools"""
        return len(self.tools)
    
    def get_tool_names(self) -> List[str]:
        """Get list of tool names"""
        return [tool.name for tool in self.tools]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "server_id": self.server_id,
            "tools": [tool.to_dict() for tool in self.tools],
            "resources": self.resources,
            "prompts": self.prompts,
            "supports_tools": self.supports_tools,
            "supports_resources": self.supports_resources,
            "supports_prompts": self.supports_prompts,
            "list_changed_notifications": self.list_changed_notifications,
            "discovered_at": self.discovered_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "tool_count": self.get_tool_count(),
            "tool_names": self.get_tool_names()
        }


class EnhancedMCPServerRegistry(MCPServerRegistry):
    """
    Enhanced MCP Server Registry with Tool Discovery    
    Extends the base registry to implement proper MCP tool discovery
    according to the specification. After servers are registered and started,
    this registry calls tools/list to discover available tools and stores
    the metadata for agent access.
    """
    
    def __init__(self):
        super().__init__()
        self.capabilities: Dict[str, MCPServerCapabilities] = {}
        self._manager = None  # Lazy initialization to avoid circular import
        self._tool_discovery_timeout = 30.0  # seconds
        self._active_sessions: Dict[str, Any] = {}
    
    @property
    def manager(self):
        """Lazy initialization of MCPServerManager to avoid circular import"""
        if self._manager is None:
            from .server.manager import MCPServerManager
            self._manager = MCPServerManager()
        return self._manager
    
    async def register_and_discover_server(self, config: MCPServerConfig) -> MCPServerRegistration:
        """
        Register server and discover its capabilities
        
        This is the main entry point that implements the complete flow:
        1. Register server
        2. Start server process
        3. Establish MCP session
        4. Discover tools/capabilities
        5. Store metadata
        """
        try:
            # Step 1: Register server
            logger.info(f"Registering MCP server: {config.name}")
            registration = await self.register_server(config)
            
            # Step 2: Start server process
            logger.info(f"Starting MCP server: {config.name}")
            await self.manager.start_server(config.id)
            
            # Step 3: Establish MCP session and discover capabilities
            logger.info(f"Discovering capabilities for server: {config.name}")
            await self._discover_server_capabilities(config.id)
            
            # Step 4: Update registration status
            registration.status = MCPServerStatus.RUNNING
            registration.start_count += 1
            registration.reset_restart_count()
            registration.update_heartbeat()
            
            logger.info(f"Successfully registered and discovered server: {config.name}")
            return registration
            
        except Exception as e:
            logger.error(f"Failed to register and discover server {config.name}: {e}")
            if config.id in self.servers:
                self.servers[config.id].set_error(str(e))
            raise
    
    async def _discover_server_capabilities(self, server_id: str) -> None:
        """
        Discover server capabilities by establishing MCP session and calling tools/list
        
        This implements the MCP specification requirement that clients MUST call
        tools/list to discover available tools from servers.
        """
        if not MCP_AVAILABLE:
            logger.warning("MCP SDK not available, skipping tool discovery")
            return
        
        registration = self.servers.get(server_id)
        if not registration:
            raise ValueError(f"Server {server_id} not found in registry")
        
        config = registration.config
        capabilities = MCPServerCapabilities(server_id=server_id)
        
        try:
            # Create MCP client session
            logger.debug(f"Establishing MCP session for server: {config.name}")
            session = await self._create_mcp_session(config)
            self._active_sessions[server_id] = session
            
            # Initialize the session
            await session.initialize()
            
            # Check server capabilities from initialization
            server_info = session.server_info
            if hasattr(server_info, 'capabilities'):
                caps = server_info.capabilities
                capabilities.supports_tools = hasattr(caps, 'tools') and caps.tools is not None
                capabilities.supports_resources = hasattr(caps, 'resources') and caps.resources is not None
                capabilities.supports_prompts = hasattr(caps, 'prompts') and caps.prompts is not None
                
                if capabilities.supports_tools and hasattr(caps.tools, 'listChanged'):
                    capabilities.list_changed_notifications = caps.tools.listChanged
            
            # Discover tools if supported
            if capabilities.supports_tools:
                logger.debug(f"Discovering tools for server: {config.name}")
                await self._discover_tools(session, capabilities)
            
            # Discover resources if supported (optional for now)
            if capabilities.supports_resources:
                logger.debug(f"Discovering resources for server: {config.name}")
                # REAL resource discovery implementation
                await self._discover_resources(session, capabilities)
            
            # Discover prompts if supported (optional for now)
            if capabilities.supports_prompts:
                logger.debug(f"Discovering prompts for server: {config.name}")
                # REAL prompt discovery implementation
                await self._discover_prompts(session, capabilities)
            
            # Store capabilities
            self.capabilities[server_id] = capabilities
            
            logger.info(f"Discovered {capabilities.get_tool_count()} tools for server: {config.name}")
            
        except Exception as e:
            logger.error(f"Failed to discover capabilities for server {config.name}: {e}")
            capabilities.last_updated = datetime.utcnow()
            self.capabilities[server_id] = capabilities
            raise
    
    async def _create_mcp_session(self, config: MCPServerConfig) -> Any:
        """Create MCP client session for the server"""
        if config.transport.value == "stdio":
            # Parse command for stdio transport
            if isinstance(config.command, str):
                command_parts = config.command.split()
            elif isinstance(config.command, list):
                command_parts = config.command
            else:
                raise ValueError(f"Invalid command format: {config.command}")
            
            server_params = StdioServerParameters(
                command=command_parts[0],
                args=command_parts[1:] if len(command_parts) > 1 else [],
                env=config.custom_config.get("env", {})
            )
            
            return await stdio_client(server_params)
        else:
            raise ValueError(f"Unsupported transport: {config.transport}")
    
    async def _discover_tools(self, session: Any, capabilities: MCPServerCapabilities) -> None:
        """
        Discover tools using the MCP tools/list endpoint
        
        This implements the core MCP specification requirement.
        """
        try:
            # Call tools/list according to MCP spec
            tools_result = await asyncio.wait_for(
                session.list_tools(),
                timeout=self._tool_discovery_timeout
            )
            
            discovered_tools = []
            for tool in tools_result.tools:
                tool_def = MCPToolDefinition.from_mcp_tool(tool, capabilities.server_id)
                discovered_tools.append(tool_def)
                logger.debug(f"Discovered tool: {tool.name} - {tool.description}")
            
            capabilities.update_tools(discovered_tools)
            logger.info(f"Successfully discovered {len(discovered_tools)} tools")
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout discovering tools for server {capabilities.server_id}")
            raise
        except Exception as e:
            logger.error(f"Failed to discover tools for server {capabilities.server_id}: {e}")
            raise
    
    async def get_all_available_tools(self) -> List[MCPToolDefinition]:
        """Get all available tools across all active servers"""
        all_tools = []
        for server_id, capabilities in self.capabilities.items():
            if self.is_server_active(server_id):
                all_tools.extend(capabilities.tools)
        return all_tools
    
    async def get_server_tools(self, server_id: str) -> List[MCPToolDefinition]:
        """Get tools for a specific server"""
        capabilities = self.capabilities.get(server_id)
        if capabilities and self.is_server_active(server_id):
            return capabilities.tools
        return []
    
    async def get_tool_by_name(self, tool_name: str) -> Optional[MCPToolDefinition]:
        """Find a tool by name across all servers"""
        for capabilities in self.capabilities.values():
            for tool in capabilities.tools:
                if tool.name == tool_name:
                    return tool
        return None
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool by name"""
        tool = await self.get_tool_by_name(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")
        
        session = self._active_sessions.get(tool.server_id)
        if not session:
            raise RuntimeError(f"No active session for server: {tool.server_id}")
        
        try:
            result = await session.call_tool(tool_name, arguments)
            return result
        except Exception as e:
            logger.error(f"Failed to call tool {tool_name}: {e}")
            raise
    
    def is_server_active(self, server_id: str) -> bool:
        """Check if server is active and has a session"""
        registration = self.servers.get(server_id)
        return (registration and 
                registration.status == MCPServerStatus.RUNNING and 
                server_id in self._active_sessions)
    
    async def get_capabilities_summary(self) -> Dict[str, Any]:
        """Get summary of all server capabilities"""
        summary = {
            "total_servers": len(self.servers),
            "active_servers": sum(1 for s in self.servers.values() if s.status == MCPServerStatus.RUNNING),
            "total_tools": sum(len(c.tools) for c in self.capabilities.values()),
            "servers": {}
        }
        
        for server_id, capabilities in self.capabilities.items():
            registration = self.servers.get(server_id)
            if registration:
                summary["servers"][server_id] = {
                    "name": registration.config.name,
                    "status": registration.status.value,
                    "tool_count": capabilities.get_tool_count(),                    "tool_names": capabilities.get_tool_names(),
                    "supports_tools": capabilities.supports_tools,
                    "last_updated": capabilities.last_updated.isoformat()
                }
        
        return summary
    
    async def refresh_server_tools(self, server_id: str) -> None:
        """Refresh tools for a specific server"""
        if server_id not in self._active_sessions:
            raise ValueError(f"No active session for server: {server_id}")
        
        capabilities = self.capabilities.get(server_id)
        if not capabilities:
            raise ValueError(f"No capabilities found for server: {server_id}")
        
        session = self._active_sessions[server_id]
        await self._discover_tools(session, capabilities)
    
    async def shutdown(self) -> None:
        """Shutdown registry and close all sessions"""
        logger.info("Shutting down enhanced MCP server registry")
        
        # Close all active sessions
        for server_id, session in self._active_sessions.items():
            try:
                await session.close()
            except Exception as e:
                logger.error(f"Error closing session for server {server_id}: {e}")
        
        self._active_sessions.clear()
        
        # Call parent shutdown if it exists
        if hasattr(super(), 'shutdown'):
            await super().shutdown()


    async def _discover_resources(self, session, capabilities):
        """REAL resource discovery implementation"""
        try:
            # Query server for available resources
            resources_response = await session.list_resources()

            if resources_response and hasattr(resources_response, 'resources'):
                for resource in resources_response.resources:
                    # Store resource information
                    resource_info = {
                        'uri': resource.uri,
                        'name': resource.name,
                        'description': getattr(resource, 'description', ''),
                        'mime_type': getattr(resource, 'mimeType', ''),
                        'annotations': getattr(resource, 'annotations', {})
                    }
                    capabilities.add_resource(resource_info)

                logger.info(f"Discovered {len(resources_response.resources)} resources")
            else:
                logger.debug("No resources discovered from server")

        except Exception as e:
            logger.warning(f"Resource discovery failed: {e}")

    async def _discover_prompts(self, session, capabilities):
        """REAL prompt discovery implementation"""
        try:
            # Query server for available prompts
            prompts_response = await session.list_prompts()

            if prompts_response and hasattr(prompts_response, 'prompts'):
                for prompt in prompts_response.prompts:
                    # Store prompt information
                    prompt_info = {
                        'name': prompt.name,
                        'description': getattr(prompt, 'description', ''),
                        'arguments': getattr(prompt, 'arguments', [])
                    }
                    capabilities.add_prompt(prompt_info)

                logger.info(f"Discovered {len(prompts_response.prompts)} prompts")
            else:
                logger.debug("No prompts discovered from server")

        except Exception as e:
            logger.warning(f"Prompt discovery failed: {e}")


# Global registry instance
enhanced_registry = EnhancedMCPServerRegistry()
