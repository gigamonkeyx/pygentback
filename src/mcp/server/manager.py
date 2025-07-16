"""
MCP Server Manager

This module provides the main management interface for MCP servers,
coordinating between registry, lifecycle, and configuration components.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from .config import MCPServerConfig, MCPServerStatus, MCPServerType
from .registry import MCPServerRegistration
from .lifecycle import MCPServerLifecycle


logger = logging.getLogger(__name__)


class MCPServerManager:
    """
    Main manager for MCP servers.
    
    Coordinates between server registry, lifecycle management, and provides
    a unified interface for all MCP server operations.
    """
    
    def __init__(self, settings=None):
        """
        Initialize the MCP server manager.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        
        # Import here to avoid circular import
        from ..enhanced_registry import EnhancedMCPServerRegistry
        
        # Core components
        self.registry = EnhancedMCPServerRegistry()
        self.lifecycle = MCPServerLifecycle()
        
        # Configuration
        self.auto_start_servers = getattr(settings, 'mcp_auto_start', True) if settings else True
        self.restart_failed_servers = getattr(settings, 'mcp_restart_failed', True) if settings else True
        
        # State
        self._initialized = False
        self._auto_restart_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Initialize the MCP server manager"""
        try:
            # Start registry
            await self.registry.start()
            
            # Start lifecycle monitoring
            await self.lifecycle.start_monitoring()
            
            # Start auto-restart monitoring if enabled
            if self.restart_failed_servers:
                self._auto_restart_task = asyncio.create_task(self._auto_restart_monitor())
            
            # Load and start configured servers
            if self.settings:
                await self._load_configured_servers()
            
            self._initialized = True
            logger.info("MCP server manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP server manager: {str(e)}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the MCP server manager"""
        try:
            # Stop auto-restart monitoring
            if self._auto_restart_task:
                self._auto_restart_task.cancel()
                try:
                    await self._auto_restart_task
                except asyncio.CancelledError:
                    pass
            
            # Stop lifecycle monitoring (this will stop all servers)
            await self.lifecycle.stop_monitoring()
            
            # Stop registry
            await self.registry.stop()
            
            self._initialized = False
            logger.info("MCP server manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during MCP server manager shutdown: {str(e)}")
    
    async def register_server(self, config: MCPServerConfig) -> str:
        """
        Register a new MCP server.
        
        Args:
            config: Server configuration
            
        Returns:
            str: Server ID
        """
        try:
            # Register in registry
            success = await self.registry.register_server(config)
            if not success:
                raise Exception("Failed to register server in registry")
            
            # Auto-start if enabled
            if config.auto_start and self.auto_start_servers:
                await self.start_server(config.id)
            
            logger.info(f"Registered MCP server: {config.name} ({config.id})")
            return config.id
            
        except Exception as e:
            logger.error(f"Failed to register MCP server: {str(e)}")
            raise
    
    async def unregister_server(self, server_id: str) -> bool:
        """
        Unregister an MCP server.
        
        Args:
            server_id: ID of the server to unregister
            
        Returns:
            bool: True if successful
        """
        try:
            # Stop server if running
            if self.lifecycle.is_server_running(server_id):
                await self.stop_server(server_id)
            
            # Unregister from registry
            success = await self.registry.unregister_server(server_id)
            
            if success:
                logger.info(f"Unregistered MCP server: {server_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to unregister MCP server {server_id}: {str(e)}")
            return False
    
    async def start_server(self, server_id: str) -> bool:
        """
        Start an MCP server.
        
        Args:
            server_id: ID of the server to start
            
        Returns:
            bool: True if successful
        """
        try:
            # Get registration
            registration = await self.registry.get_server(server_id)
            if not registration:
                raise Exception(f"Server {server_id} not found")
            
            # Update status to starting
            await self.registry.update_server_status(server_id, MCPServerStatus.STARTING)
            
            # Start server
            success = await self.lifecycle.start_server(registration)
            
            # Update status based on result
            if success:
                await self.registry.update_server_status(server_id, MCPServerStatus.RUNNING)
                logger.info(f"Started MCP server: {registration.config.name}")
            else:
                await self.registry.update_server_status(
                    server_id, MCPServerStatus.ERROR, "Failed to start"
                )
            
            return success
            
        except Exception as e:
            error_msg = f"Failed to start MCP server {server_id}: {str(e)}"
            logger.error(error_msg)
            await self.registry.update_server_status(server_id, MCPServerStatus.ERROR, error_msg)
            return False
    
    async def stop_server(self, server_id: str) -> bool:
        """
        Stop an MCP server.
        
        Args:
            server_id: ID of the server to stop
            
        Returns:
            bool: True if successful
        """
        try:
            # Get registration
            registration = await self.registry.get_server(server_id)
            if not registration:
                raise Exception(f"Server {server_id} not found")
            
            # Update status to stopping
            await self.registry.update_server_status(server_id, MCPServerStatus.STOPPING)
            
            # Stop server
            success = await self.lifecycle.stop_server(registration)
            
            # Update status based on result
            if success:
                await self.registry.update_server_status(server_id, MCPServerStatus.STOPPED)
                logger.info(f"Stopped MCP server: {registration.config.name}")
            else:
                await self.registry.update_server_status(
                    server_id, MCPServerStatus.ERROR, "Failed to stop"
                )
            
            return success
            
        except Exception as e:
            error_msg = f"Failed to stop MCP server {server_id}: {str(e)}"
            logger.error(error_msg)
            await self.registry.update_server_status(server_id, MCPServerStatus.ERROR, error_msg)
            return False
    
    async def restart_server(self, server_id: str) -> bool:
        """
        Restart an MCP server.
        
        Args:
            server_id: ID of the server to restart
            
        Returns:
            bool: True if successful
        """
        try:
            # Get registration
            registration = await self.registry.get_server(server_id)
            if not registration:
                raise Exception(f"Server {server_id} not found")
            
            # Check if restart is allowed
            if not registration.can_restart():
                error_msg = f"Server {server_id} has exceeded maximum restart attempts"
                logger.warning(error_msg)
                await self.registry.update_server_status(server_id, MCPServerStatus.ERROR, error_msg)
                return False
            
            # Update status to starting
            await self.registry.update_server_status(server_id, MCPServerStatus.STARTING)
            
            # Restart server
            success = await self.lifecycle.restart_server(registration)
            
            # Update status based on result
            if success:
                await self.registry.update_server_status(server_id, MCPServerStatus.RUNNING)
                logger.info(f"Restarted MCP server: {registration.config.name}")
            else:
                await self.registry.update_server_status(
                    server_id, MCPServerStatus.ERROR, "Failed to restart"
                )
            
            return success
            
        except Exception as e:
            error_msg = f"Failed to restart MCP server {server_id}: {str(e)}"
            logger.error(error_msg)
            await self.registry.update_server_status(server_id, MCPServerStatus.ERROR, error_msg)
            return False
    
    async def get_server_status(self, server_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed server status.
        
        Args:
            server_id: ID of the server
            
        Returns:
            Dict with server status or None if not found
        """
        registration = await self.registry.get_server(server_id)
        if not registration:
            return None
        
        # Get process information
        process = self.lifecycle.get_server_process(server_id)
        process_info = {}
        if process:
            process_info = {
                "pid": process.get_pid(),
                "uptime": str(process.get_uptime()) if process.get_uptime() else None,
                "is_running": process.is_running()
            }
        
        return {
            **registration.to_dict(),
            "process": process_info
        }
    
    async def list_servers(self, server_type: Optional[MCPServerType] = None,
                          status: Optional[MCPServerStatus] = None) -> List[Dict[str, Any]]:
        """
        List servers with optional filtering.
        
        Args:
            server_type: Filter by server type
            status: Filter by status
            
        Returns:
            List of server information
        """
        registrations = await self.registry.list_servers(server_type, status)
        
        servers = []
        for registration in registrations:
            server_info = registration.to_dict()
            
            # Add process information
            process = self.lifecycle.get_server_process(registration.config.id)
            if process:
                server_info["process"] = {
                    "pid": process.get_pid(),
                    "uptime": str(process.get_uptime()) if process.get_uptime() else None,
                    "is_running": process.is_running()
                }
            
            servers.append(server_info)
        
        return servers
    
    async def get_servers_by_capability(self, capability: str) -> List[Dict[str, Any]]:
        """Get servers that have a specific capability"""
        registrations = await self.registry.find_servers_by_capability(capability)
        return [reg.to_dict() for reg in registrations]
    
    async def get_servers_by_tool(self, tool_name: str) -> List[Dict[str, Any]]:
        """Get servers that provide a specific tool"""
        registrations = await self.registry.find_servers_by_tool(tool_name)
        return [reg.to_dict() for reg in registrations]
    
    async def get_connected_servers_count(self) -> int:
        """Get count of connected/running servers"""
        running_servers = await self.registry.get_running_servers()
        return len(running_servers)
    
    async def find_tool_server(self, tool_name: str) -> Optional[str]:
        """Find a server that provides a specific tool"""
        registrations = await self.registry.find_servers_by_tool(tool_name)
        
        # Return the first running server that has the tool
        for registration in registrations:
            if registration.status == MCPServerStatus.RUNNING:
                return registration.config.id
        
        # If no running server found, return the first available
        if registrations:
            return registrations[0].config.id
        
        return None
    
    async def get_tool(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get tool information"""
        server_id = await self.find_tool_server(tool_name)
        if not server_id:
            return None
        
        registration = await self.registry.get_server(server_id)
        if not registration:
            return None
        
        # Return basic tool information
        # In a full implementation, this would query the actual server for tool details
        return {
            "name": tool_name,
            "server_id": server_id,
            "server_name": registration.config.name,
            "description": f"Tool provided by {registration.config.name}",
            "available": registration.status == MCPServerStatus.RUNNING
        }
    
    async def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get tool information - alias for get_tool()"""
        return await self.get_tool(tool_name)

    async def is_server_available(self, server_name: str) -> bool:
        """Check if a server is available and running"""
        try:
            # Try to find server by name
            registrations = await self.registry.list_servers()
            for reg in registrations:
                if reg.config.name == server_name or reg.config.id == server_name:
                    return reg.status == MCPServerStatus.RUNNING
            return False
        except Exception:
            return False

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool on an MCP server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        # Find server that provides the tool
        server_id = await self.find_tool_server(tool_name)
        if not server_id:
            raise Exception(f"No server found for tool: {tool_name}")
        
        registration = await self.registry.get_server(server_id)
        if not registration:
            raise Exception(f"Server {server_id} not found")
        
        if registration.status != MCPServerStatus.RUNNING:
            raise Exception(f"Server {registration.config.name} is not running")
        
        # Call tool via MCP protocol
        try:
            if not registration.client_session:
                raise Exception(f"No client session available for server {registration.config.name}")
            
            # Make the MCP tool call
            logger.info(f"Calling tool {tool_name} on server {registration.config.name}")
            
            # Use the MCP client session to call the tool
            response = await registration.client_session.call_tool(tool_name, arguments)
            
            return {
                "tool": tool_name,
                "arguments": arguments,
                "response": response,
                "server": registration.config.name,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Tool call failed for {tool_name} on {registration.config.name}: {e}")
            return {
                "tool": tool_name,
                "arguments": arguments,
                "error": str(e),
                "server": registration.config.name,
                "status": "failed"
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the MCP system"""
        registry_health = await self.registry.health_check()
        lifecycle_stats = self.lifecycle.get_lifecycle_stats()
        
        return {
            "manager_initialized": self._initialized,
            "registry": registry_health,
            "lifecycle": lifecycle_stats,
            "auto_restart_enabled": self.restart_failed_servers
        }
    
    async def _load_configured_servers(self) -> None:
        """Load servers from configuration"""
        if not self.settings:
            return
        
        # Get MCP configuration
        mcp_config = getattr(self.settings, 'mcp', {})
        servers_config = getattr(mcp_config, 'servers', [])
        
        for server_config in servers_config:
            try:
                config = MCPServerConfig.from_dict(server_config)
                await self.register_server(config)
            except Exception as e:
                logger.error(f"Failed to load configured server: {str(e)}")
    
    async def _auto_restart_monitor(self) -> None:
        """Monitor for failed servers and restart them automatically"""
        while True:
            try:
                # Get failed servers
                failed_servers = await self.registry.get_failed_servers()
                
                for registration in failed_servers:
                    if registration.can_restart():
                        logger.info(f"Auto-restarting failed server: {registration.config.name}")
                        await self.restart_server(registration.config.id)
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-restart monitor: {str(e)}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    def is_initialized(self) -> bool:
        """Check if the manager is initialized"""
        return self._initialized
