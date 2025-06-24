"""
MCP Server Registry

This module provides registration and discovery functionality for MCP servers.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from .config import MCPServerConfig, MCPServerStatus, MCPServerType


logger = logging.getLogger(__name__)


@dataclass
class MCPServerRegistration:
    """Represents an MCP server registration in the registry"""
    config: MCPServerConfig
    status: MCPServerStatus = MCPServerStatus.STOPPED
    process: Optional[Any] = None  # Process handle
    registered_at: datetime = field(default_factory=datetime.utcnow)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    start_count: int = 0
    restart_count: int = 0
    last_error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_heartbeat(self) -> None:
        """Update the last heartbeat timestamp"""
        self.last_heartbeat = datetime.utcnow()
    
    def is_stale(self, timeout_seconds: int = 60) -> bool:
        """Check if registration is stale (no recent heartbeat)"""
        if self.status == MCPServerStatus.STOPPED:
            return False
        
        elapsed = (datetime.utcnow() - self.last_heartbeat).total_seconds()
        return elapsed > timeout_seconds
    
    def can_restart(self) -> bool:
        """Check if server can be restarted"""
        return (self.config.restart_on_failure and 
                self.restart_count < self.config.max_restarts)
    
    def increment_restart_count(self) -> None:
        """Increment restart count"""
        self.restart_count += 1
    
    def reset_restart_count(self) -> None:
        """Reset restart count"""
        self.restart_count = 0
    
    def set_error(self, error_message: str) -> None:
        """Set error status and message"""
        self.status = MCPServerStatus.ERROR
        self.last_error = error_message
        self.update_heartbeat()
    
    def clear_error(self) -> None:
        """Clear error status"""
        self.last_error = None
        if self.status == MCPServerStatus.ERROR:
            self.status = MCPServerStatus.STOPPED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "config": self.config.to_dict(),
            "status": self.status.value,
            "registered_at": self.registered_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "start_count": self.start_count,
            "restart_count": self.restart_count,
            "last_error": self.last_error,
            "metadata": self.metadata,
            "can_restart": self.can_restart(),
            "is_stale": self.is_stale()
        }


class MCPServerRegistry:
    """
    Registry for managing MCP server instances.
    
    Provides centralized registration, discovery, and lifecycle
    management for all MCP servers in the system.
    """
    
    def __init__(self):
        self.servers: Dict[str, MCPServerRegistration] = {}
        self.server_types: Dict[MCPServerType, List[str]] = {}
        self._lock = asyncio.Lock()
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self) -> None:
        """Start the MCP server registry"""
        if self._running:
            return
        
        self._running = True
        
        # Start heartbeat monitoring
        self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
        
        logger.info("MCP server registry started")
    
    async def stop(self) -> None:
        """Stop the MCP server registry"""
        if not self._running:
            return
        
        self._running = False
        
        # Stop heartbeat monitoring
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        logger.info("MCP server registry stopped")
    
    async def register_server(self, config: MCPServerConfig) -> bool:
        """
        Register an MCP server in the registry.
        
        Args:
            config: Server configuration
            
        Returns:
            bool: True if registration successful
        """
        async with self._lock:
            try:
                if config.id in self.servers:
                    logger.warning(f"MCP server {config.id} already registered")
                    return False
                
                registration = MCPServerRegistration(config=config)
                self.servers[config.id] = registration
                
                # Update type index
                if config.server_type not in self.server_types:
                    self.server_types[config.server_type] = []
                self.server_types[config.server_type].append(config.id)
                
                logger.info(f"Registered MCP server: {config.name} ({config.id})")
                return True
                
            except Exception as e:
                logger.error(f"Failed to register MCP server {config.id}: {str(e)}")
                return False
    
    async def unregister_server(self, server_id: str) -> bool:
        """
        Unregister an MCP server from the registry.
        
        Args:
            server_id: ID of the server to unregister
            
        Returns:
            bool: True if unregistration successful
        """
        async with self._lock:
            try:
                if server_id not in self.servers:
                    logger.warning(f"MCP server {server_id} not found in registry")
                    return False
                
                registration = self.servers[server_id]
                
                # Remove from type index
                server_type = registration.config.server_type
                if server_type in self.server_types:
                    if server_id in self.server_types[server_type]:
                        self.server_types[server_type].remove(server_id)
                    if not self.server_types[server_type]:
                        del self.server_types[server_type]
                
                del self.servers[server_id]
                
                logger.info(f"Unregistered MCP server: {registration.config.name} ({server_id})")
                return True
                
            except Exception as e:
                logger.error(f"Failed to unregister MCP server {server_id}: {str(e)}")
                return False
    
    async def get_server(self, server_id: str) -> Optional[MCPServerRegistration]:
        """
        Get a server registration by ID.
        
        Args:
            server_id: ID of the server
            
        Returns:
            MCPServerRegistration or None if not found
        """
        registration = self.servers.get(server_id)
        if registration:
            registration.update_heartbeat()
        return registration
    
    async def list_servers(self, server_type: Optional[MCPServerType] = None, 
                          status: Optional[MCPServerStatus] = None) -> List[MCPServerRegistration]:
        """
        List servers with optional filtering.
        
        Args:
            server_type: Filter by server type
            status: Filter by server status
            
        Returns:
            List[MCPServerRegistration]: List of matching servers
        """
        servers = []
        
        for registration in self.servers.values():
            # Filter by type
            if server_type and registration.config.server_type != server_type:
                continue
            
            # Filter by status
            if status and registration.status != status:
                continue
            
            servers.append(registration)
            registration.update_heartbeat()
        
        return servers
    
    async def get_servers_by_type(self, server_type: MCPServerType) -> List[MCPServerRegistration]:
        """Get all servers of a specific type"""
        if server_type not in self.server_types:
            return []
        
        servers = []
        for server_id in self.server_types[server_type]:
            if server_id in self.servers:
                registration = self.servers[server_id]
                servers.append(registration)
                registration.update_heartbeat()
        
        return servers
    
    async def get_running_servers(self) -> List[MCPServerRegistration]:
        """Get all running servers"""
        return await self.list_servers(status=MCPServerStatus.RUNNING)
    
    async def get_failed_servers(self) -> List[MCPServerRegistration]:
        """Get all failed servers"""
        return await self.list_servers(status=MCPServerStatus.ERROR)
    
    async def find_servers_by_capability(self, capability: str) -> List[MCPServerRegistration]:
        """
        Find servers that have a specific capability.
        
        Args:
            capability: Name of the capability
            
        Returns:
            List[MCPServerRegistration]: Servers with the capability
        """
        matching_servers = []
        
        for registration in self.servers.values():
            if capability in registration.config.capabilities:
                matching_servers.append(registration)
                registration.update_heartbeat()
        
        return matching_servers
    
    async def find_servers_by_tool(self, tool_name: str) -> List[MCPServerRegistration]:
        """
        Find servers that provide a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            List[MCPServerRegistration]: Servers with the tool
        """
        matching_servers = []
        
        for registration in self.servers.values():
            if tool_name in registration.config.tools:
                matching_servers.append(registration)
                registration.update_heartbeat()
        
        return matching_servers
    
    async def update_server_status(self, server_id: str, status: MCPServerStatus, 
                                  error_message: Optional[str] = None) -> bool:
        """
        Update server status.
        
        Args:
            server_id: ID of the server
            status: New status
            error_message: Optional error message
            
        Returns:
            bool: True if update successful
        """
        async with self._lock:
            if server_id not in self.servers:
                return False
            
            registration = self.servers[server_id]
            old_status = registration.status
            registration.status = status
            registration.update_heartbeat()
            
            if error_message:
                registration.last_error = error_message
            elif status != MCPServerStatus.ERROR:
                registration.last_error = None
            
            # Update counters
            if status == MCPServerStatus.RUNNING and old_status != MCPServerStatus.RUNNING:
                registration.start_count += 1
                registration.reset_restart_count()  # Reset on successful start
            
            logger.debug(f"Updated server {server_id} status: {old_status.value} -> {status.value}")
            return True
    
    async def get_server_count(self) -> int:
        """Get total number of registered servers"""
        return len(self.servers)
    
    async def get_server_count_by_type(self) -> Dict[str, int]:
        """Get server count grouped by type"""
        counts = {}
        
        for server_type, server_ids in self.server_types.items():
            counts[server_type.value] = len(server_ids)
        
        return counts
    
    async def get_server_count_by_status(self) -> Dict[str, int]:
        """Get server count grouped by status"""
        counts = {}
        
        for registration in self.servers.values():
            status = registration.status.value
            counts[status] = counts.get(status, 0) + 1
        
        return counts
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all registered servers.
        
        Returns:
            Dict with health information
        """
        total_servers = len(self.servers)
        running_servers = 0
        error_servers = 0
        stale_servers = 0
        
        for registration in self.servers.values():
            if registration.status == MCPServerStatus.RUNNING:
                running_servers += 1
            
            if registration.status == MCPServerStatus.ERROR:
                error_servers += 1
            
            if registration.is_stale():
                stale_servers += 1
        
        return {
            "total_servers": total_servers,
            "running_servers": running_servers,
            "error_servers": error_servers,
            "stale_servers": stale_servers,
            "server_types": await self.get_server_count_by_type(),
            "server_statuses": await self.get_server_count_by_status()
        }
    
    async def cleanup_stale_servers(self, timeout_seconds: int = 60) -> int:
        """
        Clean up stale server registrations.
        
        Args:
            timeout_seconds: Timeout for considering servers stale
            
        Returns:
            int: Number of servers cleaned up
        """
        async with self._lock:
            stale_servers = []
            
            for server_id, registration in self.servers.items():
                if registration.is_stale(timeout_seconds):
                    stale_servers.append(server_id)
            
            for server_id in stale_servers:
                await self.unregister_server(server_id)
                logger.warning(f"Cleaned up stale MCP server: {server_id}")
            
            return len(stale_servers)
    
    async def _heartbeat_monitor(self) -> None:
        """Monitor server heartbeats and clean up stale registrations"""
        while self._running:
            try:
                # Clean up stale servers every 2 minutes
                cleaned_up = await self.cleanup_stale_servers()
                if cleaned_up > 0:
                    logger.info(f"Cleaned up {cleaned_up} stale MCP server registrations")
                
                # Wait before next check
                await asyncio.sleep(120)  # 2 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in MCP server heartbeat monitor: {str(e)}")
                await asyncio.sleep(30)  # Wait 30 seconds before retry
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            "total_registrations": len(self.servers),
            "server_types": len(self.server_types),
            "available_types": [st.value for st in self.server_types.keys()],
            "running": self._running
        }
