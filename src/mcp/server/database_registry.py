"""
Database-backed MCP Server Registry

This module provides persistent MCP server registration using the database models.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy import select, delete

from ...database.models import MCPServer, MCPTool
from ...database.session import get_session
from .config import MCPServerConfig, MCPServerStatus, MCPServerType


logger = logging.getLogger(__name__)


class DatabaseMCPServerRegistry:
    """
    Database-backed MCP server registry.
    
    Provides persistent registration and discovery functionality for MCP servers
    using SQLAlchemy database models.
    """
    
    def __init__(self, session_factory=None):
        """
        Initialize the database MCP server registry.
        
        Args:
            session_factory: SQLAlchemy session factory
        """
        self.session_factory = session_factory or get_session
        self._lock = asyncio.Lock()
        self._running = False
    
    async def start(self) -> None:
        """Start the database MCP server registry"""
        if self._running:
            return
        
        self._running = True
        logger.info("Database MCP server registry started")
    
    async def stop(self) -> None:
        """Stop the database MCP server registry"""
        if not self._running:
            return
        
        self._running = False
        logger.info("Database MCP server registry stopped")
    
    async def register_server(self, config: MCPServerConfig) -> bool:
        """
        Register an MCP server in the database.
        
        Args:
            config: Server configuration
            
        Returns:
            bool: True if registration successful
        """
        async with self._lock:
            try:
                async with self.session_factory() as session:
                    # Check if server already exists
                    existing = await session.get(MCPServer, config.id)
                    if existing:
                        logger.warning(f"MCP server {config.id} already registered")
                        return False
                    
                    # Create new server record
                    server = MCPServer(
                        id=config.id,
                        name=config.name,
                        command=config.command if isinstance(config.command, list) else [config.command],
                        capabilities=config.capabilities,
                        transport=config.transport,
                        config=config.to_dict(),
                        status="inactive"
                    )
                    
                    session.add(server)
                    await session.commit()
                    
                    logger.info(f"Registered MCP server in database: {config.name} ({config.id})")
                    return True
                    
            except IntegrityError as e:
                logger.error(f"Database integrity error registering server {config.id}: {str(e)}")
                return False
            except Exception as e:
                logger.error(f"Failed to register MCP server {config.id} in database: {str(e)}")
                return False
    
    async def unregister_server(self, server_id: str) -> bool:
        """
        Unregister an MCP server from the database.
        
        Args:
            server_id: ID of the server to unregister
            
        Returns:
            bool: True if unregistration successful
        """
        async with self._lock:
            try:
                async with self.session_factory() as session:
                    # Find and delete server
                    server = await session.get(MCPServer, server_id)
                    if not server:
                        logger.warning(f"MCP server {server_id} not found in database")
                        return False
                    
                    server_name = server.name
                    await session.delete(server)
                    await session.commit()
                    
                    logger.info(f"Unregistered MCP server from database: {server_name} ({server_id})")
                    return True
                    
            except Exception as e:
                logger.error(f"Failed to unregister MCP server {server_id} from database: {str(e)}")
                return False
    
    async def get_server(self, server_id: str) -> Optional[MCPServerConfig]:
        """
        Get a server configuration by ID.
        
        Args:
            server_id: ID of the server
            
        Returns:
            MCPServerConfig or None if not found
        """
        try:
            async with self.session_factory() as session:
                server = await session.get(MCPServer, server_id)
                if not server:
                    return None
                
                return self._db_server_to_config(server)
                
        except Exception as e:
            logger.error(f"Failed to get server {server_id} from database: {str(e)}")
            return None
    
    async def list_servers(self) -> List[MCPServerConfig]:
        """
        List all registered servers.
        
        Returns:
            List of server configurations
        """
        try:
            async with self.session_factory() as session:
                result = await session.execute(select(MCPServer))
                servers = result.scalars().all()
                
                return [self._db_server_to_config(server) for server in servers]
                
        except Exception as e:
            logger.error(f"Failed to list servers from database: {str(e)}")
            return []
    
    async def update_server_status(self, server_id: str, status: MCPServerStatus) -> bool:
        """
        Update server status in the database.
        
        Args:
            server_id: ID of the server
            status: New status
            
        Returns:
            bool: True if update successful
        """
        try:
            async with self.session_factory() as session:
                server = await session.get(MCPServer, server_id)
                if not server:
                    logger.warning(f"Server {server_id} not found for status update")
                    return False
                
                server.status = status.value if isinstance(status, MCPServerStatus) else status
                await session.commit()
                
                logger.debug(f"Updated server {server_id} status to {status}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update server {server_id} status: {str(e)}")
            return False
    
    async def register_server_tools(self, server_id: str, tools: List[Dict[str, Any]]) -> bool:
        """
        Register tools for a server.
        
        Args:
            server_id: ID of the server
            tools: List of tool definitions
            
        Returns:
            bool: True if registration successful
        """
        try:
            async with self.session_factory() as session:
                # Delete existing tools for this server
                await session.execute(delete(MCPTool).where(MCPTool.server_id == server_id))
                
                # Add new tools
                for tool_def in tools:
                    tool = MCPTool(
                        server_id=server_id,
                        name=tool_def.get('name', ''),
                        description=tool_def.get('description', ''),
                        parameters=tool_def.get('parameters', {})
                    )
                    session.add(tool)
                
                await session.commit()
                
                logger.info(f"Registered {len(tools)} tools for server {server_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to register tools for server {server_id}: {str(e)}")
            return False
    
    async def get_servers_by_type(self, server_type: MCPServerType) -> List[MCPServerConfig]:
        """
        Get servers by type.
        
        Args:
            server_type: Type of servers to get
            
        Returns:
            List of server configurations
        """
        try:
            async with self.session_factory() as session:
                # This would require adding server_type to the database model
                # For now, filter by capabilities or other means
                result = await session.execute(select(MCPServer))
                servers = result.scalars().all()
                
                # Filter by type (this is a simplified implementation)
                filtered_servers = []
                for server in servers:
                    config = self._db_server_to_config(server)
                    if config.server_type == server_type:
                        filtered_servers.append(config)
                
                return filtered_servers
                
        except Exception as e:
            logger.error(f"Failed to get servers by type {server_type}: {str(e)}")
            return []
    
    async def clear_all_servers(self) -> bool:
        """
        Clear all servers from the database.
        
        Returns:
            bool: True if clearing successful
        """
        try:
            async with self.session_factory() as session:
                await session.execute(delete(MCPServer))
                await session.commit()
                
                logger.info("Cleared all servers from database")
                return True
                
        except Exception as e:
            logger.error(f"Failed to clear servers from database: {str(e)}")
            return False
    
    def _db_server_to_config(self, server: MCPServer) -> MCPServerConfig:
        """
        Convert database server model to config object.
        
        Args:
            server: Database server model
            
        Returns:
            MCPServerConfig object
        """
        return MCPServerConfig(
            id=server.id,
            name=server.name,
            command=server.command,
            capabilities=server.capabilities,
            transport=server.transport,
            **server.config
        )
    
    async def get_server_status(self, server_id: str) -> Optional[str]:
        """
        Get server status from database.
        
        Args:
            server_id: ID of the server
            
        Returns:
            Server status or None if not found
        """
        try:
            async with self.session_factory() as session:
                server = await session.get(MCPServer, server_id)
                return server.status if server else None
                
        except Exception as e:
            logger.error(f"Failed to get server {server_id} status: {str(e)}")
            return None
