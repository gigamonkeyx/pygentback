"""
MCP Tool Discovery Service

Implements proper tool discovery according to the MCP specification.
This service calls tools/list on each registered server and stores
the metadata in the database for agent access.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.types import Tool, Resource, Prompt
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    ClientSession = None
    StdioServerParameters = None
    Tool = None
    Resource = None
    Prompt = None

from .database.models import (
    MCPServerModel, MCPToolModel, MCPResourceModel, 
    MCPPromptModel, MCPToolCallLog
)
from .server.config import MCPServerConfig, MCPServerStatus


logger = logging.getLogger(__name__)


class MCPToolDiscoveryService:
    """
    Service for discovering and managing MCP server tools according to the MCP specification.
    
    This service implements the required client behavior:
    1. Call tools/list after server registration
    2. Store tool metadata in database
    3. Handle tools/list_changed notifications
    4. Provide tool access APIs for agents
    """
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self._active_sessions: Dict[str, ClientSession] = {}
        self._discovery_timeout = 30.0
        self._notification_handlers: Dict[str, asyncio.Task] = {}
    
    async def discover_server_capabilities(self, server_config: MCPServerConfig) -> Dict[str, Any]:
        """
        Discover all capabilities (tools, resources, prompts) for a server.
        
        This implements the MCP specification requirement that clients MUST
        call tools/list to discover available tools.
        """
        if not MCP_AVAILABLE:
            logger.warning("MCP SDK not available, skipping tool discovery")
            return {"tools": [], "resources": [], "prompts": [], "error": "MCP SDK not available"}
        
        try:
            logger.info(f"Starting capability discovery for server: {server_config.name}")
            
            # Create MCP client session
            session = await self._create_mcp_session(server_config)
            self._active_sessions[server_config.id] = session
            
            # Initialize session
            await session.initialize()
            
            capabilities = {
                "server_info": session.server_info,
                "tools": [],
                "resources": [],
                "prompts": [],
                "supports_tools": False,
                "supports_resources": False,
                "supports_prompts": False,
                "notifications": []
            }
            
            # Check server capabilities
            if hasattr(session.server_info, 'capabilities'):
                caps = session.server_info.capabilities
                capabilities["supports_tools"] = hasattr(caps, 'tools') and caps.tools is not None
                capabilities["supports_resources"] = hasattr(caps, 'resources') and caps.resources is not None
                capabilities["supports_prompts"] = hasattr(caps, 'prompts') and caps.prompts is not None
            
            # Discover tools if supported
            if capabilities["supports_tools"]:
                logger.info(f"Discovering tools for server: {server_config.name}")
                tools = await self._discover_tools(session, server_config)
                capabilities["tools"] = tools
                await self._store_tools_in_database(server_config.id, tools)
            
            # Discover resources if supported
            if capabilities["supports_resources"]:
                logger.info(f"Discovering resources for server: {server_config.name}")
                resources = await self._discover_resources(session, server_config)
                capabilities["resources"] = resources
                await self._store_resources_in_database(server_config.id, resources)
            
            # Discover prompts if supported
            if capabilities["supports_prompts"]:
                logger.info(f"Discovering prompts for server: {server_config.name}")
                prompts = await self._discover_prompts(session, server_config)
                capabilities["prompts"] = prompts
                await self._store_prompts_in_database(server_config.id, prompts)
            
            # Set up notification handlers
            await self._setup_notification_handlers(session, server_config)
            
            logger.info(f"Discovery complete for {server_config.name}: "
                       f"{len(capabilities['tools'])} tools, "
                       f"{len(capabilities['resources'])} resources, "
                       f"{len(capabilities['prompts'])} prompts")
            
            return capabilities
            
        except Exception as e:
            logger.error(f"Failed to discover capabilities for {server_config.name}: {e}")
            return {"tools": [], "resources": [], "prompts": [], "error": str(e)}
    
    async def _create_mcp_session(self, config: MCPServerConfig) -> ClientSession:
        """Create MCP client session"""
        if config.transport.value == "stdio":
            if isinstance(config.command, str):
                command_parts = config.command.split()
            elif isinstance(config.command, list):
                command_parts = config.command
            else:
                raise ValueError(f"Invalid command format: {config.command}")
            
            server_params = StdioServerParameters(
                command=command_parts[0],
                args=command_parts[1:] if len(command_parts) > 1 else [],
                env=config.custom_config.get("env", {}) if config.custom_config else {}
            )
            
            return await stdio_client(server_params)
        else:
            raise ValueError(f"Unsupported transport: {config.transport}")
    
    async def _discover_tools(self, session: ClientSession, config: MCPServerConfig) -> List[Dict[str, Any]]:
        """Discover tools using MCP tools/list endpoint"""
        try:
            # Call tools/list according to MCP spec
            tools_result = await asyncio.wait_for(
                session.list_tools(),
                timeout=self._discovery_timeout
            )
            
            discovered_tools = []
            for tool in tools_result.tools:
                tool_data = {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema or {},
                    "annotations": getattr(tool, 'annotations', None),
                    "server_id": config.id,
                    "discovered_at": datetime.utcnow().isoformat()
                }
                discovered_tools.append(tool_data)
                logger.debug(f"Discovered tool: {tool.name} - {tool.description}")
            
            return discovered_tools
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout discovering tools for server {config.name}")
            raise
        except Exception as e:
            logger.error(f"Failed to discover tools for server {config.name}: {e}")
            raise
    
    async def _discover_resources(self, session: ClientSession, config: MCPServerConfig) -> List[Dict[str, Any]]:
        """Discover resources using MCP resources/list endpoint"""
        try:
            resources_result = await asyncio.wait_for(
                session.list_resources(),
                timeout=self._discovery_timeout
            )
            
            discovered_resources = []
            for resource in resources_result.resources:
                resource_data = {
                    "uri": resource.uri,
                    "name": getattr(resource, 'name', None),
                    "description": getattr(resource, 'description', None),
                    "mime_type": getattr(resource, 'mimeType', None),
                    "annotations": getattr(resource, 'annotations', None),
                    "server_id": config.id,
                    "discovered_at": datetime.utcnow().isoformat()
                }
                discovered_resources.append(resource_data)
            
            return discovered_resources
            
        except Exception as e:
            logger.error(f"Failed to discover resources for server {config.name}: {e}")
            return []
    
    async def _discover_prompts(self, session: ClientSession, config: MCPServerConfig) -> List[Dict[str, Any]]:
        """Discover prompts using MCP prompts/list endpoint"""
        try:
            prompts_result = await asyncio.wait_for(
                session.list_prompts(),
                timeout=self._discovery_timeout
            )
            
            discovered_prompts = []
            for prompt in prompts_result.prompts:
                prompt_data = {
                    "name": prompt.name,
                    "description": getattr(prompt, 'description', None),
                    "arguments": getattr(prompt, 'arguments', None),
                    "annotations": getattr(prompt, 'annotations', None),
                    "server_id": config.id,
                    "discovered_at": datetime.utcnow().isoformat()
                }
                discovered_prompts.append(prompt_data)
            
            return discovered_prompts
            
        except Exception as e:
            logger.error(f"Failed to discover prompts for server {config.name}: {e}")
            return []
    
    async def _store_tools_in_database(self, server_id: str, tools: List[Dict[str, Any]]) -> None:
        """Store discovered tools in database"""
        try:
            # Remove existing tools for this server
            self.db_session.query(MCPToolModel).filter(
                MCPToolModel.server_id == server_id
            ).delete()
            
            # Add new tools
            for tool_data in tools:
                tool_model = MCPToolModel(
                    server_id=server_id,
                    name=tool_data["name"],
                    description=tool_data["description"],
                    input_schema=tool_data["input_schema"],
                    annotations=tool_data["annotations"],
                    discovered_at=datetime.utcnow(),
                    is_available=True
                )
                self.db_session.add(tool_model)
            
            self.db_session.commit()
            logger.info(f"Stored {len(tools)} tools in database for server {server_id}")
            
        except Exception as e:
            logger.error(f"Failed to store tools in database: {e}")
            self.db_session.rollback()
            raise
    
    async def _store_resources_in_database(self, server_id: str, resources: List[Dict[str, Any]]) -> None:
        """Store discovered resources in database"""
        try:
            # Remove existing resources for this server
            self.db_session.query(MCPResourceModel).filter(
                MCPResourceModel.server_id == server_id
            ).delete()
            
            # Add new resources
            for resource_data in resources:
                resource_model = MCPResourceModel(
                    server_id=server_id,
                    uri=resource_data["uri"],
                    name=resource_data["name"],
                    description=resource_data["description"],
                    mime_type=resource_data["mime_type"],
                    annotations=resource_data["annotations"],
                    discovered_at=datetime.utcnow(),
                    is_available=True
                )
                self.db_session.add(resource_model)
            
            self.db_session.commit()
            logger.info(f"Stored {len(resources)} resources in database for server {server_id}")
            
        except Exception as e:
            logger.error(f"Failed to store resources in database: {e}")
            self.db_session.rollback()
            raise
    
    async def _store_prompts_in_database(self, server_id: str, prompts: List[Dict[str, Any]]) -> None:
        """Store discovered prompts in database"""
        try:
            # Remove existing prompts for this server
            self.db_session.query(MCPPromptModel).filter(
                MCPPromptModel.server_id == server_id
            ).delete()
            
            # Add new prompts
            for prompt_data in prompts:
                prompt_model = MCPPromptModel(
                    server_id=server_id,
                    name=prompt_data["name"],
                    description=prompt_data["description"],
                    arguments=prompt_data["arguments"],
                    annotations=prompt_data["annotations"],
                    discovered_at=datetime.utcnow(),
                    is_available=True
                )
                self.db_session.add(prompt_model)
            
            self.db_session.commit()
            logger.info(f"Stored {len(prompts)} prompts in database for server {server_id}")
            
        except Exception as e:
            logger.error(f"Failed to store prompts in database: {e}")
            self.db_session.rollback()
            raise
    
    async def _setup_notification_handlers(self, session: ClientSession, config: MCPServerConfig) -> None:
        """Set up handlers for MCP notifications"""
        # Set up notification handlers for dynamic tool updates
        try:
            # Handler for tools/list_changed notifications
            async def on_tools_changed(notification):
                logger.info(f"Tools changed notification from {config.name}")
                await self._refresh_server_tools(config.name)
            
            # Handler for resources/list_changed notifications  
            async def on_resources_changed(notification):
                logger.info(f"Resources changed notification from {config.name}")
                # Refresh resource listings if needed
                
            # Register notification handlers with the session
            if hasattr(session, 'set_notification_handler'):
                session.set_notification_handler("tools/list_changed", on_tools_changed)
                session.set_notification_handler("resources/list_changed", on_resources_changed)
                
        except Exception as e:
            logger.warning(f"Failed to set up notification handlers for {config.name}: {e}")
    
    def get_all_available_tools(self) -> List[MCPToolModel]:
        """Get all available tools from database"""
        return self.db_session.query(MCPToolModel).filter(
            MCPToolModel.is_available == True
        ).all()
    
    def get_tools_by_server(self, server_id: str) -> List[MCPToolModel]:
        """Get tools for a specific server"""
        return self.db_session.query(MCPToolModel).filter(
            and_(
                MCPToolModel.server_id == server_id,
                MCPToolModel.is_available == True
            )
        ).all()
    
    def get_tool_by_name(self, tool_name: str) -> Optional[MCPToolModel]:
        """Find tool by name across all servers"""
        return self.db_session.query(MCPToolModel).filter(
            and_(
                MCPToolModel.name == tool_name,
                MCPToolModel.is_available == True
            )
        ).first()
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any], agent_id: str = None) -> Any:
        """Call a tool by name and log the call"""
        tool_model = self.get_tool_by_name(tool_name)
        if not tool_model:
            raise ValueError(f"Tool not found: {tool_name}")
        
        session = self._active_sessions.get(tool_model.server_id)
        if not session:
            raise RuntimeError(f"No active session for server: {tool_model.server_id}")
        
        start_time = datetime.utcnow()
        success = False
        error_message = None
        response_data = None
        
        try:
            # Call the tool
            result = await session.call_tool(tool_name, arguments)
            response_data = result
            success = True
            
            # Update tool usage statistics
            tool_model.call_count += 1
            tool_model.last_called = start_time
            
            return result
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Failed to call tool {tool_name}: {e}")
            raise
            
        finally:
            # Log the call
            end_time = datetime.utcnow()
            response_time = int((end_time - start_time).total_seconds() * 1000)
            
            call_log = MCPToolCallLog(
                tool_id=tool_model.id,
                server_id=tool_model.server_id,
                agent_id=agent_id,
                arguments=arguments,
                response_data=response_data,
                success=success,
                error_message=error_message,
                response_time_ms=response_time,
                called_at=start_time,
                completed_at=end_time
            )
            
            self.db_session.add(call_log)
            self.db_session.commit()
    
    def get_discovery_summary(self) -> Dict[str, Any]:
        """Get summary of discovered capabilities"""
        tools = self.get_all_available_tools()
        
        summary = {
            "total_tools": len(tools),
            "servers_with_tools": len(set(tool.server_id for tool in tools)),
            "tools_by_server": {},
            "most_used_tools": [],
            "discovery_timestamp": datetime.utcnow().isoformat()
        }
        
        # Group tools by server
        for tool in tools:
            server_id = tool.server_id
            if server_id not in summary["tools_by_server"]:
                summary["tools_by_server"][server_id] = []
            summary["tools_by_server"][server_id].append({
                "name": tool.name,
                "description": tool.description,
                "call_count": tool.call_count,
                "last_called": tool.last_called.isoformat() if tool.last_called else None
            })
        
        # Get most used tools
        most_used = sorted(tools, key=lambda t: t.call_count, reverse=True)[:10]
        summary["most_used_tools"] = [
            {
                "name": tool.name,
                "server_id": tool.server_id,
                "call_count": tool.call_count,
                "description": tool.description
            }
            for tool in most_used
        ]
        
        return summary
    
    async def shutdown(self) -> None:
        """Close all MCP sessions"""
        for server_id, session in self._active_sessions.items():
            try:
                await session.close()
            except Exception as e:
                logger.error(f"Error closing session for server {server_id}: {e}")
        
        self._active_sessions.clear()
        
        # Cancel notification handlers
        for task in self._notification_handlers.values():
            task.cancel()
        
        self._notification_handlers.clear()
