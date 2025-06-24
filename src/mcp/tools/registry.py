"""
MCP Tool Registry

This module provides registration and discovery functionality for MCP tools.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from dataclasses import dataclass, field

try:
    from mcp.types import Tool
except ImportError:
    # Fallback if mcp is not available
    Tool = Any


logger = logging.getLogger(__name__)


@dataclass
class MCPToolInfo:
    """Information about an MCP tool"""
    name: str
    description: str
    server_id: str
    server_name: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    categories: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    
    # Metadata
    registered_at: datetime = field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None
    usage_count: int = 0
    
    # Availability
    available: bool = True
    error_message: Optional[str] = None
    
    def mark_used(self) -> None:
        """Mark tool as used"""
        self.last_used = datetime.utcnow()
        self.usage_count += 1
    
    def set_unavailable(self, error_message: str) -> None:
        """Mark tool as unavailable"""
        self.available = False
        self.error_message = error_message
    
    def set_available(self) -> None:
        """Mark tool as available"""
        self.available = True
        self.error_message = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "server_id": self.server_id,
            "server_name": self.server_name,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "categories": self.categories,
            "tags": self.tags,
            "version": self.version,
            "registered_at": self.registered_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "usage_count": self.usage_count,
            "available": self.available,
            "error_message": self.error_message
        }


class MCPToolRegistry:
    """
    Registry for managing MCP tools.
    
    Provides centralized registration, discovery, and metadata
    management for all MCP tools in the system.
    """
    
    def __init__(self):
        self.tools: Dict[str, MCPToolInfo] = {}  # tool_name -> tool_info
        self.server_tools: Dict[str, Set[str]] = {}  # server_id -> set of tool names
        self.categories: Dict[str, Set[str]] = {}  # category -> set of tool names
        self.tags: Dict[str, Set[str]] = {}  # tag -> set of tool names
        self._lock = asyncio.Lock()
    
    async def register_tool(self, tool_info: MCPToolInfo) -> bool:
        """
        Register a tool in the registry.
        
        Args:
            tool_info: Tool information
            
        Returns:
            bool: True if registration successful
        """
        async with self._lock:
            try:
                tool_name = tool_info.name
                
                # Check for conflicts
                if tool_name in self.tools:
                    existing_tool = self.tools[tool_name]
                    if existing_tool.server_id != tool_info.server_id:
                        logger.warning(f"Tool name conflict: {tool_name} already exists from server {existing_tool.server_id}")
                        return False
                
                # Register tool
                self.tools[tool_name] = tool_info
                
                # Update server index
                server_id = tool_info.server_id
                if server_id not in self.server_tools:
                    self.server_tools[server_id] = set()
                self.server_tools[server_id].add(tool_name)
                
                # Update category index
                for category in tool_info.categories:
                    if category not in self.categories:
                        self.categories[category] = set()
                    self.categories[category].add(tool_name)
                
                # Update tag index
                for tag in tool_info.tags:
                    if tag not in self.tags:
                        self.tags[tag] = set()
                    self.tags[tag].add(tool_name)
                
                logger.info(f"Registered MCP tool: {tool_name} from server {tool_info.server_name}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to register MCP tool {tool_info.name}: {str(e)}")
                return False
    
    async def unregister_tool(self, tool_name: str) -> bool:
        """
        Unregister a tool from the registry.
        
        Args:
            tool_name: Name of the tool to unregister
            
        Returns:
            bool: True if unregistration successful
        """
        async with self._lock:
            try:
                if tool_name not in self.tools:
                    logger.warning(f"Tool {tool_name} not found in registry")
                    return False
                
                tool_info = self.tools[tool_name]
                
                # Remove from server index
                server_id = tool_info.server_id
                if server_id in self.server_tools:
                    self.server_tools[server_id].discard(tool_name)
                    if not self.server_tools[server_id]:
                        del self.server_tools[server_id]
                
                # Remove from category index
                for category in tool_info.categories:
                    if category in self.categories:
                        self.categories[category].discard(tool_name)
                        if not self.categories[category]:
                            del self.categories[category]
                
                # Remove from tag index
                for tag in tool_info.tags:
                    if tag in self.tags:
                        self.tags[tag].discard(tool_name)
                        if not self.tags[tag]:
                            del self.tags[tag]
                
                # Remove tool
                del self.tools[tool_name]
                
                logger.info(f"Unregistered MCP tool: {tool_name}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to unregister MCP tool {tool_name}: {str(e)}")
                return False
    
    async def unregister_server_tools(self, server_id: str) -> int:
        """
        Unregister all tools from a server.
        
        Args:
            server_id: ID of the server
            
        Returns:
            int: Number of tools unregistered
        """
        if server_id not in self.server_tools:
            return 0
        
        tool_names = list(self.server_tools[server_id])
        count = 0
        
        for tool_name in tool_names:
            if await self.unregister_tool(tool_name):
                count += 1
        
        return count
    
    async def get_tool(self, tool_name: str) -> Optional[MCPToolInfo]:
        """
        Get tool information by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            MCPToolInfo or None if not found
        """
        return self.tools.get(tool_name)
    
    async def list_tools(self, server_id: Optional[str] = None,
                        category: Optional[str] = None,
                        tag: Optional[str] = None,
                        available_only: bool = False) -> List[MCPToolInfo]:
        """
        List tools with optional filtering.
        
        Args:
            server_id: Filter by server ID
            category: Filter by category
            tag: Filter by tag
            available_only: Only return available tools
            
        Returns:
            List[MCPToolInfo]: List of matching tools
        """
        tools = []
        
        # Determine which tools to check
        if server_id and server_id in self.server_tools:
            tool_names = self.server_tools[server_id]
        elif category and category in self.categories:
            tool_names = self.categories[category]
        elif tag and tag in self.tags:
            tool_names = self.tags[tag]
        else:
            tool_names = self.tools.keys()
        
        for tool_name in tool_names:
            if tool_name not in self.tools:
                continue
            
            tool_info = self.tools[tool_name]
            
            # Apply filters
            if server_id and tool_info.server_id != server_id:
                continue
            
            if category and category not in tool_info.categories:
                continue
            
            if tag and tag not in tool_info.tags:
                continue
            
            if available_only and not tool_info.available:
                continue
            
            tools.append(tool_info)
        
        return tools
    
    async def search_tools(self, query: str) -> List[MCPToolInfo]:
        """
        Search tools by name or description.
        
        Args:
            query: Search query
            
        Returns:
            List[MCPToolInfo]: Matching tools
        """
        query_lower = query.lower()
        matching_tools = []
        
        for tool_info in self.tools.values():
            # Search in name and description
            if (query_lower in tool_info.name.lower() or 
                query_lower in tool_info.description.lower()):
                matching_tools.append(tool_info)
                continue
            
            # Search in categories and tags
            if any(query_lower in cat.lower() for cat in tool_info.categories):
                matching_tools.append(tool_info)
                continue
            
            if any(query_lower in tag.lower() for tag in tool_info.tags):
                matching_tools.append(tool_info)
                continue
        
        return matching_tools
    
    async def get_tools_by_server(self, server_id: str) -> List[MCPToolInfo]:
        """Get all tools from a specific server"""
        return await self.list_tools(server_id=server_id)
    
    async def get_available_tools(self) -> List[MCPToolInfo]:
        """Get all available tools"""
        return await self.list_tools(available_only=True)
    
    async def mark_tool_used(self, tool_name: str) -> bool:
        """
        Mark a tool as used (updates usage statistics).
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            bool: True if successful
        """
        tool_info = self.tools.get(tool_name)
        if tool_info:
            tool_info.mark_used()
            return True
        return False
    
    async def set_tool_availability(self, tool_name: str, available: bool, 
                                   error_message: Optional[str] = None) -> bool:
        """
        Set tool availability status.
        
        Args:
            tool_name: Name of the tool
            available: Whether tool is available
            error_message: Optional error message if unavailable
            
        Returns:
            bool: True if successful
        """
        tool_info = self.tools.get(tool_name)
        if tool_info:
            if available:
                tool_info.set_available()
            else:
                tool_info.set_unavailable(error_message or "Tool unavailable")
            return True
        return False
    
    async def get_tool_count(self) -> int:
        """Get total number of registered tools"""
        return len(self.tools)
    
    async def get_tool_count_by_server(self) -> Dict[str, int]:
        """Get tool count grouped by server"""
        return {server_id: len(tools) for server_id, tools in self.server_tools.items()}
    
    async def get_tool_count_by_category(self) -> Dict[str, int]:
        """Get tool count grouped by category"""
        return {category: len(tools) for category, tools in self.categories.items()}
    
    async def get_available_categories(self) -> List[str]:
        """Get list of available categories"""
        return list(self.categories.keys())
    
    async def get_available_tags(self) -> List[str]:
        """Get list of available tags"""
        return list(self.tags.keys())
    
    async def get_most_used_tools(self, limit: int = 10) -> List[MCPToolInfo]:
        """
        Get most frequently used tools.
        
        Args:
            limit: Maximum number of tools to return
            
        Returns:
            List[MCPToolInfo]: Most used tools
        """
        sorted_tools = sorted(
            self.tools.values(),
            key=lambda t: t.usage_count,
            reverse=True
        )
        return sorted_tools[:limit]
    
    async def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        total_tools = len(self.tools)
        available_tools = len([t for t in self.tools.values() if t.available])
        
        return {
            "total_tools": total_tools,
            "available_tools": available_tools,
            "unavailable_tools": total_tools - available_tools,
            "servers": len(self.server_tools),
            "categories": len(self.categories),
            "tags": len(self.tags),
            "tools_by_server": await self.get_tool_count_by_server(),
            "tools_by_category": await self.get_tool_count_by_category()
        }
    
    def clear_registry(self) -> None:
        """Clear all tools from the registry"""
        self.tools.clear()
        self.server_tools.clear()
        self.categories.clear()
        self.tags.clear()
        logger.info("Cleared MCP tool registry")
