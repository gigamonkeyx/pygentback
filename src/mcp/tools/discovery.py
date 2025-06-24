"""
MCP Tool Discovery

Provides tool discovery and registration capabilities for MCP servers,
supporting automatic tool detection and capability mapping.
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Tool categories for classification"""
    FILE_OPERATIONS = "file_operations"
    DATABASE = "database"
    WEB_SCRAPING = "web_scraping"
    API_INTEGRATION = "api_integration"
    DATA_PROCESSING = "data_processing"
    SYSTEM_OPERATIONS = "system_operations"
    COMMUNICATION = "communication"
    ANALYSIS = "analysis"
    AUTOMATION = "automation"
    UNKNOWN = "unknown"


@dataclass
class ToolCapability:
    """Tool capability description"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Optional[Dict[str, Any]] = None
    category: ToolCategory = ToolCategory.UNKNOWN
    tags: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MCPServerInfo:
    """MCP Server information"""
    server_name: str
    server_url: str
    capabilities: List[ToolCapability] = field(default_factory=list)
    status: str = "unknown"
    last_discovered: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MCPToolDiscovery:
    """
    MCP Tool Discovery service for finding and cataloging available tools.
    
    Provides automatic discovery of MCP server capabilities and
    tool registration for the PyGent Factory ecosystem.
    """
    
    def __init__(self):
        self.discovered_servers: Dict[str, MCPServerInfo] = {}
        self.tool_catalog: Dict[str, ToolCapability] = {}
        self.category_index: Dict[ToolCategory, Set[str]] = {}
        self.is_initialized = False
        
        # Initialize category index
        for category in ToolCategory:
            self.category_index[category] = set()
    
    async def initialize(self):
        """Initialize tool discovery service"""
        try:
            self.is_initialized = True
            logger.info("MCP Tool Discovery initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MCP Tool Discovery: {e}")
            raise
    
    async def discover_server(self, server_name: str, server_url: str) -> MCPServerInfo:
        """Discover tools available on an MCP server"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Create or update server info
            server_info = MCPServerInfo(
                server_name=server_name,
                server_url=server_url,
                last_discovered=datetime.utcnow()
            )
            
            # REAL tool discovery by querying the actual MCP server
            capabilities = await self._real_server_discovery(server_name, server_url)
            server_info.capabilities = capabilities
            server_info.status = "active"
            
            # Store server info
            self.discovered_servers[server_name] = server_info
            
            # Update tool catalog
            for capability in capabilities:
                tool_key = f"{server_name}.{capability.name}"
                self.tool_catalog[tool_key] = capability
                
                # Update category index
                self.category_index[capability.category].add(tool_key)
            
            logger.info(f"Discovered {len(capabilities)} tools from server {server_name}")
            return server_info
            
        except Exception as e:
            logger.error(f"Failed to discover server {server_name}: {e}")
            # Create failed server info
            server_info = MCPServerInfo(
                server_name=server_name,
                server_url=server_url,
                status="failed",
                last_discovered=datetime.utcnow()
            )
            self.discovered_servers[server_name] = server_info
            raise
    
    async def _real_server_discovery(self, server_name: str, server_url: str) -> List[ToolCapability]:
        """REAL server discovery by querying actual MCP server"""
        capabilities = []

        try:
            # REAL MCP server capability discovery
            from mcp.client import MCPClient

            # Connect to real MCP server
            mcp_client = MCPClient(server_url)
            connection_result = await mcp_client.connect()

            if connection_result.get('success'):
                # Query real server capabilities
                server_capabilities = await mcp_client.list_tools()

                # Convert to ToolCapability objects
                for tool_info in server_capabilities:
                    capability = ToolCapability(
                        name=tool_info.get('name', 'unknown'),
                        description=tool_info.get('description', 'No description'),
                        input_schema=tool_info.get('inputSchema', {}),
                        category=self._determine_tool_category(tool_info),
                        tags=tool_info.get('tags', [])
                    )
                    capabilities.append(capability)

                await mcp_client.disconnect()
                logger.info(f"Discovered {len(capabilities)} real capabilities from {server_name}")

            else:
                logger.warning(f"Failed to connect to MCP server {server_name}: {connection_result.get('error')}")
                # Fallback to pattern-based discovery
                capabilities = self._fallback_capability_discovery(server_name)

        except Exception as e:
            logger.error(f"Real MCP discovery failed for {server_name}: {e}")
            # Fallback to pattern-based discovery
            capabilities = self._fallback_capability_discovery(server_name)

        return capabilities

    def _fallback_capability_discovery(self, server_name: str) -> List[ToolCapability]:
        """Fallback capability discovery when real MCP connection fails"""
        capabilities = []

        # Pattern-based fallback discovery
        if "file" in server_name.lower():
            capabilities.extend([
                ToolCapability(
                    name="read_file",
                    description="Read contents of a file",
                    input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
                    category=ToolCategory.FILE_OPERATIONS,
                    tags=["file", "read", "io"]
                ),
                ToolCapability(
                    name="write_file",
                    description="Write contents to a file",
                    input_schema={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}},
                    category=ToolCategory.FILE_OPERATIONS,
                    tags=["file", "write", "io"]
                ),
                ToolCapability(
                    name="list_directory",
                    description="List files in a directory",
                    input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
                    category=ToolCategory.FILE_OPERATIONS,
                    tags=["file", "directory", "list"]
                )
            ])
        
        if "database" in server_name.lower() or "sql" in server_name.lower():
            capabilities.extend([
                ToolCapability(
                    name="execute_query",
                    description="Execute SQL query",
                    input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
                    category=ToolCategory.DATABASE,
                    tags=["database", "sql", "query"]
                ),
                ToolCapability(
                    name="get_schema",
                    description="Get database schema information",
                    input_schema={"type": "object", "properties": {"table": {"type": "string"}}},
                    category=ToolCategory.DATABASE,
                    tags=["database", "schema", "metadata"]
                )
            ])
        
        if "web" in server_name.lower() or "http" in server_name.lower():
            capabilities.extend([
                ToolCapability(
                    name="fetch_url",
                    description="Fetch content from URL",
                    input_schema={"type": "object", "properties": {"url": {"type": "string"}}},
                    category=ToolCategory.WEB_SCRAPING,
                    tags=["web", "http", "fetch"]
                ),
                ToolCapability(
                    name="scrape_page",
                    description="Scrape structured data from web page",
                    input_schema={"type": "object", "properties": {"url": {"type": "string"}, "selector": {"type": "string"}}},
                    category=ToolCategory.WEB_SCRAPING,
                    tags=["web", "scraping", "extraction"]
                )
            ])
        
        if "api" in server_name.lower():
            capabilities.extend([
                ToolCapability(
                    name="call_api",
                    description="Make API call",
                    input_schema={"type": "object", "properties": {"endpoint": {"type": "string"}, "method": {"type": "string"}}},
                    category=ToolCategory.API_INTEGRATION,
                    tags=["api", "integration", "http"]
                )
            ])
        
        # Default capabilities for any server
        if not capabilities:
            capabilities.append(
                ToolCapability(
                    name="ping",
                    description="Check server status",
                    input_schema={"type": "object", "properties": {}},
                    category=ToolCategory.SYSTEM_OPERATIONS,
                    tags=["system", "health", "ping"]
                )
            )
        
        return capabilities
    
    def get_server_info(self, server_name: str) -> Optional[MCPServerInfo]:
        """Get information about a discovered server"""
        return self.discovered_servers.get(server_name)
    
    def get_all_servers(self) -> List[MCPServerInfo]:
        """Get all discovered servers"""
        return list(self.discovered_servers.values())
    
    def get_tool_capability(self, tool_key: str) -> Optional[ToolCapability]:
        """Get capability information for a tool"""
        return self.tool_catalog.get(tool_key)
    
    def search_tools(self, 
                    query: Optional[str] = None,
                    category: Optional[ToolCategory] = None,
                    tags: Optional[List[str]] = None) -> List[str]:
        """Search for tools by query, category, or tags"""
        results = set()
        
        # Search by category
        if category:
            results.update(self.category_index.get(category, set()))
        else:
            # Start with all tools
            results.update(self.tool_catalog.keys())
        
        # Filter by query
        if query:
            query_lower = query.lower()
            filtered_results = set()
            for tool_key in results:
                capability = self.tool_catalog[tool_key]
                if (query_lower in capability.name.lower() or 
                    query_lower in capability.description.lower()):
                    filtered_results.add(tool_key)
            results = filtered_results
        
        # Filter by tags
        if tags:
            filtered_results = set()
            for tool_key in results:
                capability = self.tool_catalog[tool_key]
                if any(tag in capability.tags for tag in tags):
                    filtered_results.add(tool_key)
            results = filtered_results
        
        return list(results)
    
    def get_tools_by_category(self, category: ToolCategory) -> List[str]:
        """Get all tools in a specific category"""
        return list(self.category_index.get(category, set()))
    
    def get_category_stats(self) -> Dict[ToolCategory, int]:
        """Get statistics about tools by category"""
        return {category: len(tools) for category, tools in self.category_index.items()}
    
    async def refresh_server(self, server_name: str) -> bool:
        """Refresh discovery for a specific server"""
        if server_name not in self.discovered_servers:
            return False
        
        try:
            server_info = self.discovered_servers[server_name]
            await self.discover_server(server_name, server_info.server_url)
            return True
        except Exception as e:
            logger.error(f"Failed to refresh server {server_name}: {e}")
            return False
    
    async def refresh_all_servers(self) -> int:
        """Refresh discovery for all servers"""
        refreshed_count = 0
        for server_name in list(self.discovered_servers.keys()):
            if await self.refresh_server(server_name):
                refreshed_count += 1
        
        logger.info(f"Refreshed {refreshed_count} servers")
        return refreshed_count

    def _determine_tool_category(self, tool_info: Dict[str, Any]) -> ToolCategory:
        """Determine tool category from tool information"""
        tool_name = tool_info.get('name', '').lower()
        tool_desc = tool_info.get('description', '').lower()

        # Categorize based on name and description patterns
        if any(keyword in tool_name or keyword in tool_desc for keyword in ['file', 'read', 'write', 'directory']):
            return ToolCategory.FILE_OPERATIONS
        elif any(keyword in tool_name or keyword in tool_desc for keyword in ['database', 'sql', 'query']):
            return ToolCategory.DATABASE
        elif any(keyword in tool_name or keyword in tool_desc for keyword in ['web', 'http', 'api', 'request']):
            return ToolCategory.WEB_OPERATIONS
        elif any(keyword in tool_name or keyword in tool_desc for keyword in ['search', 'find', 'lookup']):
            return ToolCategory.SEARCH
        elif any(keyword in tool_name or keyword in tool_desc for keyword in ['data', 'process', 'transform']):
            return ToolCategory.DATA_PROCESSING
        else:
            return ToolCategory.UTILITY
    
    def remove_server(self, server_name: str) -> bool:
        """Remove a server from discovery"""
        if server_name not in self.discovered_servers:
            return False
        
        # Remove server
        server_info = self.discovered_servers[server_name]
        del self.discovered_servers[server_name]
        
        # Remove tools from catalog
        tools_to_remove = []
        for tool_key in self.tool_catalog:
            if tool_key.startswith(f"{server_name}."):
                tools_to_remove.append(tool_key)
        
        for tool_key in tools_to_remove:
            capability = self.tool_catalog[tool_key]
            del self.tool_catalog[tool_key]
            
            # Remove from category index
            self.category_index[capability.category].discard(tool_key)
        
        logger.info(f"Removed server {server_name} and {len(tools_to_remove)} tools")
        return True
    
    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get discovery statistics"""
        active_servers = sum(1 for s in self.discovered_servers.values() if s.status == "active")
        failed_servers = sum(1 for s in self.discovered_servers.values() if s.status == "failed")
        
        return {
            "total_servers": len(self.discovered_servers),
            "active_servers": active_servers,
            "failed_servers": failed_servers,
            "total_tools": len(self.tool_catalog),
            "categories": self.get_category_stats(),
            "is_initialized": self.is_initialized
        }
