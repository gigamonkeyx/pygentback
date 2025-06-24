"""
MCP (Model Context Protocol) API Routes

This module provides REST API endpoints for MCP server management
including server registration, tool discovery, and execution.
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel

from ...mcp.server_registry import MCPServerManager, MCPServerConfig
from ...security.auth import get_current_user, require_mcp_execute, User


logger = logging.getLogger(__name__)

router = APIRouter()

# Global MCP manager instance (will be set by main.py)
_mcp_manager: Optional[MCPServerManager] = None

def set_mcp_manager(manager: MCPServerManager):
    """Set the global MCP manager instance"""
    global _mcp_manager
    _mcp_manager = manager

def get_mcp_manager() -> MCPServerManager:
    """Get the MCP manager dependency"""
    if _mcp_manager is None:
        raise HTTPException(status_code=500, detail="MCP manager not initialized")
    return _mcp_manager


# Request/Response models
class RegisterServerRequest(BaseModel):
    name: str
    command: List[str]
    capabilities: List[str] = []
    transport: str = "stdio"
    config: Dict[str, Any] = {}
    auto_start: bool = True


class CallToolRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = {}


class InstallServerRequest(BaseModel):
    server_name: str
    source_type: str = "npm"  # npm, git, pip, local
    source_url: Optional[str] = None
    version: Optional[str] = None
    auto_start: bool = True


class InstallationStatus(BaseModel):
    server_name: str
    status: str  # pending, installing, completed, failed
    progress: int = 0
    message: str = ""
    install_path: Optional[str] = None


@router.get("/servers")
async def list_servers(
    mcp_manager: MCPServerManager = Depends(get_mcp_manager)
):
    """List all registered MCP servers (public endpoint)"""
    try:
        servers = await mcp_manager.list_servers()
        return {"servers": servers}
        
    except Exception as e:
        logger.error(f"Failed to list MCP servers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list servers: {str(e)}")


@router.get("/servers/public")
async def list_servers_public(
    mcp_manager: MCPServerManager = Depends(get_mcp_manager)
):
    """List all registered MCP servers (public endpoint - no authentication required)"""
    try:
        servers = await mcp_manager.list_servers()
        return {"servers": servers}
        
    except Exception as e:
        logger.error(f"Failed to list MCP servers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list servers: {str(e)}")


@router.post("/servers")
async def register_server(
    request: RegisterServerRequest,
    mcp_manager: MCPServerManager = Depends(get_mcp_manager),
    current_user: User = Depends(get_current_user)
):
    """Register a new MCP server"""
    try:
        config = MCPServerConfig(
            name=request.name,
            command=request.command,
            capabilities=request.capabilities,
            transport=request.transport,
            config=request.config,
            auto_start=request.auto_start
        )
        
        server_id = await mcp_manager.register_server(config)
        
        return {
            "server_id": server_id,
            "message": f"Server '{request.name}' registered successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to register MCP server: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to register server: {str(e)}")


@router.get("/servers/{server_id}")
async def get_server_status(
    server_id: str,
    mcp_manager: MCPServerManager = Depends(get_mcp_manager),
    current_user: User = Depends(get_current_user)
):
    """Get MCP server status"""
    try:
        status = await mcp_manager.get_server_status(server_id)
        if not status:
            raise HTTPException(status_code=404, detail="Server not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get server status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get server status: {str(e)}")


@router.post("/servers/{server_id}/start")
async def start_server(
    server_id: str,
    mcp_manager: MCPServerManager = Depends(get_mcp_manager),
    current_user: User = Depends(get_current_user)
):
    """Start an MCP server"""
    try:
        success = await mcp_manager.start_server(server_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to start server")
        
        return {"message": f"Server {server_id} started successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start server {server_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start server: {str(e)}")


@router.post("/servers/{server_id}/stop")
async def stop_server(
    server_id: str,
    mcp_manager: MCPServerManager = Depends(get_mcp_manager),
    current_user: User = Depends(get_current_user)
):
    """Stop an MCP server"""
    try:
        success = await mcp_manager.stop_server(server_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to stop server")
        
        return {"message": f"Server {server_id} stopped successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop server {server_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to stop server: {str(e)}")


@router.post("/servers/{server_id}/restart")
async def restart_server(
    server_id: str,
    mcp_manager: MCPServerManager = Depends(get_mcp_manager),
    current_user: User = Depends(get_current_user)
):
    """Restart an MCP server"""
    try:
        success = await mcp_manager.restart_server(server_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to restart server")
        
        return {"message": f"Server {server_id} restarted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to restart server {server_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to restart server: {str(e)}")


@router.delete("/servers/{server_id}")
async def unregister_server(
    server_id: str,
    mcp_manager: MCPServerManager = Depends(get_mcp_manager),
    current_user: User = Depends(get_current_user)
):
    """Unregister an MCP server"""
    try:
        success = await mcp_manager.unregister_server(server_id)
        if not success:
            raise HTTPException(status_code=404, detail="Server not found")
        
        return {"message": f"Server {server_id} unregistered successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to unregister server {server_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to unregister server: {str(e)}")


@router.get("/tools")
async def list_tools(
    mcp_manager: MCPServerManager = Depends(get_mcp_manager),
    current_user: User = Depends(get_current_user)
):
    """List all available MCP tools"""
    try:
        tools = []
        servers = await mcp_manager.list_servers()
        
        for server in servers:
            server_tools = server.get("tools", [])
            for tool in server_tools:
                tools.append({
                    "name": tool,
                    "server_id": server["id"],
                    "server_name": server["name"]
                })
        
        return {"tools": tools}
        
    except Exception as e:
        logger.error(f"Failed to list MCP tools: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list tools: {str(e)}")


@router.get("/tools/{tool_name}")
async def get_tool_info(
    tool_name: str,
    mcp_manager: MCPServerManager = Depends(get_mcp_manager),
    current_user: User = Depends(get_current_user)
):
    """Get information about a specific tool"""
    try:
        tool = await mcp_manager.get_tool(tool_name)
        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")
        
        server_id = await mcp_manager.find_tool_server(tool_name)
        
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema,
            "server_id": server_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get tool info for {tool_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get tool info: {str(e)}")


@router.post("/tools/call")
async def call_tool(
    request: CallToolRequest,
    mcp_manager: MCPServerManager = Depends(get_mcp_manager),
    current_user: User = Depends(require_mcp_execute)
):
    """Call an MCP tool"""
    try:
        result = await mcp_manager.call_tool(request.tool_name, request.arguments)
        
        return {
            "tool_name": request.tool_name,
            "arguments": request.arguments,
            "result": result,
            "executed_by": current_user.username
        }
        
    except Exception as e:
        logger.error(f"Failed to call tool {request.tool_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to call tool: {str(e)}")


@router.get("/status")
async def get_mcp_status(
    mcp_manager: MCPServerManager = Depends(get_mcp_manager),
    current_user: User = Depends(get_current_user)
):
    """Get overall MCP system status"""
    try:
        servers = await mcp_manager.list_servers()
        connected_count = await mcp_manager.get_connected_servers_count()

        return {
            "total_servers": len(servers),
            "connected_servers": connected_count,
            "servers": servers
        }

    except Exception as e:
        logger.error(f"Failed to get MCP status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get MCP status: {str(e)}")


@router.get("/discovery/status")
async def get_discovery_status():
    """Get MCP server auto-discovery status and results"""
    try:
        from ...api.main import app_state

        discovery_results = app_state.get("mcp_discovery_results", {})

        if not discovery_results:
            return {
                "discovery_enabled": False,
                "status": "not_run",
                "message": "Auto-discovery has not been run"
            }

        return {
            "discovery_enabled": True,
            "status": "completed" if discovery_results.get("success", False) else "failed",
            "results": discovery_results,
            "summary": {
                "servers_discovered": discovery_results.get("servers_discovered", 0),
                "servers_registered": discovery_results.get("total_servers_registered", 0),
                "priority_servers": discovery_results.get("priority_servers_registered", 0),
                "startup_time_ms": discovery_results.get("startup_time_ms", 0)
            }
        }

    except Exception as e:
        logger.error(f"Failed to get discovery status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get discovery status: {str(e)}")


@router.get("/discovery/servers")
async def get_discovered_servers():
    """Get list of all discovered MCP servers (from cache)"""
    try:
        import json
        from pathlib import Path

        cache_file = Path("./data/mcp_cache/discovered_servers.json")
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                discovered_servers = json.load(f)

            # Group by category
            categories = {}
            for server_name, server_data in discovered_servers.items():
                category = server_data.get("category", "unknown")
                if category not in categories:
                    categories[category] = []
                categories[category].append({
                    "name": server_name,
                    "description": server_data.get("description", ""),
                    "author": server_data.get("author", ""),
                    "verified": server_data.get("verified", False),
                    "capabilities": server_data.get("capabilities", []),
                    "tools": server_data.get("tools", [])
                })

            return {
                "total_discovered": len(discovered_servers),
                "categories": categories,
                "cache_file": str(cache_file)
            }
        else:
            return {
                "total_discovered": 0,
                "categories": {},
                "message": "No discovery cache found"
            }

    except Exception as e:
        logger.error(f"Failed to get discovered servers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get discovered servers: {str(e)}")


# Global installation status tracking
_installation_status: Dict[str, InstallationStatus] = {}


async def install_mcp_server_background(server_name: str, source_type: str, source_url: str, version: Optional[str] = None):
    """Background task to install MCP server"""
    try:
        # Update status to installing
        _installation_status[server_name] = InstallationStatus(
            server_name=server_name,
            status="installing",
            progress=10,
            message="Starting installation..."
        )

        # Create installation directory
        install_dir = Path("./mcp_servers") / server_name
        install_dir.mkdir(parents=True, exist_ok=True)

        # Update progress
        _installation_status[server_name].progress = 30
        _installation_status[server_name].message = "Installing dependencies..."

        if source_type == "npm":
            # Install npm package
            if source_url.startswith("@"):
                # Scoped package
                npm_cmd = ["npm", "install", "-g", source_url]
            else:
                # Regular package
                npm_cmd = ["npm", "install", "-g", source_url]

            if version:
                npm_cmd[-1] += f"@{version}"

            process = await asyncio.create_subprocess_exec(
                *npm_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=install_dir
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                raise RuntimeError(f"npm install failed: {stderr.decode()}")

            # Update progress
            _installation_status[server_name].progress = 80
            _installation_status[server_name].message = "Configuring server..."

            # Determine the executable command
            if source_url == "@modelcontextprotocol/server-filesystem":
                command = ["npx", "@modelcontextprotocol/server-filesystem"]
            elif source_url == "@modelcontextprotocol/server-postgres":
                command = ["npx", "@modelcontextprotocol/server-postgres"]
            elif source_url == "@modelcontextprotocol/server-github":
                command = ["npx", "@modelcontextprotocol/server-github"]
            elif source_url == "@notionhq/notion-mcp-server":
                command = ["npx", "@notionhq/notion-mcp-server"]
            else:
                # Generic npm package
                command = ["npx", source_url]

            # Register the server with the MCP manager
            from ...api.main import get_mcp_manager as get_main_mcp_manager
            mcp_manager = await get_main_mcp_manager()
            config = MCPServerConfig(
                name=server_name,
                command=command,
                capabilities=[],  # Will be discovered when server starts
                transport="stdio",
                config={
                    "installed": True,
                    "install_path": str(install_dir),
                    "source_type": source_type,
                    "source_url": source_url,
                    "version": version or "latest"
                },
                auto_start=True,
                restart_on_failure=True,
                max_restarts=3
            )

            await mcp_manager.register_server(config)

            # Complete installation
            _installation_status[server_name].status = "completed"
            _installation_status[server_name].progress = 100
            _installation_status[server_name].message = "Installation completed successfully"
            _installation_status[server_name].install_path = str(install_dir)

        else:
            raise ValueError(f"Unsupported source type: {source_type}")

    except Exception as e:
        logger.error(f"Failed to install MCP server {server_name}: {str(e)}")
        _installation_status[server_name].status = "failed"
        _installation_status[server_name].message = f"Installation failed: {str(e)}"


@router.post("/servers/install")
async def install_server(
    request: InstallServerRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Install an MCP server"""
    try:
        # Determine source URL if not provided
        source_url = request.source_url
        if not source_url:
            if request.source_type == "npm":
                # Map common server names to npm packages
                npm_packages = {
                    "filesystem": "@modelcontextprotocol/server-filesystem",
                    "postgres": "@modelcontextprotocol/server-postgres",
                    "github": "@modelcontextprotocol/server-github",
                    "brave-search": "@modelcontextprotocol/server-brave-search",
                    "notion": "@notionhq/notion-mcp-server",
                    "puppeteer": "puppeteer-mcp-server",
                    "figma": "figma-mcp"
                }
                source_url = npm_packages.get(request.server_name, request.server_name)
            else:
                raise ValueError("source_url is required for non-npm installations")

        # Initialize installation status
        _installation_status[request.server_name] = InstallationStatus(
            server_name=request.server_name,
            status="pending",
            progress=0,
            message="Installation queued..."
        )

        # Start background installation
        background_tasks.add_task(
            install_mcp_server_background,
            request.server_name,
            request.source_type,
            source_url,
            request.version
        )

        return {
            "message": f"Installation of {request.server_name} started",
            "server_name": request.server_name,
            "status": "pending"
        }

    except Exception as e:
        logger.error(f"Failed to start installation of {request.server_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start installation: {str(e)}")


@router.get("/servers/install/{server_name}/status")
async def get_installation_status(
    server_name: str,
    current_user: User = Depends(get_current_user)
):
    """Get installation status for a specific server"""
    if server_name not in _installation_status:
        raise HTTPException(status_code=404, detail="Installation not found")

    return _installation_status[server_name]


@router.get("/servers/install/status")
async def get_all_install_status(
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get installation status for all servers"""
    return {
        "message": "Installation status tracking not implemented",
        "servers": {}
    }


@router.get("/marketplace/featured")
async def get_featured_servers(
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get featured MCP servers from the marketplace"""
    try:
        # Load discovery data
        discovery_cache_path = Path("data/mcp_cache/discovered_servers.json")
        if not discovery_cache_path.exists():
            return {"featured": [], "total": 0}
        
        import json
        with open(discovery_cache_path, 'r') as f:
            discovery_data = json.load(f)
        
        # Get featured servers (prioritize verified Cloudflare and official servers)
        featured = []
        
        # Add Cloudflare servers as featured
        for category, servers in discovery_data.get("categories", {}).items():
            for server in servers:
                if server.get("author") == "Cloudflare" and server.get("verified", False):
                    featured.append({
                        **server,
                        "category": category,
                        "featured_reason": "Official Cloudflare integration"
                    })
        
        # Add other verified servers
        for category, servers in discovery_data.get("categories", {}).items():
            for server in servers:
                if (server.get("verified", False) and 
                    server.get("author") != "Cloudflare" and 
                    len(featured) < 8):
                    featured.append({
                        **server,
                        "category": category,
                        "featured_reason": "Verified and reliable"
                    })
        
        return {
            "featured": featured[:8],  # Limit to 8 featured servers
            "total": len(featured),
            "last_updated": discovery_data.get("last_updated", "unknown")
        }
        
    except Exception as e:
        logger.error(f"Error getting featured servers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get featured servers: {str(e)}")


@router.get("/marketplace/popular")
async def get_popular_servers(
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get popular MCP servers from the marketplace"""
    try:
        # Load discovery data
        discovery_cache_path = Path("data/mcp_cache/discovered_servers.json")
        if not discovery_cache_path.exists():
            return {"popular": [], "total": 0}
        
        import json
        with open(discovery_cache_path, 'r') as f:
            discovery_data = json.load(f)
        
        # Get popular servers (based on category and capabilities)
        popular = []
        
        # Prioritize servers with many capabilities
        for category, servers in discovery_data.get("categories", {}).items():
            for server in servers:
                capability_count = len(server.get("capabilities", []))
                tool_count = len(server.get("tools", []))
                popularity_score = capability_count + tool_count
                
                if popularity_score > 0:
                    popular.append({
                        **server,
                        "category": category,
                        "popularity_score": popularity_score,
                        "usage_estimate": "Medium" if popularity_score > 3 else "Low"
                    })
        
        # Sort by popularity score
        popular.sort(key=lambda x: x.get("popularity_score", 0), reverse=True)
        
        return {
            "popular": popular[:10],  # Top 10 popular servers
            "total": len(popular),
            "last_updated": discovery_data.get("last_updated", "unknown")
        }
        
    except Exception as e:
        logger.error(f"Error getting popular servers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get popular servers: {str(e)}")


@router.get("/marketplace/categories")
async def get_marketplace_categories(
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get available server categories in the marketplace"""
    try:
        # Load discovery data
        discovery_cache_path = Path("data/mcp_cache/discovered_servers.json")
        if not discovery_cache_path.exists():
            return {"categories": {}, "total": 0}
        
        import json
        with open(discovery_cache_path, 'r') as f:
            discovery_data = json.load(f)
        
        categories = discovery_data.get("categories", {})
        
        # Enhance category data with counts and descriptions
        enhanced_categories = {}
        category_descriptions = {
            "cloud": "Cloud services and infrastructure tools",
            "web": "Web scraping and browser automation",
            "development": "Software development and version control",
            "database": "Database management and operations",
            "nlp": "Natural language processing and search",
            "academic_research": "Research and academic paper tools",
            "coding": "Code editing and development environments",
            "web_ui": "User interface and dashboard tools",
            "npm": "NPM package ecosystem servers"
        }
        
        for category, servers in categories.items():
            enhanced_categories[category] = {
                "name": category.replace("_", " ").title(),
                "description": category_descriptions.get(category, f"{category.replace('_', ' ').title()} related tools"),
                "server_count": len(servers),
                "verified_count": len([s for s in servers if s.get("verified", False)]),
                "servers": servers[:5]  # Preview of first 5 servers
            }
        
        return {
            "categories": enhanced_categories,
            "total_categories": len(enhanced_categories),
            "total_servers": sum(len(servers) for servers in categories.values()),
            "last_updated": discovery_data.get("last_updated", "unknown")
        }
        
    except Exception as e:
        logger.error(f"Error getting marketplace categories: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get marketplace categories: {str(e)}")


@router.get("/marketplace/search")
async def search_marketplace(
    q: str,
    category: Optional[str] = None,
    verified_only: bool = False,
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Search servers in the marketplace"""
    try:
        # Load discovery data
        discovery_cache_path = Path("data/mcp_cache/discovered_servers.json")
        if not discovery_cache_path.exists():
            return {"results": [], "total": 0, "query": q}
        
        import json
        with open(discovery_cache_path, 'r') as f:
            discovery_data = json.load(f)
        
        results = []
        query_lower = q.lower()
        
        # Search through all servers
        for cat, servers in discovery_data.get("categories", {}).items():
            # Skip category if filter specified
            if category and cat != category:
                continue
                
            for server in servers:
                # Skip if verified_only filter is set
                if verified_only and not server.get("verified", False):
                    continue
                
                # Search in name, description, capabilities, and tools
                searchable_text = " ".join([
                    server.get("name", ""),
                    server.get("description", ""),
                    " ".join(server.get("capabilities", [])),
                    " ".join(server.get("tools", [])),
                    server.get("author", "")
                ]).lower()
                
                if query_lower in searchable_text:
                    results.append({
                        **server,
                        "category": cat,
                        "match_score": searchable_text.count(query_lower)
                    })
        
        # Sort by match score
        results.sort(key=lambda x: x.get("match_score", 0), reverse=True)
        
        return {
            "results": results,
            "total": len(results),
            "query": q,
            "filters": {
                "category": category,
                "verified_only": verified_only
            },
            "last_updated": discovery_data.get("last_updated", "unknown")
        }
        
    except Exception as e:
        logger.error(f"Error searching marketplace: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to search marketplace: {str(e)}")


@router.get("/servers/public")
async def list_servers_public(
    mcp_manager: MCPServerManager = Depends(get_mcp_manager)
):
    """List all registered MCP servers (public endpoint)"""
    try:
        servers = await mcp_manager.list_servers()
        return {"servers": servers}
        
    except Exception as e:
        logger.error(f"Failed to list MCP servers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list servers: {str(e)}")
