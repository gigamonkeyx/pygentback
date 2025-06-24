"""
MCP Server Auto-Discovery Integration

This module integrates MCP server discovery into PyGent Factory's startup process,
automatically discovering and registering available MCP servers.
"""

import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from .server_registry import MCPServerConfig  # Use legacy config with to_modular_config()
from .server.config import MCPServerType
from .server.manager import MCPServerManager

logger = logging.getLogger(__name__)


class MCPAutoDiscovery:
    """
    Auto-discovery system for MCP servers during PyGent Factory startup.
    
    This class handles:
    - Loading discovered servers from cache
    - Auto-registering priority servers
    - Integrating with the MCP server manager
    """
    
    def __init__(self, mcp_manager: MCPServerManager, cache_dir: str = "./data/mcp_cache"):
        self.mcp_manager = mcp_manager
        self.cache_dir = Path(cache_dir)
        self.discovered_servers: Dict[str, Dict[str, Any]] = {}
        self.auto_registered_servers: List[str] = []
        
        # Priority servers to register first
        self.priority_servers = [
            "filesystem",
            "@modelcontextprotocol/server-filesystem",
            "brave-search",
            "postgres", 
            "github",
            "@notionhq/notion-mcp-server",
            "puppeteer-mcp-server"
        ]
    
    async def load_discovery_cache(self) -> bool:
        """Load discovered servers from cache"""
        try:
            cache_file = self.cache_dir / "discovered_servers.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    self.discovered_servers = json.load(f)
                
                logger.info(f"Loaded {len(self.discovered_servers)} MCP servers from discovery cache")
                return True
            else:
                logger.warning("No MCP discovery cache found")
                return False
        
        except Exception as e:
            logger.error(f"Failed to load MCP discovery cache: {e}")
            return False
    
    def _create_server_config(self, server_name: str, server_data: Dict[str, Any]) -> Optional[MCPServerConfig]:
        """Create MCP server configuration from discovery data"""
        try:
            # Create configuration using legacy MCPServerConfig format
            # NOTE: Using install_command as placeholder - servers need actual installation first
            config = MCPServerConfig(
                name=server_data["name"],
                command=server_data["install_command"],  # This is actually an install command, not run command
                capabilities=server_data.get("capabilities", []),
                transport="stdio",  # Default transport
                config={
                    "category": server_data.get("category", "unknown"),
                    "author": server_data.get("author", "unknown"),
                    "verified": server_data.get("verified", False),
                    "discovered": True,
                    "description": server_data.get("description", ""),
                    "tools": server_data.get("tools", []),
                    "installation_required": True  # Mark as needing installation
                },
                auto_start=False,  # Don't auto-start until properly installed
                restart_on_failure=False,  # Don't restart uninstalled servers
                max_restarts=0,  # No restarts for discovered servers
                timeout=30
            )

            return config

        except Exception as e:
            logger.error(f"Failed to create config for server {server_name}: {e}")
            return None
    
    async def auto_register_priority_servers(self) -> int:
        """Auto-register priority MCP servers"""
        if not self.discovered_servers:
            logger.warning("No discovered servers available for auto-registration")
            return 0
        
        logger.info("Auto-registering priority MCP servers...")
        registered_count = 0
        
        for server_name in self.priority_servers:
            if server_name in self.discovered_servers:
                try:
                    server_data = self.discovered_servers[server_name]
                    config = self._create_server_config(server_name, server_data)
                    
                    if config:
                        # Register with MCP manager
                        server_id = await self.mcp_manager.register_server(config)
                        self.auto_registered_servers.append(server_id)
                        registered_count += 1
                        
                        logger.info(f"Auto-registered priority server: {server_name}")
                    else:
                        logger.warning(f"Failed to create config for priority server: {server_name}")

                except Exception as e:
                    logger.error(f"Failed to register priority server {server_name}: {e}")

        logger.info(f"Auto-registered {registered_count} priority MCP servers")
        return registered_count
    
    async def auto_register_additional_servers(self, max_additional: int = 5) -> int:
        """Auto-register additional non-priority servers"""
        logger.info(f"Auto-registering up to {max_additional} additional MCP servers...")
        
        registered_count = 0
        registered_names = set(self.priority_servers)
        
        # Register verified community servers first
        for server_name, server_data in self.discovered_servers.items():
            if (server_name not in registered_names and 
                server_data.get("verified", False) and 
                registered_count < max_additional):
                
                try:
                    config = self._create_server_config(server_name, server_data)
                    if config:
                        server_id = await self.mcp_manager.register_server(config)
                        self.auto_registered_servers.append(server_id)
                        registered_count += 1
                        registered_names.add(server_name)
                        
                        logger.info(f"Auto-registered additional server: {server_name}")

                except Exception as e:
                    logger.error(f"Failed to register additional server {server_name}: {e}")
        
        # Register interesting npm servers if we have room
        interesting_servers = [
            "@notionhq/notion-mcp-server",
            "puppeteer-mcp-server", 
            "figma-mcp"
        ]
        
        for server_name in interesting_servers:
            if (server_name in self.discovered_servers and 
                server_name not in registered_names and 
                registered_count < max_additional):
                
                try:
                    server_data = self.discovered_servers[server_name]
                    config = self._create_server_config(server_name, server_data)
                    if config:
                        server_id = await self.mcp_manager.register_server(config)
                        self.auto_registered_servers.append(server_id)
                        registered_count += 1
                        registered_names.add(server_name)
                        
                        logger.info(f"Auto-registered interesting server: {server_name}")

                except Exception as e:
                    logger.error(f"Failed to register interesting server {server_name}: {e}")

        logger.info(f"Auto-registered {registered_count} additional MCP servers")
        return registered_count
    
    async def run_auto_discovery(self) -> Dict[str, Any]:
        """Run the complete auto-discovery process"""
        logger.info("Starting MCP server auto-discovery...")
        
        start_time = datetime.now()
        results = {
            "cache_loaded": False,
            "servers_discovered": 0,
            "priority_servers_registered": 0,
            "additional_servers_registered": 0,
            "total_servers_registered": 0,
            "auto_registered_servers": [],
            "startup_time_ms": 0,
            "success": False
        }
        
        try:
            # Step 1: Load discovery cache
            cache_loaded = await self.load_discovery_cache()
            results["cache_loaded"] = cache_loaded
            results["servers_discovered"] = len(self.discovered_servers)
            
            if cache_loaded and self.discovered_servers:
                # Step 2: Auto-register priority servers
                priority_count = await self.auto_register_priority_servers()
                results["priority_servers_registered"] = priority_count
                
                # Step 3: Auto-register additional servers
                additional_count = await self.auto_register_additional_servers()
                results["additional_servers_registered"] = additional_count
                
                # Step 4: Calculate totals
                results["total_servers_registered"] = priority_count + additional_count
                results["auto_registered_servers"] = self.auto_registered_servers.copy()
                results["success"] = True
                
                logger.info(f"Auto-discovery completed: {results['total_servers_registered']} servers registered")
            else:
                logger.warning("No servers available for auto-registration")
            
            # Calculate timing
            end_time = datetime.now()
            startup_time = (end_time - start_time).total_seconds() * 1000
            results["startup_time_ms"] = round(startup_time, 2)
            
        except Exception as e:
            logger.error(f"Auto-discovery failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def get_discovery_summary(self) -> Dict[str, Any]:
        """Get a summary of discovered servers"""
        if not self.discovered_servers:
            return {"total": 0, "categories": {}, "verified": 0}
        
        categories = {}
        verified_count = 0
        
        for server_data in self.discovered_servers.values():
            category = server_data.get("category", "unknown")
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
            
            if server_data.get("verified", False):
                verified_count += 1
        
        return {
            "total": len(self.discovered_servers),
            "categories": categories,
            "verified": verified_count,
            "priority_available": sum(1 for name in self.priority_servers if name in self.discovered_servers)
        }


async def initialize_mcp_auto_discovery(mcp_manager: MCPServerManager) -> Dict[str, Any]:
    """
    Initialize MCP auto-discovery during PyGent Factory startup.
    
    This function should be called during application startup to automatically
    discover and register MCP servers.
    
    Args:
        mcp_manager: The MCP server manager instance
        
    Returns:
        Dict containing discovery results and statistics
    """
    logger.info("Initializing MCP auto-discovery...")
    
    try:
        # Create auto-discovery instance
        auto_discovery = MCPAutoDiscovery(mcp_manager)
        
        # Run auto-discovery process
        results = await auto_discovery.run_auto_discovery()
        
        # Log summary
        if results["success"]:
            logger.info(f"MCP auto-discovery successful:")
            logger.info(f"   Servers discovered: {results['servers_discovered']}")
            logger.info(f"   Priority servers registered: {results['priority_servers_registered']}")
            logger.info(f"   Additional servers registered: {results['additional_servers_registered']}")
            logger.info(f"   Total servers registered: {results['total_servers_registered']}")
            logger.info(f"   Startup time: {results['startup_time_ms']}ms")
        else:
            logger.warning("MCP auto-discovery completed with issues")
        
        return results
        
    except Exception as e:
        logger.error(f"MCP auto-discovery initialization failed: {e}")
        return {"success": False, "error": str(e)}
