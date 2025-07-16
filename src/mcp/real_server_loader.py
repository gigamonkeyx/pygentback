"""
MCP Server Configuration Loader

This module loads real MCP server configurations to replace mock/discovered servers.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

from src.mcp.server_registry import MCPServerConfig, MCPServerManager

logger = logging.getLogger(__name__)


async def load_real_mcp_servers(mcp_manager: MCPServerManager) -> Dict[str, Any]:
    """
    Load real MCP server configurations and replace any mock/discovered servers.
    
    Args:
        mcp_manager: The MCP server manager instance
        
    Returns:
        Dict containing load results and statistics
    """
    logger.info("Loading real MCP server configurations...")
    
    # Load the server configurations
    config_file = Path("data/mcp_server_configs.json")
    if not config_file.exists():
        logger.error("data/mcp_server_configs.json not found")
        return {"success": False, "error": "Configuration file not found"}
    
    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        results = {
            "servers_loaded": 0,
            "servers_started": 0,
            "servers_failed": 0,
            "server_details": [],
            "success": False
        }
        
        # Register and start each server
        for server_config in config_data.get("servers", []):
            try:
                # Create MCPServerConfig
                config = MCPServerConfig(
                    id=server_config["id"],
                    name=server_config["name"],
                    command=server_config["command"],
                    capabilities=server_config["capabilities"],
                    transport=server_config["transport"],
                    config=server_config["config"],
                    auto_start=server_config["auto_start"],
                    restart_on_failure=server_config["restart_on_failure"],
                    max_restarts=server_config["max_restarts"],
                    timeout=server_config["timeout"]
                )
                
                # Register the server
                server_id = await mcp_manager.register_server(config)
                results["servers_loaded"] += 1
                logger.info(f"Registered server: {config.name} ({server_id})")
                
                # Start the server if auto_start is enabled
                if config.auto_start:
                    success = await mcp_manager.start_server(server_id)
                    if success:
                        results["servers_started"] += 1
                        logger.info(f"Started server: {config.name}")
                        results["server_details"].append({
                            "id": server_id,
                            "name": config.name,
                            "status": "started"
                        })
                    else:
                        results["servers_failed"] += 1
                        logger.warning(f"Failed to start server: {config.name}")
                        results["server_details"].append({
                            "id": server_id,
                            "name": config.name,
                            "status": "failed"
                        })
                else:
                    results["server_details"].append({
                        "id": server_id,
                        "name": config.name,
                        "status": "registered"
                    })
                
            except Exception as e:
                results["servers_failed"] += 1
                logger.error(f"Failed to process server {server_config.get('name', 'Unknown')}: {e}")
                results["server_details"].append({
                    "name": server_config.get('name', 'Unknown'),
                    "status": "error",
                    "error": str(e)
                })
        
        # Mark as successful if at least one server was loaded
        results["success"] = results["servers_loaded"] > 0
        
        logger.info(f"Real MCP server loading complete: {results['servers_loaded']} loaded, {results['servers_started']} started")
        return results
        
    except Exception as e:
        logger.error(f"Failed to load real MCP servers: {e}")
        return {"success": False, "error": str(e)}


async def replace_discovered_servers_with_real_servers(mcp_manager: MCPServerManager) -> Dict[str, Any]:
    """
    Replace any discovered/mock servers with real working servers.
    
    Args:
        mcp_manager: The MCP server manager instance
        
    Returns:
        Dict containing replacement results and statistics
    """
    logger.info("Replacing discovered servers with real servers...")
    
    try:
        # Get list of currently registered servers
        servers_to_remove = []
        if hasattr(mcp_manager, 'servers'):
            servers_to_remove = list(mcp_manager.servers.keys())
        
        # Remove discovered/mock servers
        removed_count = 0
        for server_id in servers_to_remove:
            try:
                await mcp_manager.unregister_server(server_id)
                removed_count += 1
                logger.info(f"Removed discovered server: {server_id}")
            except Exception as e:
                logger.warning(f"Failed to remove server {server_id}: {e}")
        
        # Load real servers
        load_results = await load_real_mcp_servers(mcp_manager)
        
        # Combine results
        results = {
            "removed_servers": removed_count,
            "load_results": load_results,
            "success": load_results.get("success", False)
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to replace discovered servers: {e}")
        return {"success": False, "error": str(e)}
