#!/usr/bin/env python3
"""
Update MCP servers via the running backend API
"""

import asyncio
import json
import aiohttp
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def update_mcp_servers_via_api():
    """Update MCP servers by calling the backend API"""
    
    # Load MCP server configs
    config_path = Path(__file__).parent / "mcp_server_configs.json"
    
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return False
        
    with open(config_path, 'r') as f:
        configs = json.load(f)
    
    async with aiohttp.ClientSession() as session:
        # First, check if backend is running
        try:
            async with session.get('http://localhost:8000/health') as response:
                if response.status != 200:
                    logger.error("Backend is not running or not healthy")
                    return False
                logger.info("Backend is running and healthy")
        except Exception as e:
            logger.error(f"Cannot connect to backend: {e}")
            return False
        
        # Get current MCP servers
        try:
            async with session.get('http://localhost:8000/api/mcp/servers') as response:
                if response.status == 200:
                    current_servers = await response.json()
                    logger.info(f"Current servers: {len(current_servers.get('servers', []))}")
                    
                    # Clear existing servers if any
                    for server in current_servers.get('servers', []):
                        try:
                            async with session.delete(f'http://localhost:8000/api/mcp/servers/{server["id"]}') as del_response:
                                if del_response.status == 200:
                                    logger.info(f"Removed server: {server['name']}")
                        except Exception as e:
                            logger.warning(f"Failed to remove server {server.get('name', 'unknown')}: {e}")
                            
        except Exception as e:
            logger.warning(f"Failed to get current servers: {e}")
        
        # Register new servers
        for server_id, server_config in configs.items():
            try:
                # Register server
                payload = {
                    "name": server_config["name"],
                    "command": server_config["command"],
                    "args": server_config.get("args", []),
                    "env": server_config.get("env", {}),
                    "capabilities": server_config.get("capabilities", []),
                    "auto_start": True
                }
                
                async with session.post('http://localhost:8000/api/mcp/servers', json=payload) as response:
                    if response.status == 201:
                        result = await response.json()
                        server_id = result.get('id')
                        logger.info(f"Registered server: {server_config['name']} (ID: {server_id})")
                        
                        # Start server
                        try:
                            async with session.post(f'http://localhost:8000/api/mcp/servers/{server_id}/start') as start_response:
                                if start_response.status == 200:
                                    logger.info(f"Started server: {server_config['name']}")
                                else:
                                    logger.warning(f"Failed to start server {server_config['name']}: {start_response.status}")
                        except Exception as e:
                            logger.warning(f"Failed to start server {server_config['name']}: {e}")
                            
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to register server {server_config['name']}: {response.status} - {error_text}")
                        
            except Exception as e:
                logger.error(f"Failed to register server {server_config['name']}: {e}")
        
        # Check final status
        try:
            await asyncio.sleep(2)  # Give servers time to start
            async with session.get('http://localhost:8000/api/mcp/servers') as response:
                if response.status == 200:
                    final_servers = await response.json()
                    logger.info("Final server status:")
                    for server in final_servers.get('servers', []):
                        status = server.get('status', 'unknown')
                        logger.info(f"  {server['name']}: {status}")
                        
        except Exception as e:
            logger.warning(f"Failed to get final server status: {e}")
    
    return True

if __name__ == "__main__":
    asyncio.run(update_mcp_servers_via_api())
