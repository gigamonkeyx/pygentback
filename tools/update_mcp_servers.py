#!/usr/bin/env python3
"""
MCP Server Configuration Updater

This script updates the MCP server registry with real, working MCP servers
to replace the mock/placeholder configurations.
"""

import asyncio
import json
import logging
from pathlib import Path
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.mcp.server_registry import MCPServerConfig, MCPServerManager
from src.config.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def update_mcp_servers():
    """Update MCP server configurations with real working servers"""
    
    # Load the server configurations
    config_file = Path("mcp_server_configs.json")
    if not config_file.exists():
        logger.error("mcp_server_configs.json not found")
        return False
    
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    
    # Initialize settings and manager
    settings = Settings()
    manager = MCPServerManager(settings)
    
    try:        # Initialize the manager
        await manager.start()
        
        # Clear existing discovered servers (they're just mocks)
        logger.info("Clearing existing mock/placeholder servers...")
        
        # Get all server IDs to remove
        servers_to_remove = []
        if hasattr(manager, 'servers'):
            servers_to_remove = list(manager.servers.keys())
        
        # Remove existing servers
        for server_id in servers_to_remove:
            try:
                await manager.unregister_server(server_id)
                logger.info(f"Removed mock server: {server_id}")
            except Exception as e:
                logger.warning(f"Failed to remove server {server_id}: {e}")
        
        # Register new working servers
        logger.info("Registering working MCP servers...")
        
        for server_config in config_data["servers"]:
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
                server_id = await manager.register_server(config)
                logger.info(f"Registered server: {config.name} ({server_id})")
                
                # Start the server if auto_start is enabled
                if config.auto_start:
                    logger.info(f"Starting server: {config.name}...")
                    success = await manager.start_server(server_id)
                    if success:
                        logger.info(f"Server started successfully: {config.name}")
                    else:
                        logger.warning(f"Failed to start server: {config.name}")
                
            except Exception as e:
                logger.error(f"Failed to register server {server_config['name']}: {e}")
        
        # List all servers
        logger.info("Current server status:")
        if hasattr(manager, 'servers'):
            for server_id, server in manager.servers.items():
                logger.info(f"  {server.config.name}: {server.status.value}")
        
        logger.info("MCP server configuration update complete!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update MCP servers: {e}")
        return False
    
    finally:        # Shutdown the manager
        await manager.stop()


async def main():
    """Main entry point for external scripts."""
    return await update_mcp_servers()


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
