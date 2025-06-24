#!/usr/bin/env python3
"""
Check status of all MCP servers
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.mcp.server_registry import MCPServerManager
from src.config.settings import get_settings


async def check_servers():
    """Check the status of all MCP servers"""
    try:
        settings = get_settings()
        manager = MCPServerManager(settings)
        await manager.start()
        servers = await manager.list_servers()
        print(f'Total registered servers: {len(servers)}')
        
        for server in servers:
            status = await manager.get_server_status(server.id)
            print(f'- {server.name}: {status["status"]} (ID: {server.id})')
            
    except Exception as e:
        print(f"Error checking servers: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(check_servers())
