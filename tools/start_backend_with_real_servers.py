#!/usr/bin/env python3
"""
Start backend with real MCP servers pre-loaded
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def main():
    print("Starting backend with real MCP servers...")
    
    # First load and start the real MCP servers
    from src.mcp.server_registry import MCPServerManager
    from src.config.settings import get_settings
    from src.mcp.real_server_loader import load_real_mcp_servers
    
    settings = get_settings()
    mcp_manager = MCPServerManager(settings)
    await mcp_manager.start()
    
    # Load real servers first
    print("Loading real MCP servers...")
    result = await load_real_mcp_servers(mcp_manager)
    print(f"Real MCP servers loaded: {result}")
    
    # Now start the main app
    import uvicorn
    from src.api.main import create_app
    
    # Set environment variable to skip auto-discovery
    os.environ["SKIP_MCP_AUTO_DISCOVERY"] = "true"
    
    app = create_app()
    
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False  # Disable reload to prevent constant restarting
    )
    
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
