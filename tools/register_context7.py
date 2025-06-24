#!/usr/bin/env python3
"""
Register and start Context7 MCP server
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.mcp.server_registry import MCPServerConfig, MCPServerManager
from src.config.settings import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def register_context7():
    """Register and start the Context7 MCP server"""
    try:
        # Initialize settings and MCP manager
        settings = get_settings()
        mcp_manager = MCPServerManager(settings)
        await mcp_manager.start()        # Create Context7 server configuration
        context7_config = MCPServerConfig(
            name="context7",
            command=["D:\\nodejs\\npx.cmd", "-y", "@upstash/context7-mcp"],
            capabilities=[
                "library-documentation",
                "code-examples", 
                "api-reference",
                "version-specific-docs"
            ],
            transport="stdio",
            config={
                "category": "development",
                "author": "Upstash",
                "verified": True,
                "description": "Up-to-date code documentation and examples for any library",
                "tools": [
                    "resolve-library-id",
                    "get-library-docs"
                ],
                "installation_required": False,  # npx handles this
                "official": True
            },
            auto_start=True,  # We want this to start automatically
            restart_on_failure=True,  # Critical server
            max_restarts=3,
            timeout=30
        )

        # Register the server
        logger.info("Registering Context7 MCP server...")
        server_id = await mcp_manager.register_server(context7_config)
        logger.info(f"✅ Context7 registered with ID: {server_id}")

        # Start the server
        logger.info("Starting Context7 MCP server...")
        success = await mcp_manager.start_server(server_id)
        
        if success:
            logger.info("🚀 Context7 MCP server started successfully!")
            
            # Test the server by calling one of its tools
            logger.info("Testing Context7 server...")
            try:
                # Get server status
                status = await mcp_manager.get_server_status(server_id)
                logger.info(f"Context7 status: {status}")
                
                # List available tools
                tools = await mcp_manager.get_server_tools(server_id)
                logger.info(f"Context7 tools: {[tool.name for tool in tools] if tools else 'No tools available yet'}")
                
            except Exception as e:
                logger.warning(f"Could not test Context7 immediately: {e}")
                logger.info("This is normal - server may still be initializing")
        else:
            logger.error("❌ Failed to start Context7 MCP server")
            return False

        return True

    except Exception as e:
        logger.error(f"Failed to register Context7: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(register_context7())
    if success:
        print("\n🎉 Context7 MCP server is now available!")
        print("You can now use 'use context7' in prompts to get live documentation!")
    else:
        print("\n💥 Failed to register Context7 MCP server")
        sys.exit(1)
