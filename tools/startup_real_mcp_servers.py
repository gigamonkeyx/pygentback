#!/usr/bin/env python3
"""
Startup script to load real MCP servers after the backend is running.
This is a workaround to replace auto-discovered mock servers with real ones.
"""

import asyncio
import logging

async def wait_for_backend_and_load_real_servers():
    """Wait for backend to start, then load real MCP servers."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Wait for backend to be fully started
    logger.info("Waiting for backend to start...")
    await asyncio.sleep(10)
    
    # Now run our MCP server update
    logger.info("Loading real MCP servers...")
    
    try:
        # Import and run the update script
        import sys
        sys.path.append('.')
        
        from update_mcp_servers import main as update_main
        await update_main()
        
        logger.info("Real MCP servers loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load real MCP servers: {e}")

if __name__ == "__main__":
    asyncio.run(wait_for_backend_and_load_real_servers())
