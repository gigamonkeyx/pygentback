#!/usr/bin/env python3
"""
Simple MCP Tool Discovery Test

This script tests basic MCP tool discovery functionality using the Context7 server
to validate that tool discovery works according to the MCP specification.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging to avoid Unicode issues
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('tool_discovery_test.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.types import Tool
    MCP_AVAILABLE = True
    logger.info("MCP SDK is available")
except ImportError as e:
    MCP_AVAILABLE = False
    logger.error(f"MCP SDK not available: {e}")


async def test_context7_tool_discovery():
    """Test tool discovery with Context7 MCP server"""
    if not MCP_AVAILABLE:
        logger.error("Cannot test - MCP SDK not available")
        return False
    
    logger.info("Starting Context7 tool discovery test")
    
    try:
        # Create server parameters for Context7
        server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@upstash/context7-mcp"],
            env={}
        )
        
        logger.info("Creating MCP client session...")
        session = await stdio_client(server_params)
        
        logger.info("Initializing session...")
        await session.initialize()
        
        # Get server info
        server_info = session.server_info
        logger.info(f"Connected to server: {server_info.name} v{server_info.version}")
        
        # Check capabilities
        if hasattr(server_info, 'capabilities') and hasattr(server_info.capabilities, 'tools'):
            logger.info("Server supports tools capability")
        else:
            logger.warning("Server may not support tools")
        
        # Discover tools using tools/list
        logger.info("Discovering tools via tools/list...")
        tools_result = await session.list_tools()
        
        discovered_tools = []
        for tool in tools_result.tools:
            tool_info = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
                "annotations": getattr(tool, 'annotations', None)
            }
            discovered_tools.append(tool_info)
            logger.info(f"Discovered tool: {tool.name} - {tool.description}")
        
        logger.info(f"Total tools discovered: {len(discovered_tools)}")
        
        # Test calling a tool if available
        if discovered_tools:
            first_tool = discovered_tools[0]
            logger.info(f"Testing tool call: {first_tool['name']}")
            
            # For Context7, we know the tools usually require library name
            if first_tool['name'] == 'resolve-library-id':
                try:
                    result = await session.call_tool(
                        first_tool['name'], 
                        {"library": "fastapi"}
                    )
                    logger.info(f"Tool call successful: {result.content[0].text[:100]}...")
                except Exception as e:
                    logger.error(f"Tool call failed: {e}")
        
        # Save results
        results = {
            "test_timestamp": str(asyncio.get_event_loop().time()),
            "server_info": {
                "name": server_info.name,
                "version": server_info.version
            },
            "tools_discovered": len(discovered_tools),
            "tools": discovered_tools,
            "test_successful": len(discovered_tools) > 0
        }
        
        with open("simple_tool_discovery_results.json", "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Cleanup
        await session.close()
        
        logger.info(f"Test completed. Results saved to simple_tool_discovery_results.json")
        
        if len(discovered_tools) > 0:
            logger.info("SUCCESS: Tool discovery working correctly!")
            return True
        else:
            logger.error("FAILED: No tools discovered")
            return False
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function"""
    print("=" * 60)
    print("Simple MCP Tool Discovery Test")
    print("=" * 60)
    
    success = await test_context7_tool_discovery()
    
    if success:
        print("\nSUCCESS: Tool discovery is working correctly!")
        print("The MCP specification requires clients to call tools/list to discover tools.")
        print("This test confirms that the basic tool discovery mechanism works.")
        sys.exit(0)
    else:
        print("\nFAILED: Tool discovery test failed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
