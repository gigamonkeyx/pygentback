#!/usr/bin/env python3
"""
Test Enhanced MCP Registry with Tool Discovery

This script tests the enhanced MCP registry to verify it properly discovers
and stores tool metadata according to the MCP specification.
"""

import asyncio
import json
import logging
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.mcp.enhanced_registry import EnhancedMCPServerRegistry, MCPToolDefinition
from src.mcp.server.config import MCPServerConfig, MCPServerType, MCPTransportType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_enhanced_registry():
    """Test the enhanced MCP registry with tool discovery"""
    logger.info("Starting enhanced MCP registry test")
    
    # Create enhanced registry
    registry = EnhancedMCPServerRegistry()
    
    try:        # Test configuration - using Context7 as it's known to work
        context7_config = MCPServerConfig(
            id="context7-test",
            name="Context7 Documentation",
            command=["npx", "-y", "@upstash/context7-mcp"],
            server_type=MCPServerType.CUSTOM,
            transport=MCPTransportType.STDIO,
            capabilities=["tools"],
            auto_start=True
        )
        
        logger.info("Testing Context7 server registration and tool discovery...")
        
        # Register and discover server
        registration = await registry.register_and_discover_server(context7_config)
        
        logger.info(f"Registration status: {registration.status}")
        logger.info(f"Server active: {registry.is_server_active('context7-test')}")
        
        # Get discovered tools
        tools = await registry.get_server_tools("context7-test")
        logger.info(f"Discovered {len(tools)} tools:")
        
        for tool in tools:
            logger.info(f"  - {tool.name}: {tool.description}")
        
        # Get capabilities summary
        summary = await registry.get_capabilities_summary()
        logger.info(f"Capabilities summary:")
        logger.info(f"  Total servers: {summary['total_servers']}")
        logger.info(f"  Active servers: {summary['active_servers']}")
        logger.info(f"  Total tools: {summary['total_tools']}")
        
        # Test tool lookup
        if tools:
            first_tool = tools[0]
            found_tool = await registry.get_tool_by_name(first_tool.name)
            if found_tool:
                logger.info(f"Successfully found tool by name: {found_tool.name}")
            else:
                logger.error(f"Failed to find tool by name: {first_tool.name}")
        
        # Save results to file
        results = {
            "test_timestamp": str(asyncio.get_event_loop().time()),
            "registry_summary": summary,
            "discovered_tools": [tool.to_dict() for tool in tools],
            "test_successful": len(tools) > 0
        }
        
        with open("enhanced_registry_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Test results saved to enhanced_registry_test_results.json")
        
        if len(tools) > 0:
            logger.info("✅ Tool discovery test PASSED")
            return True
        else:
            logger.error("❌ Tool discovery test FAILED - no tools discovered")
            return False
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        try:
            await registry.shutdown()
        except Exception as e:
            logger.error(f"Error during registry shutdown: {e}")


async def test_mcp_availability():
    """Test if MCP SDK is available and working"""
    logger.info("Testing MCP SDK availability...")
    
    try:
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client
        from mcp.types import Tool
        logger.info("✅ MCP SDK is available")
        return True
    except ImportError as e:
        logger.error(f"❌ MCP SDK not available: {e}")
        return False


async def main():
    """Main test function"""
    print("=" * 60)
    print("Enhanced MCP Registry Tool Discovery Test")
    print("=" * 60)
    
    # Test MCP availability first
    mcp_available = await test_mcp_availability()
    if not mcp_available:
        print("❌ Cannot run test - MCP SDK not available")
        sys.exit(1)
    
    # Test enhanced registry
    success = await test_enhanced_registry()
    
    if success:
        print("\n✅ All tests passed! Tool discovery is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Tests failed! Tool discovery needs debugging.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
