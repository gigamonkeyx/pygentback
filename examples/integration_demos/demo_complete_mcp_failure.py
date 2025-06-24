#!/usr/bin/env python3
"""
Demo: What happens when ALL MCP fails
Shows the complete fallback hierarchy in action
"""

import asyncio
import logging
from demo_smart_mcp_fallbacks import SmartMCPRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demo_complete_mcp_failure():
    """Show what happens when ALL MCP options fail."""
    
    print("🚨 COMPLETE MCP FAILURE DEMO")
    print("=" * 50)
    
    registry = SmartMCPRegistry()
    
    # Register native fallbacks for some tools
    registry.native_tool_registry["create_file"] = registry._native_create_file
    registry.native_tool_registry["read_file"] = registry._native_read_file
    # NOTE: NO native fallback for analyze_data
    
    # Override to make ALL MCP fail
    async def failing_mcp(*args, **kwargs):
        raise Exception("All MCP servers down!")
    
    registry._try_primary_mcp = failing_mcp
    registry._try_alternative_mcp = failing_mcp  
    registry._try_degraded_mcp = failing_mcp
    
    # Register tools
    registry.register_tool_with_smart_fallbacks("create_file")
    registry.register_tool_with_smart_fallbacks("read_file") 
    registry.register_tool_with_smart_fallbacks("analyze_data")
    
    print("⚠️ Simulating: ALL MCP servers are down!")
    
    # Test 1: Tool WITH native fallback
    print("\n📝 Test 1: create_file (HAS native fallback)")
    result = await registry.execute_mcp_tool("create_file", {
        "path": "emergency_file.txt",
        "content": "Created during MCP outage"
    })
    print(f"Result: {result}")
    print(f"Success: {result['success']}")
    print(f"Warning: {result.get('warning', 'None')}")
    
    # Test 2: Tool WITHOUT native fallback
    print("\n🧠 Test 2: analyze_data (NO native fallback)")
    result = await registry.execute_mcp_tool("analyze_data", {
        "data": [1, 2, 3, 4, 5],
        "analysis_type": "statistical"
    })
    print(f"Result: {result}")
    print(f"Success: {result['success']}")
    print(f"Suggestions: {result.get('suggestions', [])}")
    
    print("\n💡 WHAT THIS DEMONSTRATES:")
    print("✅ Native fallbacks work when MCP completely fails")
    print("❌ Tools without fallbacks give helpful error messages")
    print("⚠️ Agent knows when it's using emergency fallbacks")
    print("🎯 Agent will prefer MCP when it comes back online")

if __name__ == "__main__":
    asyncio.run(demo_complete_mcp_failure())
