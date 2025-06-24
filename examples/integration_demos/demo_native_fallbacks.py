#!/usr/bin/env python3
"""
Demo: Native Fallbacks for MCP Tools
Shows how agents get reliable tool access even when MCP servers fail
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai.providers.provider_registry import ProviderRegistry

async def demo_native_fallbacks():
    """Demonstrate native fallbacks in action."""
    
    print("🎯 NATIVE FALLBACKS DEMO")
    print("=" * 50)
    
    # Initialize provider registry
    registry = ProviderRegistry()
    await registry.initialize()
    
    # Register native fallbacks
    registry.register_native_fallbacks()
    
    print(f"✅ Registered native fallbacks for {len(registry.native_tool_registry)} tools")
    print("Available fallbacks:", list(registry.native_tool_registry.keys()))
    
    # Demo 1: Create file (will use native fallback since no MCP server)
    print("\n📝 Demo 1: Create File")
    result = await registry.execute_mcp_tool("create_file", {
        "path": "demo_file.txt",
        "content": "Hello from native fallback!"
    })
    print(f"Result: {result}")
    
    # Demo 2: Read file
    print("\n📖 Demo 2: Read File")
    result = await registry.execute_mcp_tool("read_file", {
        "path": "demo_file.txt"
    })
    print(f"Result: {result}")
    
    # Demo 3: List directory
    print("\n📁 Demo 3: List Directory")
    result = await registry.execute_mcp_tool("list_directory", {
        "path": "."
    })
    print(f"Result: {result}")
    
    # Show tool status
    print("\n📊 Tool Status:")
    status = await registry.get_mcp_tool_status()
    print(f"• Registered tools: {status['registered_tools']}")
    print(f"• Native fallbacks: {status['native_fallbacks']}")
    print(f"• Circuit breakers: {status['circuit_breakers']}")
    
    print("\n✅ Demo complete! Files created using native Python fallbacks.")

if __name__ == "__main__":
    asyncio.run(demo_native_fallbacks())
