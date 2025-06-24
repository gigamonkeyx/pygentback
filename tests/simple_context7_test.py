#!/usr/bin/env python3
"""
Simple Context7 test using direct HTTP calls
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("🐍 Testing Context7 for Python SDK information...")

# Try to import and test
try:
    from src.mcp.server_registry import MCPServerManager
    from src.config.settings import get_settings
    print("✅ Imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

async def simple_context7_test():
    """Simple test of Context7"""
    try:
        print("📡 Initializing MCP manager...")
        settings = get_settings()
        manager = MCPServerManager(settings)
        await manager.start()
        print("✅ Manager started")

        # List servers
        servers = await manager.list_servers()
        print(f"📊 Found {len(servers)} servers")
        
        for server in servers:
            print(f"   - {server.name}: {server.id}")
            if "context7" in server.name.lower():
                print(f"🎯 Found Context7: {server.name}")
                
                # Check status
                try:
                    status = await manager.get_server_status(server.id)
                    print(f"   Status: {status.get('status', 'unknown')}")
                    
                    if status.get('status') == 'active':
                        print("🚀 Context7 is active! Testing tool call...")
                        
                        # Try a simple tool call
                        result = await manager.call_server_tool(
                            server.id,
                            "resolve-library-id",
                            {"libraryName": "python"}
                        )
                        print(f"📚 Result: {result}")
                        
                except Exception as tool_error:
                    print(f"❌ Tool call error: {tool_error}")
                
                break
        else:
            print("❌ Context7 not found")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(simple_context7_test())
