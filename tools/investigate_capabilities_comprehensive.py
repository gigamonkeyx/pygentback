#!/usr/bin/env python3
"""
Comprehensive MCP Server Capabilities Investigation
Checks registration capture, storage, and tool advertisement
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def investigate_comprehensive():
    """Comprehensive investigation of MCP server capabilities"""
    
    print("=" * 80)
    print("COMPREHENSIVE MCP CAPABILITIES INVESTIGATION")
    print("=" * 80)
    
    try:
        # 1. Initialize MCP Manager
        print("\n1. INITIALIZING MCP COMPONENTS:")
        print("-" * 50)
        
        from src.mcp.server.manager import MCPServerManager
        from src.mcp.server.config import MCPServerConfig, MCPServerType
        from src.config.settings import Settings
        from src.mcp.server.database_registry import DatabaseMCPServerRegistry
        
        settings = Settings()
        manager = MCPServerManager(settings)
        await manager.initialize()
        print("✓ MCP Manager initialized")
        
        # 2. Check existing registrations in memory
        print("\n2. CHECKING IN-MEMORY REGISTRATIONS:")
        print("-" * 50)
        
        existing_servers = await manager.list_servers()
        print(f"In-memory servers: {len(existing_servers)}")
        
        for server in existing_servers:
            print(f"  - {server.get('name', 'unknown')}: {server.get('status', 'unknown')}")
            if 'tools' in server:
                print(f"    Tools: {len(server.get('tools', []))}")
                for tool in server.get('tools', [])[:3]:  # Show first 3 tools
                    print(f"      * {tool.get('name', 'unknown')}: {tool.get('description', 'no description')[:50]}...")
        
        # 3. Check database persistence
        print("\n3. CHECKING DATABASE PERSISTENCE:")
        print("-" * 50)
        
        try:
            db_registry = DatabaseMCPServerRegistry()
            await db_registry.initialize()
            
            # Check if servers are in database
            db_servers = await db_registry.list_servers()
            print(f"Database servers: {len(db_servers)}")
            
            for server in db_servers:
                print(f"  - {server.name}: {server.status}")
                if hasattr(server, 'config') and server.config:
                    print(f"    Config: {server.config.get('command', 'no command')}")
                
        except Exception as e:
            print(f"❌ Database check failed: {e}")
        
        # 4. Check tool discovery capabilities
        print("\n4. CHECKING TOOL DISCOVERY:")
        print("-" * 50)
        
        try:
            # Get servers with tools
            tool_servers = await manager.get_servers_by_capability("tools")
            print(f"Servers with tools capability: {len(tool_servers)}")
            
            # Test tool discovery for each server
            for server in tool_servers:
                server_name = server.get('name', 'unknown')
                print(f"\n  Testing tool discovery for: {server_name}")
                
                try:
                    # Try to get available tools
                    tools = await manager.get_tool("*")  # Wildcard to see if it works
                    print(f"    Direct tool query result: {tools}")
                except Exception as e:
                    print(f"    Tool query failed: {e}")
                    
        except Exception as e:
            print(f"❌ Tool discovery check failed: {e}")
        
        # 5. Check capability advertising
        print("\n5. CHECKING CAPABILITY ADVERTISING:")
        print("-" * 50)
        
        try:
            # Check if capabilities are properly stored
            for server in existing_servers:
                server_name = server.get('name', 'unknown')
                print(f"\n  Analyzing capabilities for: {server_name}")
                
                # Check what capabilities are advertised
                capabilities = server.get('capabilities', {})
                print(f"    Advertised capabilities: {list(capabilities.keys())}")
                
                # Check tools specifically
                if 'tools' in capabilities:
                    tools_cap = capabilities['tools']
                    print(f"    Tools capability: {tools_cap}")
                
                # Check if tools are actually available
                tools = server.get('tools', [])
                print(f"    Available tools: {len(tools)}")
                
                for tool in tools[:2]:  # Show first 2 tools in detail
                    print(f"      Tool: {tool.get('name', 'unknown')}")
                    print(f"        Description: {tool.get('description', 'none')}")
                    print(f"        Input Schema: {tool.get('inputSchema', {}).get('type', 'unknown')}")
                    
                    # Check annotations
                    annotations = tool.get('annotations', {})
                    if annotations:
                        print(f"        Annotations: {annotations}")
                
        except Exception as e:
            print(f"❌ Capability advertising check failed: {e}")
        
        # 6. Test real Context7 if available
        print("\n6. TESTING REAL CONTEXT7 CONNECTION:")
        print("-" * 50)
        
        try:
            # Try to use our SDK test to verify Context7 is working
            import subprocess
            result = subprocess.run([
                sys.executable, "context7_sdk_test.py"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print("✓ Context7 SDK test passed")
                print("    Context7 is properly connected and working")
            else:
                print(f"❌ Context7 SDK test failed: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Context7 test failed: {e}")
        
        # 7. Summary and recommendations
        print("\n7. SUMMARY AND RECOMMENDATIONS:")
        print("-" * 50)
        
        print("\nFINDINGS:")
        print(f"  - In-memory servers: {len(existing_servers)}")
        print(f"  - Database servers: {len(db_servers) if 'db_servers' in locals() else 'unknown'}")
        
        # Check for capability gaps
        capability_gaps = []
        
        if len(existing_servers) == 0:
            capability_gaps.append("No MCP servers registered")
        
        for server in existing_servers:
            if not server.get('tools'):
                capability_gaps.append(f"Server {server.get('name')} has no tools")
            
            if not server.get('capabilities'):
                capability_gaps.append(f"Server {server.get('name')} has no capabilities metadata")
        
        if capability_gaps:
            print("\nCAPABILITY GAPS FOUND:")
            for gap in capability_gaps:
                print(f"  ❌ {gap}")
        else:
            print("\n✓ No major capability gaps detected")
        
        print("\nRECOMMENDATIONS:")
        print("  1. Migrate to database-backed registry for persistence")
        print("  2. Ensure all servers properly advertise tools via tools/list")
        print("  3. Store complete capability metadata including annotations")
        print("  4. Implement dynamic tool discovery notifications")
        print("  5. Add capability validation during registration")
        
    except Exception as e:
        print(f"ERROR: Investigation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(investigate_comprehensive())
