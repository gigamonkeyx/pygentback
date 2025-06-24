#!/usr/bin/env python3
"""
Investigate MCP Tool Capabilities - Simple Version

This script investigates how MCP server capabilities are captured and stored
by examining already running servers and their tool advertisements.
"""

import asyncio
import sys
import json
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def investigate_simple():
    """Simple investigation of MCP capabilities"""
    
    print("=" * 80)
    print("INVESTIGATING MCP CAPABILITIES - SIMPLE VERSION")
    print("=" * 80)
    
    try:
        # 1. Initialize manager and start a single server for testing
        print("\n1. INITIALIZING MCP MANAGER:")
        print("-" * 50)
        
        from src.mcp.server.manager import MCPServerManager
        from src.mcp.server.config import MCPServerConfig, MCPServerType
        from src.config.settings import Settings
        
        settings = Settings()
        manager = MCPServerManager(settings)
        await manager.initialize()
        print("✓ MCP Manager initialized")
          # 2. Check if any servers are already registered
        print("\n2. CHECKING EXISTING SERVERS:")
        print("-" * 50)
        
        existing_servers = await manager.list_servers()
        print(f"Registered servers: {len(existing_servers)}")
        
        if not existing_servers:
            # Register a simple test server (Context7 - it's reliable)
            print("\nRegistering Context7 MCP server for testing...")
            
            config = MCPServerConfig(
                id="context7-test",
                name="Context7 Documentation Test",
                command=[
                    "C:/Users/admin/AppData/Roaming/npm/npx.cmd",
                    "@upstash/context7-mcp"
                ],
                server_type=MCPServerType.CUSTOM,
                auto_start=True
            )
            
            success = await manager.register_server(config)
            if success:
                await manager.start_server("context7-test")
                print("✓ Context7 server registered and started")
                existing_servers = await manager.list_servers()
            else:
                print("❌ Failed to register Context7 server")
        
        # 3. Examine server capabilities in detail
        print("\n3. EXAMINING SERVER CAPABILITIES:")
        print("-" * 50)
        
        for server in existing_servers:
            print(f"\n--- Server: {server.config.name} ---")
            print(f"ID: {server.config.id}")
            print(f"Type: {server.config.server_type}")
            print(f"Status: {server.status}")
            
            # Examine the full server registration
            registration = await manager.registry.get_server(server.config.id)
            if registration:
                print(f"Start count: {registration.start_count}")
                print(f"Last heartbeat: {registration.last_heartbeat}")
                print(f"Metadata: {registration.metadata}")
                
                # Check configuration details
                config = registration.config
                print(f"Command: {config.command}")
                print(f"Transport: {config.transport}")
                print(f"Environment: {getattr(config, 'environment', 'None')}")
                
                # Check if tools are captured
                if hasattr(config, 'tools') and config.tools:
                    print(f"Configured tools: {config.tools}")
                else:
                    print("❌ No tools in configuration")
        
        # 4. Test direct tool discovery
        print("\n4. TESTING TOOL DISCOVERY:")
        print("-" * 50)
        
        for server in existing_servers:
            server_id = server.config.id
            print(f"\n--- Tool Discovery: {server.config.name} ---")
            
            try:
                # Check if manager has method to get tools
                if hasattr(manager, 'get_server_tools'):
                    tools = await manager.get_server_tools(server_id)
                    if tools:
                        print(f"Discovered {len(tools)} tools:")
                        for tool in tools[:3]:  # Show first 3
                            print(f"  • {tool.name}")
                            if hasattr(tool, 'description'):
                                print(f"    Description: {tool.description}")
                            if hasattr(tool, 'inputSchema'):
                                schema = tool.inputSchema
                                if schema and isinstance(schema, dict):
                                    props = schema.get('properties', {})
                                    print(f"    Parameters: {list(props.keys())}")
                    else:
                        print("  No tools discovered")
                else:
                    print("  ❌ Manager doesn't have get_server_tools method")
                    
                # Try alternative tool discovery methods
                if hasattr(manager, 'lifecycle'):
                    lifecycle = manager.lifecycle
                    if hasattr(lifecycle, 'get_server_capabilities'):
                        caps = await lifecycle.get_server_capabilities(server_id)
                        print(f"  Server capabilities: {caps}")
                
            except Exception as e:
                print(f"  ❌ Tool discovery failed: {e}")
        
        # 5. Check tool executor capabilities
        print("\n5. CHECKING TOOL EXECUTOR:")
        print("-" * 50)
        
        try:
            from src.mcp.tools.executor import MCPToolExecutor
            
            # Check if we can create a tool executor
            executor = MCPToolExecutor()
            print("✓ Tool executor created")
            
            # Check executor methods
            methods = [m for m in dir(executor) if not m.startswith('_')]
            print(f"Available methods: {methods}")
            
            # Try to get all available tools
            if hasattr(executor, 'get_available_tools'):
                all_tools = await executor.get_available_tools()
                print(f"Total available tools: {len(all_tools) if all_tools else 0}")
            
        except Exception as e:
            print(f"❌ Tool executor failed: {e}")
        
        # 6. Check database tool storage capability
        print("\n6. CHECKING DATABASE TOOL MODELS:")
        print("-" * 50)
        
        from src.database.models import MCPServer, MCPTool
        
        # Examine the tool model structure
        print("MCPTool model fields:")
        for column in MCPTool.__table__.columns:
            print(f"  • {column.name}: {column.type}")
        
        print("\nMCPServer model fields:")
        for column in MCPServer.__table__.columns:
            print(f"  • {column.name}: {column.type}")
        
        # 7. Analysis summary
        print("\n7. CAPABILITY CAPTURE ANALYSIS:")
        print("-" * 50)
        
        print("FINDINGS:")
        print("1. Server registration captures basic metadata")
        print("2. Tool discovery depends on runtime introspection")
        print("3. Database models support rich tool metadata")
        print("4. Need to verify tool schema capture completeness")
        print("5. Agent advertisement mechanism needs investigation")
        
        await manager.shutdown()
        
    except Exception as e:
        logger.error(f"Investigation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(investigate_simple())
