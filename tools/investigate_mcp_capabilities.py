#!/usr/bin/env python3
"""
Investigate MCP Tool Capabilities and Advertisement

This script investigates how MCP server capabilities are captured, stored,
and advertised to agents for maximum usability in the DGM-inspired system.
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

async def investigate_capabilities():
    """Investigate MCP capabilities capture and advertisement"""
    
    print("=" * 80)
    print("INVESTIGATING MCP CAPABILITIES CAPTURE & ADVERTISEMENT")
    print("=" * 80)
    
    try:
        # 1. Start MCP servers to capture their capabilities
        print("\n1. STARTING MCP SERVERS TO CAPTURE CAPABILITIES:")
        print("-" * 60)
        
        from src.mcp.server.manager import MCPServerManager
        from src.config.settings import Settings
        
        settings = Settings()
        manager = MCPServerManager(settings)
        await manager.initialize()
        
        # Register real servers
        from update_mcp_servers import load_mcp_server_configs, register_and_start_server
        
        configs = load_mcp_server_configs()
        print(f"Loading {len(configs)} MCP server configurations...")
        
        for config in configs[:2]:  # Test with first 2 servers
            print(f"\nRegistering server: {config['name']}")
            result = await register_and_start_server(manager, config)
            if result:
                print(f"✓ Server '{config['name']}' registered successfully")
            else:
                print(f"❌ Server '{config['name']}' registration failed")
        
        # 2. Examine registered server capabilities
        print("\n2. EXAMINING REGISTERED SERVER CAPABILITIES:")
        print("-" * 60)
        
        servers = await manager.list_active_servers()
        print(f"Active servers: {len(servers)}")
        
        for server in servers:
            print(f"\n--- Server: {server.config.name} ---")
            print(f"ID: {server.config.id}")
            print(f"Type: {server.config.server_type}")
            print(f"Transport: {server.config.transport}")
            print(f"Status: {server.status}")
            
            # Check if capabilities are captured
            if hasattr(server.config, 'capabilities'):
                capabilities = server.config.capabilities
                print(f"Capabilities: {capabilities}")
            else:
                print("❌ No capabilities field found")
            
            # Check metadata
            if hasattr(server, 'metadata'):
                metadata = server.metadata
                print(f"Metadata: {json.dumps(metadata, indent=2)}")
            else:
                print("❌ No metadata found")
        
        # 3. Test MCP tool discovery
        print("\n3. TESTING MCP TOOL DISCOVERY:")
        print("-" * 60)
        
        from src.mcp.tools.executor import MCPToolExecutor
        
        executor = MCPToolExecutor()
        
        for server in servers:
            server_id = server.config.id
            print(f"\n--- Tools for {server.config.name} ---")
            
            try:
                # Get tools from server
                tools = await manager.get_server_tools(server_id)
                print(f"Available tools: {len(tools) if tools else 0}")
                
                if tools:
                    for tool in tools[:3]:  # Show first 3 tools
                        print(f"  • {tool.name}: {tool.description}")
                        if hasattr(tool, 'inputSchema'):
                            schema = tool.inputSchema
                            if schema and 'properties' in schema:
                                params = list(schema['properties'].keys())
                                print(f"    Parameters: {params}")
                else:
                    print("  No tools discovered")
                    
            except Exception as e:
                print(f"  ❌ Tool discovery failed: {e}")
        
        # 4. Check tool advertisement to agents
        print("\n4. CHECKING TOOL ADVERTISEMENT TO AGENTS:")
        print("-" * 60)
        
        # Check how tools are made available to agents
        from src.core.agent_factory import AgentFactory
        from src.ai.reasoning.unified_pipeline import UnifiedReasoningPipeline
        
        try:
            agent_factory = AgentFactory()
            print("✓ Agent factory initialized")
            
            # Check if agent factory has access to MCP tools
            if hasattr(agent_factory, 'mcp_manager'):
                print("✓ Agent factory has MCP manager access")
            else:
                print("❌ Agent factory missing MCP manager access")
            
            # Test reasoning pipeline tool access
            pipeline = UnifiedReasoningPipeline()
            print("✓ Unified reasoning pipeline initialized")
            
            if hasattr(pipeline, 'tool_executor'):
                print("✓ Pipeline has tool executor access")
            else:
                print("❌ Pipeline missing tool executor access")
                
        except Exception as e:
            print(f"❌ Agent/pipeline initialization failed: {e}")
        
        # 5. Analyze tool capability metadata structure
        print("\n5. ANALYZING TOOL CAPABILITY METADATA:")
        print("-" * 60)
        
        for server in servers:
            server_id = server.config.id
            print(f"\n--- Capability Analysis: {server.config.name} ---")
            
            try:
                # Get server registration details
                registration = await manager.registry.get_server(server_id)
                if registration:
                    config = registration.config
                    
                    # Check what capability info is stored
                    print("Stored Configuration:")
                    print(f"  Command: {config.command}")
                    print(f"  Environment: {getattr(config, 'environment', 'None')}")
                    print(f"  Args: {getattr(config, 'args', 'None')}")
                    
                    # Check if server exposes its own capabilities
                    try:
                        server_info = await manager.get_server_info(server_id)
                        if server_info:
                            print("Server Info:")
                            print(f"  {json.dumps(server_info, indent=4)}")
                        else:
                            print("  No server info available")
                    except Exception as e:
                        print(f"  ❌ Could not get server info: {e}")
                
            except Exception as e:
                print(f"❌ Capability analysis failed: {e}")
        
        # 6. Check database tool storage
        print("\n6. CHECKING DATABASE TOOL STORAGE:")
        print("-" * 60)
        
        from src.database.connection import initialize_database
        from src.database.models import MCPServer, MCPTool
        
        db_manager = await initialize_database(settings)
        
        async with db_manager.get_session() as session:
            from sqlalchemy import select
            
            # Check if any tools are stored in database
            result = await session.execute(select(MCPTool))
            db_tools = result.scalars().all()
            print(f"Tools in database: {len(db_tools)}")
            
            if db_tools:
                for tool in db_tools[:5]:  # Show first 5
                    print(f"  • {tool.name}: {tool.description}")
                    print(f"    Server ID: {tool.server_id}")
                    print(f"    Parameters: {tool.parameters}")
            else:
                print("❌ No tools stored in database (expected due to persistence issue)")
        
        # 7. Summary and recommendations
        print("\n7. CAPABILITY CAPTURE ANALYSIS:")
        print("-" * 60)
        
        print("Key Questions:")
        print("1. Are server capabilities fully captured during registration?")
        print("2. Are tool schemas and parameters properly stored?")
        print("3. Can agents discover and access all available tools?")
        print("4. Is tool metadata rich enough for intelligent selection?")
        print("5. Are tool combinations and dependencies tracked?")
        
        await manager.shutdown()
        
    except Exception as e:
        logger.error(f"Investigation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(investigate_capabilities())
