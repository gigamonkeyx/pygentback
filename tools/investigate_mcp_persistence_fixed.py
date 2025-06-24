#!/usr/bin/env python3
"""
Investigate MCP Server Persistence Issue

This script investigates why MCP server registrations are not being persisted
to the database and shows the current state of both in-memory and database storage.
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

async def investigate_persistence():
    """Investigate MCP server persistence issues"""
    
    print("=" * 80)
    print("INVESTIGATING MCP SERVER PERSISTENCE")
    print("=" * 80)
    
    try:
        # 1. Check database connection and models
        print("\n1. CHECKING DATABASE CONNECTION:")
        print("-" * 50)
        
        from src.database.connection import initialize_database
        from src.database.models import MCPServer, MCPTool
        from src.config.settings import Settings
        
        settings = Settings()
        db_manager = await initialize_database(settings)
        print(f"✓ Database connected: {db_manager is not None}")
        
        # 2. Check what's in the database
        print("\n2. CHECKING DATABASE CONTENT:")
        print("-" * 50)
        
        async with db_manager.get_session() as session:
            from sqlalchemy import select
            
            # Count MCP servers in database
            result = await session.execute(select(MCPServer))
            db_servers = result.scalars().all()
            print(f"MCP servers in database: {len(db_servers)}")
            
            for server in db_servers:
                print(f"  - {server.name} (status: {server.status})")
            
            # Count MCP tools in database
            result = await session.execute(select(MCPTool))
            db_tools = result.scalars().all()
            print(f"MCP tools in database: {len(db_tools)}")
        
        # 3. Check in-memory registry
        print("\n3. CHECKING IN-MEMORY REGISTRY:")
        print("-" * 50)
        
        from src.mcp.server.registry import MCPServerRegistry
        
        registry = MCPServerRegistry()
        await registry.start()
          registered_servers = await registry.list_servers()
        print(f"Servers in memory registry: {len(registered_servers)}")
        
        for registration in registered_servers:
            config = registration.config
            print(f"  - {config.name} (status: {registration.status.value})")
        
        # 4. Check MCP manager setup
        print("\n4. CHECKING MCP MANAGER SETUP:")
        print("-" * 50)
        
        from src.mcp.server.manager import MCPServerManager
        
        manager = MCPServerManager()
        await manager.initialize()
        
        print(f"Manager registry type: {type(manager.registry).__name__}")
        print(f"Manager initialized: {manager._initialized}")
        
        # Check if manager uses database registry
        uses_db = hasattr(manager.registry, 'session_factory')
        print(f"Manager uses database registry: {uses_db}")
        
        # 5. Check cached discovered servers
        print("\n5. CHECKING CACHED DISCOVERED SERVERS:")
        print("-" * 50)
        
        cache_file = Path("data/mcp_cache/discovered_servers.json")
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            print(f"Cached servers found: {len(cached_data)}")
            for server_name in cached_data.keys():
                print(f"  - {server_name}")
        else:
            print("No cached servers file found")
        
        # 6. Summary and diagnosis
        print("\n6. DIAGNOSIS:")
        print("-" * 50)
        
        if len(db_servers) == 0 and len(registered_servers) > 0:
            print("❌ PROBLEM IDENTIFIED:")
            print("   - MCP servers are registered in memory but NOT in database")
            print("   - In-memory registry is not connected to database")
            print("   - All registrations will be lost on restart")
        elif len(db_servers) > 0 and len(registered_servers) == 0:
            print("❌ PROBLEM IDENTIFIED:")
            print("   - MCP servers exist in database but NOT loaded to memory")
            print("   - Registry is not loading from database on startup")
        elif len(db_servers) == 0 and len(registered_servers) == 0:
            print("⚠️  NO MCP SERVERS FOUND:")
            print("   - Neither in database nor in memory")
            print("   - May need to run registration scripts")
        else:
            print("✓ Both database and memory have MCP servers")
            if len(db_servers) != len(registered_servers):
                print("⚠️  WARNING: Different number of servers in DB vs memory")
        
        print("\n7. RECOMMENDED ACTIONS:")
        print("-" * 50)
        
        if not uses_db:
            print("1. Replace MCPServerRegistry with DatabaseMCPServerRegistry")
            print("2. Update MCPServerManager to use database-backed registry")
            print("3. Migrate existing in-memory registrations to database")
            print("4. Update all scripts to work with persistent registry")
        
        await registry.stop()
        
    except Exception as e:
        logger.error(f"Investigation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(investigate_persistence())
