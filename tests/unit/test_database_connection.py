#!/usr/bin/env python3
"""
Test database connection and MCP server persistence
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.database.connection import initialize_database
from src.config.settings import get_settings


async def test_database_connection():
    """Test database connection and MCP server models"""
    try:
        # Initialize database
        settings = get_settings()
        db_manager = await initialize_database(settings)
        
        print("✅ Database connection successful")
        
        # Test health check
        health = await db_manager.health_check()
        print(f"✅ Database health check: {health}")
        
        # Test session creation
        async with db_manager.get_session() as session:
            print("✅ Database session created successfully")
              # Check if MCP server table exists
            from sqlalchemy import text
            result = await session.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='servers'"))
            server_table = result.fetchone()
            
            if server_table:
                print("✅ MCP servers table exists")
                
                # Count existing servers
                result = await session.execute(text("SELECT COUNT(*) FROM servers"))
                count = result.scalar()
                print(f"📊 Current MCP servers in database: {count}")
                
                # List existing servers
                if count > 0:
                    result = await session.execute(text("SELECT id, name, status FROM servers"))
                    servers = result.fetchall()
                    print("📋 Existing servers:")
                    for server in servers:
                        print(f"  - {server[1]} ({server[0]}): {server[2]}")
                else:
                    print("📋 No servers currently in database")
                    
            else:
                print("⚠️  MCP servers table does not exist - need to run migrations")
        
        await db_manager.close()
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_database_connection())
    sys.exit(0 if success else 1)
