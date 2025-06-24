#!/usr/bin/env python3
"""
Basic test to check what's working
"""

import os
import sys
from pathlib import Path

# Set environment variables
os.environ["DB_HOST"] = "localhost"
os.environ["DB_PORT"] = "54321"
os.environ["DB_NAME"] = "pygent_factory"
os.environ["DB_USER"] = "postgres"
os.environ["DB_PASSWORD"] = "postgres"

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_basic_imports():
    """Test basic imports"""
    print("🧪 Testing basic imports...")
    
    try:
        print("1. Testing database import...")
        from database.production_manager import db_manager
        print(f"   ✅ Database manager imported: {type(db_manager)}")
        
        print("2. Testing Redis import...")
        from cache.redis_manager import redis_manager
        print(f"   ✅ Redis manager imported: {type(redis_manager)}")
        
        print("3. Testing agent import...")
        from agents.base_agent import BaseAgent, AgentType
        print(f"   ✅ Agent classes imported: {BaseAgent}, {AgentType}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_direct_connections():
    """Test direct service connections"""
    print("\n🔌 Testing direct connections...")
    
    try:
        print("1. Testing PostgreSQL connection...")
        import asyncpg
        import asyncio
        
        async def test_pg():
            conn = await asyncpg.connect(
                host="localhost",
                port=54321,
                database="pygent_factory", 
                user="postgres",
                password="postgres"
            )
            version = await conn.fetchval("SELECT version()")
            await conn.close()
            return version
        
        version = asyncio.run(test_pg())
        print(f"   ✅ PostgreSQL connected: {version[:50]}...")
        
        print("2. Testing Redis connection...")
        import redis
        
        r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        pong = r.ping()
        print(f"   ✅ Redis connected: ping = {pong}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 BASIC CONNECTIVITY TEST")
    print("=" * 40)
    
    import_success = test_basic_imports()
    connection_success = test_direct_connections()
    
    print("\n" + "=" * 40)
    print("📊 BASIC TEST RESULTS")
    print("=" * 40)
    print(f"Imports: {'✅ PASSED' if import_success else '❌ FAILED'}")
    print(f"Connections: {'✅ PASSED' if connection_success else '❌ FAILED'}")
    
    if import_success and connection_success:
        print("\n🎉 BASIC INFRASTRUCTURE WORKING!")
        print("Ready for agent communication test!")
    else:
        print("\n❌ Basic infrastructure issues detected")
