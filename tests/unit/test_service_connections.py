#!/usr/bin/env python3
"""
Test Direct Service Connections

Test direct connections to PostgreSQL and Redis services.
"""

import os
import asyncio
import sys
from pathlib import Path

# Set environment variables
os.environ["DB_HOST"] = "localhost"
os.environ["DB_PORT"] = "54321"
os.environ["DB_NAME"] = "pygent_factory"
os.environ["DB_USER"] = "postgres"
os.environ["DB_PASSWORD"] = "postgres"
os.environ["DATABASE_URL"] = "postgresql+asyncpg://postgres:postgres@localhost:54321/pygent_factory"

os.environ["REDIS_HOST"] = "localhost"
os.environ["REDIS_PORT"] = "6379"
os.environ["REDIS_DB"] = "0"
os.environ["REDIS_URL"] = "redis://localhost:6379/0"

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

async def test_direct_postgres():
    """Test direct PostgreSQL connection"""
    print("🔬 Testing Direct PostgreSQL Connection...")
    
    try:
        import asyncpg
        
        # Direct connection test
        conn = await asyncpg.connect(
            host="localhost",
            port=54321,
            database="pygent_factory",
            user="postgres",
            password="postgres"
        )
        
        # Test query
        version = await conn.fetchval("SELECT version()")
        print(f"✅ PostgreSQL connected: {version[:50]}...")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        return False

async def test_direct_redis():
    """Test direct Redis connection"""
    print("\n🔬 Testing Direct Redis Connection...")
    
    try:
        import redis.asyncio as redis
        
        # Direct connection test
        r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        
        # Test ping
        pong = await r.ping()
        print(f"✅ Redis connected: ping returned {pong}")
        
        # Test set/get
        await r.set("test_key", "test_value")
        value = await r.get("test_key")
        print(f"✅ Redis operations working: {value}")
        
        await r.close()
        return True
        
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

async def test_pygent_database_manager():
    """Test PyGent database manager with services"""
    print("\n🔬 Testing PyGent Database Manager...")
    
    try:
        from database.production_manager import ProductionDatabaseManager
        
        # Create and initialize database manager
        db_manager = ProductionDatabaseManager()
        success = await db_manager.initialize()
        
        if success:
            print("✅ PyGent Database Manager initialized")
            
            # Test health check
            health = await db_manager.health_check()
            if health.get('status') == 'healthy':
                print(f"✅ Database health check passed: {health.get('postgresql_version', 'Unknown')}")
                return True
            else:
                print(f"❌ Database health check failed: {health}")
                return False
        else:
            print("❌ PyGent Database Manager initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ PyGent Database Manager test failed: {e}")
        return False

async def test_pygent_redis_manager():
    """Test PyGent Redis manager with services"""
    print("\n🔬 Testing PyGent Redis Manager...")
    
    try:
        from cache.redis_manager import RedisManager
        
        # Create and initialize Redis manager
        redis_manager = RedisManager()
        success = await redis_manager.initialize()
        
        if success:
            print("✅ PyGent Redis Manager initialized")
            
            # Test operations
            await redis_manager.set("test_pygent_key", "test_pygent_value", ttl=60)
            value = await redis_manager.get("test_pygent_key")
            
            if value == "test_pygent_value":
                print("✅ PyGent Redis operations working")
                
                # Test health check
                health = await redis_manager.health_check()
                if health.get('status') == 'healthy':
                    print(f"✅ Redis health check passed: {health.get('redis_version', 'Unknown')}")
                    return True
                else:
                    print(f"❌ Redis health check failed: {health}")
                    return False
            else:
                print(f"❌ PyGent Redis get/set failed: expected test_pygent_value, got {value}")
                return False
        else:
            print("❌ PyGent Redis Manager initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ PyGent Redis Manager test failed: {e}")
        return False

async def main():
    """Run all connection tests"""
    print("🧪 TESTING SERVICE CONNECTIONS")
    print("=" * 50)
    
    tests = [
        ("Direct PostgreSQL", test_direct_postgres),
        ("Direct Redis", test_direct_redis),
        ("PyGent Database Manager", test_pygent_database_manager),
        ("PyGent Redis Manager", test_pygent_redis_manager)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        try:
            if await test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} error: {e}")
    
    total = len(tests)
    print("\n" + "=" * 50)
    print("📊 CONNECTION TEST SUMMARY")
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL SERVICE CONNECTIONS WORKING!")
    else:
        print(f"⚠️ {total - passed} connection tests failed")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
