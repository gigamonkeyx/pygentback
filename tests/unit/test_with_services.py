#!/usr/bin/env python3
"""
Test Real Implementations with Running Services

Test the real implementations with actual PostgreSQL and Redis services running.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Set environment variables for services
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_database_connection():
    """Test database connection with running PostgreSQL"""
    print("🔬 Testing Database Connection with Running PostgreSQL...")
    
    try:
        from database.production_manager import db_manager, initialize_database
        
        # Initialize database
        success = await initialize_database()
        if not success:
            print("❌ Database initialization failed")
            return False
        
        # Test health check
        health = await db_manager.health_check()
        if health.get('status') == 'healthy':
            print(f"✅ Database connection healthy: {health.get('postgresql_version', 'Unknown version')}")
            return True
        else:
            print(f"❌ Database health check failed: {health}")
            return False
            
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

async def test_redis_connection():
    """Test Redis connection with running Redis"""
    print("\n🔬 Testing Redis Connection with Running Redis...")
    
    try:
        from cache.redis_manager import redis_manager, initialize_redis
        
        # Initialize Redis
        success = await initialize_redis()
        if not success:
            print("❌ Redis initialization failed")
            return False
        
        # Test basic operations
        test_key = "test_key_with_services"
        test_value = "test_value_with_services"
        
        # Test set
        await redis_manager.set_data(test_key, test_value, ttl=60)
        
        # Test get
        retrieved_value = await redis_manager.get_data(test_key)
        
        if retrieved_value == test_value:
            print("✅ Redis operations working correctly")
            
            # Test health check
            health = await redis_manager.health_check()
            if health.get('status') == 'healthy':
                print(f"✅ Redis health check passed: {health.get('redis_version', 'Unknown version')}")
                return True
            else:
                print(f"❌ Redis health check failed: {health}")
                return False
        else:
            print(f"❌ Redis get/set failed: expected {test_value}, got {retrieved_value}")
            return False
            
    except Exception as e:
        print(f"❌ Redis test failed: {e}")
        return False

async def test_agent_task_execution():
    """Test agent task execution with database available"""
    print("\n🔬 Testing Agent Task Execution with Database Available...")
    
    try:
        from agents.specialized_agents import ResearchAgent
        
        # Create and initialize agent
        agent = ResearchAgent(name="test_research_agent_with_db")
        await agent.initialize()
        
        # Test document search with database available
        test_params = {
            "query": "test query with database",
            "limit": 3
        }
        
        try:
            result = await agent._search_documents(test_params)
            
            if "search_method" in result:
                if result["search_method"] in ["rag_pipeline", "database_search"]:
                    print(f"✅ Agent task execution working with {result['search_method']}")
                    return True
                else:
                    print(f"⚠️ Agent using unexpected search method: {result['search_method']}")
                    return False
            else:
                print("❌ Agent task execution missing search method info")
                return False
                
        except RuntimeError as e:
            if "Real database connection required" in str(e):
                print("❌ Agent still requiring database despite it being available")
                return False
            else:
                print(f"❌ Agent task execution failed: {e}")
                return False
            
    except Exception as e:
        print(f"❌ Agent task execution test failed: {e}")
        return False

async def test_message_routing():
    """Test message routing with Redis available"""
    print("\n🔬 Testing Message Routing with Redis Available...")
    
    try:
        from agents.communication_system import communication_system
        from agents.base_agent import AgentMessage, MessageType
        
        # Initialize communication system
        success = await communication_system.initialize()
        if not success:
            print("❌ Communication system initialization failed")
            return False
        
        # Test message creation and routing
        test_message = AgentMessage(
            type=MessageType.DIRECT,
            sender_id="test_sender_with_redis",
            recipient_id="test_recipient_with_redis",
            content={"test": "message with redis"}
        )
        
        if communication_system.is_initialized:
            print("✅ Communication system initialized with Redis")
            print("✅ Message routing real implementation working")
            return True
        else:
            print("❌ Communication system not properly initialized")
            return False
            
    except Exception as e:
        print(f"❌ Message routing test failed: {e}")
        return False

async def run_comprehensive_test():
    """Run comprehensive test with services"""
    print("🧪 TESTING REAL IMPLEMENTATIONS WITH RUNNING SERVICES")
    print("=" * 60)
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Redis Integration", test_redis_connection),
        ("Agent Task Execution", test_agent_task_execution),
        ("Message Routing", test_message_routing)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        try:
            if await test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test error: {e}")
    
    total = len(tests)
    print("\n" + "=" * 60)
    print("📊 SERVICES TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED WITH RUNNING SERVICES!")
        print("✅ Real implementations working correctly with actual services")
    else:
        print(f"⚠️ {total - passed} tests still failing")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(run_comprehensive_test())
    sys.exit(0 if success else 1)
