#!/usr/bin/env python3
"""
Test Production Database Setup

Test the production PostgreSQL database implementation.
"""

import asyncio
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from database.production_manager import db_manager
from database.migrations import migration_manager, initialize_production_database
from database.models import User, Agent, Task, Document, AgentType, TaskState

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_database_connection():
    """Test basic database connection"""
    print("🔌 Testing Database Connection...")
    
    try:
        # Set test database URL (use SQLite for testing if PostgreSQL not available)
        test_db_url = "sqlite+aiosqlite:///test_production.db"
        db_manager.database_url = test_db_url
        
        # Initialize database
        success = await db_manager.initialize()
        if not success:
            print("❌ Database initialization failed")
            return False
        
        # Test health check
        health = await db_manager.health_check()
        print(f"✅ Database connection successful")
        print(f"   Status: {health.get('status')}")
        print(f"   Response time: {health.get('response_time_ms', 0):.2f}ms")
        
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


async def test_model_creation():
    """Test creating model instances"""
    print("\n📝 Testing Model Creation...")
    
    try:
        async with db_manager.get_session() as session:
            # Create test user
            user = User(
                username="test_user",
                email="test@example.com",
                role="admin"
            )
            session.add(user)
            await session.flush()  # Get the ID
            
            # Create test agent
            agent = Agent(
                user_id=user.id,
                name="Test Agent",
                agent_type=AgentType.CODE_ANALYZER,
                description="Test agent for production database",
                capabilities=["code_analysis", "documentation"],
                configuration={"model": "gpt-4", "temperature": 0.7}
            )
            session.add(agent)
            await session.flush()
            
            # Create test task
            task = Task(
                task_type="code_analysis",
                description="Analyze Python code for quality metrics",
                input_data={"code": "def hello(): return 'world'"},
                state=TaskState.PENDING,
                agent_id=agent.id
            )
            session.add(task)
            
            # Create test document
            document = Document(
                user_id=user.id,
                title="Test Document",
                content="This is a test document for the production database.",
                content_type="text/plain",
                document_metadata={"source": "test", "category": "documentation"}
            )
            session.add(document)
            
            await session.commit()
            
            print(f"✅ Models created successfully")
            print(f"   User ID: {user.id}")
            print(f"   Agent ID: {agent.id}")
            print(f"   Task ID: {task.id}")
            print(f"   Document ID: {document.id}")
            
            return True
            
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False


async def test_queries():
    """Test database queries"""
    print("\n🔍 Testing Database Queries...")
    
    try:
        async with db_manager.get_session() as session:
            # Query users
            from sqlalchemy import select
            
            result = await session.execute(select(User))
            users = result.scalars().all()
            print(f"✅ Found {len(users)} users")
            
            # Query agents with relationships
            result = await session.execute(
                select(Agent).where(Agent.agent_type == AgentType.CODE_ANALYZER)
            )
            agents = result.scalars().all()
            print(f"✅ Found {len(agents)} code analyzer agents")
            
            # Query tasks with filters
            result = await session.execute(
                select(Task).where(Task.state == TaskState.PENDING)
            )
            tasks = result.scalars().all()
            print(f"✅ Found {len(tasks)} pending tasks")
            
            return True
            
    except Exception as e:
        print(f"❌ Database queries failed: {e}")
        return False


async def test_performance():
    """Test database performance"""
    print("\n⚡ Testing Database Performance...")
    
    try:
        start_time = datetime.utcnow()
        
        # Bulk insert test
        async with db_manager.get_session() as session:
            # Create multiple users
            users = []
            for i in range(100):
                user = User(
                    username=f"perf_user_{i}",
                    email=f"perf_user_{i}@example.com",
                    role="user"
                )
                users.append(user)
            
            session.add_all(users)
            await session.commit()
        
        bulk_time = (datetime.utcnow() - start_time).total_seconds()
        print(f"✅ Bulk insert (100 users): {bulk_time:.3f}s")
        
        # Query performance test
        start_time = datetime.utcnow()
        
        async with db_manager.get_session() as session:
            from sqlalchemy import select, func
            
            # Count query
            result = await session.execute(select(func.count(User.id)))
            user_count = result.scalar()
            
            # Complex query with joins
            result = await session.execute(
                select(User, Agent)
                .join(Agent, User.id == Agent.user_id, isouter=True)
                .limit(50)
            )
            user_agent_pairs = result.all()
        
        query_time = (datetime.utcnow() - start_time).total_seconds()
        print(f"✅ Complex queries: {query_time:.3f}s")
        print(f"   Total users: {user_count}")
        print(f"   User-agent pairs: {len(user_agent_pairs)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


async def test_migrations():
    """Test migration system"""
    print("\n🔄 Testing Migration System...")
    
    try:
        # Initialize migration manager
        await migration_manager.initialize()
        
        # Get migration status
        status = await migration_manager.get_migration_status()
        print(f"✅ Migration system initialized")
        print(f"   Total migrations: {status.get('total_migrations', 0)}")
        print(f"   Applied migrations: {status.get('applied_migrations', 0)}")
        print(f"   Pending migrations: {status.get('pending_migrations', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration test failed: {e}")
        return False


async def test_cleanup():
    """Cleanup test data"""
    print("\n🧹 Cleaning up test data...")
    
    try:
        # Remove test database file if using SQLite
        if "sqlite" in db_manager.database_url:
            db_file = Path("test_production.db")
            if db_file.exists():
                db_file.unlink()
                print("✅ Test database file removed")
        
        # Cleanup database manager
        await db_manager.cleanup()
        print("✅ Database manager cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        return False


async def run_all_tests():
    """Run all production database tests"""
    print("🚀 Production Database Test Suite")
    print("=" * 50)
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Model Creation", test_model_creation),
        ("Database Queries", test_queries),
        ("Performance Testing", test_performance),
        ("Migration System", test_migrations),
        ("Cleanup", test_cleanup)
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
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("   Production database implementation is working correctly.")
    else:
        print("⚠️ SOME TESTS FAILED")
        print("   Check the errors above and fix issues before production deployment.")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
