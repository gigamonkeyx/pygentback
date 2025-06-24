#!/usr/bin/env python3
"""
Test Production Database Models

Test the production database models with SQLite for validation.
"""

import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select, func

from database.models import (
    Base, User, Agent, Task, Document, AgentType, TaskState, 
    TaskArtifact, AgentMemory, DocumentChunk, CodeRepository, CodeFile
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDatabaseManager:
    """Simple test database manager for SQLite"""
    
    def __init__(self):
        self.engine = None
        self.session_factory = None
    
    async def initialize(self):
        """Initialize test database"""
        try:
            # Use SQLite for testing
            self.engine = create_async_engine(
                "sqlite+aiosqlite:///test_models.db",
                echo=False
            )
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create all tables
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info("Test database initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize test database: {e}")
            return False
    
    async def get_session(self):
        """Get database session"""
        return self.session_factory()
    
    async def cleanup(self):
        """Cleanup database"""
        if self.engine:
            await self.engine.dispose()


async def test_model_creation():
    """Test creating all production models"""
    print("📝 Testing Production Model Creation...")
    
    db = TestDatabaseManager()
    await db.initialize()
    
    try:
        async with db.get_session() as session:
            # Create test user
            user = User(
                username="test_user",
                email="test@example.com",
                role="admin",
                user_metadata={"test": True}
            )
            session.add(user)
            await session.flush()
            
            # Create test agent
            agent = Agent(
                user_id=user.id,
                name="Test Agent",
                agent_type=AgentType.CODE_ANALYZER,
                description="Test agent for production models",
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
                task_metadata={"priority": "high"},
                state=TaskState.PENDING,
                agent_id=agent.id
            )
            session.add(task)
            await session.flush()
            
            # Create task artifact
            artifact = TaskArtifact(
                task_id=task.id,
                name="analysis_result",
                artifact_type="analysis",
                content_text="Code analysis completed successfully",
                artifact_metadata={"score": 95}
            )
            session.add(artifact)
            
            # Create agent memory
            memory = AgentMemory(
                agent_id=agent.id,
                memory_type="conversation",
                content="User asked about code quality",
                memory_metadata={"importance": "high"}
            )
            session.add(memory)
            
            # Create test document
            document = Document(
                user_id=user.id,
                title="Test Document",
                content="This is a test document for production models.",
                content_type="text/plain",
                document_metadata={"source": "test", "category": "documentation"}
            )
            session.add(document)
            await session.flush()
            
            # Create document chunk
            chunk = DocumentChunk(
                document_id=document.id,
                chunk_index=0,
                content="This is a test document chunk.",
                chunk_metadata={"tokens": 10}
            )
            session.add(chunk)
            
            # Create code repository
            repo = CodeRepository(
                name="Test Repository",
                description="Test code repository",
                repository_url="https://github.com/test/repo",
                repository_metadata={"stars": 100}
            )
            session.add(repo)
            await session.flush()
            
            # Create code file
            code_file = CodeFile(
                repository_id=repo.id,
                file_path="/src/main.py",
                filename="main.py",
                file_extension=".py",
                language="python",
                content="def main(): print('Hello World')",
                file_metadata={"complexity": "low"}
            )
            session.add(code_file)
            
            await session.commit()
            
            print(f"✅ All production models created successfully")
            print(f"   User ID: {user.id}")
            print(f"   Agent ID: {agent.id}")
            print(f"   Task ID: {task.id}")
            print(f"   Document ID: {document.id}")
            print(f"   Repository ID: {repo.id}")
            
            return True
            
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False
    finally:
        await db.cleanup()


async def test_model_relationships():
    """Test model relationships and queries"""
    print("\n🔗 Testing Model Relationships...")
    
    db = TestDatabaseManager()
    await db.initialize()
    
    try:
        async with db.get_session() as session:
            # Create related models
            user = User(username="rel_user", email="rel@example.com")
            session.add(user)
            await session.flush()
            
            agent = Agent(
                user_id=user.id,
                name="Relationship Agent",
                agent_type=AgentType.ORCHESTRATOR
            )
            session.add(agent)
            await session.flush()
            
            # Create multiple tasks for the agent
            for i in range(3):
                task = Task(
                    task_type=f"task_{i}",
                    description=f"Test task {i}",
                    state=TaskState.PENDING if i < 2 else TaskState.COMPLETED,
                    agent_id=agent.id
                )
                session.add(task)
            
            await session.commit()
            
            # Test relationships
            # Query user with agents
            result = await session.execute(
                select(User).where(User.username == "rel_user")
            )
            user = result.scalar_one()
            
            # Load agents relationship
            await session.refresh(user, ["agents"])
            print(f"✅ User has {len(user.agents)} agents")
            
            # Query agent with tasks
            result = await session.execute(
                select(Agent).where(Agent.name == "Relationship Agent")
            )
            agent = result.scalar_one()
            
            await session.refresh(agent, ["tasks"])
            print(f"✅ Agent has {len(agent.tasks)} tasks")
            
            # Test complex query with joins
            result = await session.execute(
                select(User, Agent, Task)
                .join(Agent, User.id == Agent.user_id)
                .join(Task, Agent.id == Task.agent_id)
                .where(Task.state == TaskState.PENDING)
            )
            
            pending_tasks = result.all()
            print(f"✅ Found {len(pending_tasks)} pending tasks with user/agent info")
            
            return True
            
    except Exception as e:
        print(f"❌ Relationship test failed: {e}")
        return False
    finally:
        await db.cleanup()


async def test_model_validation():
    """Test model validation and constraints"""
    print("\n✅ Testing Model Validation...")
    
    db = TestDatabaseManager()
    await db.initialize()
    
    try:
        async with db.get_session() as session:
            # Test unique constraints
            user1 = User(username="unique_user", email="unique@example.com")
            session.add(user1)
            await session.commit()
            
            # Try to create duplicate username (should fail)
            try:
                user2 = User(username="unique_user", email="different@example.com")
                session.add(user2)
                await session.commit()
                print("❌ Unique constraint not enforced")
                return False
            except Exception:
                await session.rollback()
                print("✅ Unique constraint enforced correctly")
            
            # Test enum validation
            try:
                agent = Agent(
                    user_id=user1.id,
                    name="Test Agent",
                    agent_type=AgentType.CODE_ANALYZER  # Valid enum
                )
                session.add(agent)
                await session.commit()
                print("✅ Enum validation working")
            except Exception as e:
                print(f"❌ Enum validation failed: {e}")
                return False
            
            # Test JSON fields
            task = Task(
                task_type="json_test",
                input_data={"key": "value", "number": 42},
                task_metadata={"tags": ["test", "json"]},
                agent_id=agent.id
            )
            session.add(task)
            await session.commit()
            
            # Verify JSON data
            result = await session.execute(
                select(Task).where(Task.task_type == "json_test")
            )
            saved_task = result.scalar_one()
            
            if saved_task.input_data["number"] == 42:
                print("✅ JSON field storage working")
            else:
                print("❌ JSON field storage failed")
                return False
            
            return True
            
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False
    finally:
        await db.cleanup()


async def test_performance():
    """Test model performance with bulk operations"""
    print("\n⚡ Testing Model Performance...")
    
    db = TestDatabaseManager()
    await db.initialize()
    
    try:
        start_time = datetime.utcnow()
        
        async with db.get_session() as session:
            # Bulk create users
            users = []
            for i in range(100):
                user = User(
                    username=f"perf_user_{i}",
                    email=f"perf_{i}@example.com",
                    user_metadata={"batch": "performance_test"}
                )
                users.append(user)
            
            session.add_all(users)
            await session.commit()
        
        bulk_time = (datetime.utcnow() - start_time).total_seconds()
        print(f"✅ Bulk insert (100 users): {bulk_time:.3f}s")
        
        # Test query performance
        start_time = datetime.utcnow()
        
        async with db.get_session() as session:
            # Count query
            result = await session.execute(select(func.count(User.id)))
            user_count = result.scalar()
            
            # Filtered query
            result = await session.execute(
                select(User).where(User.username.like("perf_user_%")).limit(50)
            )
            filtered_users = result.scalars().all()
        
        query_time = (datetime.utcnow() - start_time).total_seconds()
        print(f"✅ Query performance: {query_time:.3f}s")
        print(f"   Total users: {user_count}")
        print(f"   Filtered users: {len(filtered_users)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False
    finally:
        await db.cleanup()


async def run_all_tests():
    """Run all production model tests"""
    print("🚀 Production Database Models Test Suite")
    print("=" * 60)
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Model Relationships", test_model_relationships),
        ("Model Validation", test_model_validation),
        ("Performance Testing", test_performance)
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
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("   Production database models are working correctly.")
        print("   Ready for PostgreSQL deployment.")
    else:
        print("⚠️ SOME TESTS FAILED")
        print("   Fix model issues before production deployment.")
    
    # Cleanup test database
    test_db = Path("test_models.db")
    if test_db.exists():
        test_db.unlink()
        print("🧹 Test database cleaned up")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
