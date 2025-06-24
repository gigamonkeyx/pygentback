#!/usr/bin/env python3
"""
Database Integration Test

Tests database operations to ensure proper connectivity and CRUD operations
following Context7 MCP best practices with SQLAlchemy session management.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Change to project root for proper imports
import os
os.chdir(project_root)

from database.session import get_session
from database.models import Agent, AgentMemory, Document
from database.connection import get_db_session
from sqlalchemy.ext.asyncio import AsyncSession


async def test_database_connection():
    """Test basic database connection"""
    print("🧪 Testing database connection...")
    
    try:
        async for session in get_db_session():
            # Simple query to test connection
            result = await session.execute("SELECT 1 as test")
            test_value = result.scalar()
            
            if test_value == 1:
                print("✅ Database connection successful")
                return True
            else:
                print("❌ Database connection failed - unexpected result")
                return False
                
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


async def test_agent_crud_operations():
    """Test CRUD operations on Agent model"""
    print("\n🧪 Testing Agent CRUD operations...")
    
    try:
        async for session in get_db_session():
            # CREATE - Create a test agent
            test_agent = Agent(
                name="test_agent_crud",
                agent_type="reasoning",
                description="Test agent for CRUD operations",
                config={"test": True, "model": "gpt-4"},
                status="active"
            )
            
            session.add(test_agent)
            await session.commit()
            await session.refresh(test_agent)
            
            agent_id = test_agent.id
            print(f"✅ CREATE: Agent created with ID {agent_id}")
            
            # READ - Retrieve the agent
            retrieved_agent = await session.get(Agent, agent_id)
            if retrieved_agent and retrieved_agent.name == "test_agent_crud":
                print(f"✅ READ: Agent retrieved successfully")
            else:
                print("❌ READ: Failed to retrieve agent")
                return False
            
            # UPDATE - Modify the agent
            retrieved_agent.description = "Updated test agent description"
            retrieved_agent.config = {"test": True, "model": "gpt-4", "updated": True}
            await session.commit()
            
            # Verify update
            await session.refresh(retrieved_agent)
            if "updated" in retrieved_agent.config:
                print("✅ UPDATE: Agent updated successfully")
            else:
                print("❌ UPDATE: Failed to update agent")
                return False
            
            # DELETE - Remove the agent
            await session.delete(retrieved_agent)
            await session.commit()
            
            # Verify deletion
            deleted_agent = await session.get(Agent, agent_id)
            if deleted_agent is None:
                print("✅ DELETE: Agent deleted successfully")
            else:
                print("❌ DELETE: Failed to delete agent")
                return False
            
            return True
            
    except Exception as e:
        print(f"❌ Agent CRUD operations failed: {e}")
        return False


async def test_memory_operations():
    """Test memory storage and retrieval operations"""
    print("\n🧪 Testing Memory operations...")
    
    try:
        async for session in get_db_session():
            # Create test memories
            memories = [
                Memory(
                    content="Test memory 1: Python is a programming language",
                    memory_type="fact",
                    source="test_system",
                    metadata={"category": "programming", "confidence": 0.9}
                ),
                Memory(
                    content="Test memory 2: The sky is blue during the day",
                    memory_type="fact",
                    source="test_system",
                    metadata={"category": "nature", "confidence": 0.95}
                ),
                Memory(
                    content="Test memory 3: User prefers JSON format for API responses",
                    memory_type="preference",
                    source="user_interaction",
                    metadata={"category": "user_preference", "confidence": 0.8}
                )
            ]
            
            # Add memories to session
            for memory in memories:
                session.add(memory)
            
            await session.commit()
            
            # Refresh to get IDs
            for memory in memories:
                await session.refresh(memory)
            
            print(f"✅ Created {len(memories)} test memories")
            
            # Test memory retrieval by type
            from sqlalchemy import select
            
            fact_memories = await session.execute(
                select(Memory).where(Memory.memory_type == "fact")
            )
            fact_count = len(fact_memories.scalars().all())
            
            if fact_count >= 2:
                print(f"✅ Retrieved {fact_count} fact memories")
            else:
                print(f"❌ Expected at least 2 fact memories, got {fact_count}")
                return False
            
            # Test memory search by content
            search_result = await session.execute(
                select(Memory).where(Memory.content.contains("Python"))
            )
            python_memories = search_result.scalars().all()
            
            if len(python_memories) >= 1:
                print(f"✅ Found {len(python_memories)} memories containing 'Python'")
            else:
                print("❌ Failed to find memories containing 'Python'")
                return False
            
            # Cleanup - delete test memories
            for memory in memories:
                await session.delete(memory)
            await session.commit()
            
            print("✅ Cleaned up test memories")
            return True
            
    except Exception as e:
        print(f"❌ Memory operations failed: {e}")
        return False


async def test_task_operations():
    """Test task storage and management operations"""
    print("\n🧪 Testing Task operations...")
    
    try:
        async for session in get_db_session():
            # Create a test task
            test_task = Task(
                name="test_database_task",
                description="Test task for database operations",
                task_type="test",
                status="pending",
                priority=1,
                input_data={"test_input": "sample data"},
                metadata={"created_by": "test_system"}
            )
            
            session.add(test_task)
            await session.commit()
            await session.refresh(test_task)
            
            task_id = test_task.id
            print(f"✅ Created test task with ID {task_id}")
            
            # Test task status update
            test_task.status = "running"
            test_task.metadata = {**test_task.metadata, "started_at": "2024-01-01T00:00:00"}
            await session.commit()
            
            # Verify status update
            await session.refresh(test_task)
            if test_task.status == "running":
                print("✅ Task status updated successfully")
            else:
                print("❌ Failed to update task status")
                return False
            
            # Test task completion
            test_task.status = "completed"
            test_task.result_data = {"output": "test completed successfully"}
            await session.commit()
            
            # Verify completion
            await session.refresh(test_task)
            if test_task.status == "completed" and test_task.result_data:
                print("✅ Task completed successfully")
            else:
                print("❌ Failed to complete task")
                return False
            
            # Cleanup
            await session.delete(test_task)
            await session.commit()
            
            print("✅ Cleaned up test task")
            return True
            
    except Exception as e:
        print(f"❌ Task operations failed: {e}")
        return False


async def test_session_management():
    """Test proper session management and cleanup"""
    print("\n🧪 Testing session management...")
    
    try:
        # Test multiple concurrent sessions
        sessions_created = 0
        
        async def create_test_session():
            nonlocal sessions_created
            async for session in get_db_session():
                sessions_created += 1
                # Simple operation to ensure session works
                result = await session.execute("SELECT 1")
                return result.scalar() == 1
        
        # Create multiple sessions concurrently
        tasks = [create_test_session() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        if all(results) and sessions_created == 5:
            print(f"✅ Successfully created and managed {sessions_created} concurrent sessions")
            return True
        else:
            print(f"❌ Session management failed: {sessions_created} sessions, results: {results}")
            return False
            
    except Exception as e:
        print(f"❌ Session management test failed: {e}")
        return False


async def run_all_tests():
    """Run all database tests"""
    print("🗄️ PyGent Factory Database Integration Tests")
    print("=" * 60)
    
    test_results = {
        "database_connection": False,
        "agent_crud": False,
        "memory_operations": False,
        "task_operations": False,
        "session_management": False
    }
    
    # Test 1: Database Connection
    test_results["database_connection"] = await test_database_connection()
    
    # Test 2: Agent CRUD Operations
    test_results["agent_crud"] = await test_agent_crud_operations()
    
    # Test 3: Memory Operations
    test_results["memory_operations"] = await test_memory_operations()
    
    # Test 4: Task Operations
    test_results["task_operations"] = await test_task_operations()
    
    # Test 5: Session Management
    test_results["session_management"] = await test_session_management()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
        if passed:
            passed_tests += 1
    
    print(f"\n🎯 Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All database tests passed!")
        return True
    else:
        print("💥 Some database tests failed!")
        return False


async def main():
    """Main test function"""
    try:
        success = await run_all_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {e}")
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())
