#!/usr/bin/env python3
"""
Debug Agent Creation and Document Search Failures

Let's find out exactly what's failing and fix it.
"""

import os
import sys
import asyncio
import traceback
from pathlib import Path

# Environment setup
os.environ.update({
    "DATABASE_URL": "postgresql+asyncpg://postgres:postgres@localhost:54321/pygent_factory",
    "DB_HOST": "localhost",
    "DB_PORT": "54321", 
    "DB_NAME": "pygent_factory",
    "DB_USER": "postgres",
    "DB_PASSWORD": "postgres",
    "REDIS_HOST": "localhost",
    "REDIS_PORT": "6379",
    "REDIS_DB": "0"
})

sys.path.append(str(Path(__file__).parent / "src"))

async def debug_agent_creation():
    """Debug agent creation step by step"""
    print("🔍 DEBUGGING AGENT CREATION")
    print("=" * 40)
    
    try:
        print("Step 1: Testing imports...")
        from agents.specialized_agents import ResearchAgent
        print("   ✅ ResearchAgent imported successfully")
        
        print("Step 2: Testing agent instantiation...")
        agent = ResearchAgent(name="DebugAgent")
        print("   ✅ ResearchAgent instantiated successfully")
        print(f"   Agent ID: {agent.agent_id}")
        print(f"   Agent Name: {agent.name}")
        print(f"   Agent Type: {agent.agent_type}")
        print(f"   Agent Status: {agent.status}")
        
        print("Step 3: Testing agent initialization...")
        success = await agent.initialize()
        print(f"   Initialization result: {success}")
        print(f"   Agent status after init: {agent.status}")
        print(f"   Agent capabilities: {len(agent.capabilities)}")
        
        if success:
            print("   ✅ Agent creation: FULLY WORKING")
            return agent
        else:
            print("   ❌ Agent initialization failed")
            return None
            
    except Exception as e:
        print(f"   ❌ Agent creation failed: {e}")
        print("   Full traceback:")
        traceback.print_exc()
        return None

async def debug_document_search(agent):
    """Debug document search step by step"""
    print("\n🔍 DEBUGGING DOCUMENT SEARCH")
    print("=" * 40)
    
    if not agent:
        print("   ❌ No agent available for testing")
        return False
    
    try:
        print("Step 1: Testing database initialization...")
        from database.production_manager import db_manager, initialize_database
        
        if not db_manager.is_initialized:
            print("   Database not initialized, initializing...")
            db_success = await initialize_database()
            print(f"   Database initialization: {db_success}")
        else:
            print("   ✅ Database already initialized")
        
        print("Step 2: Testing database connection...")
        health = await db_manager.health_check()
        print(f"   Database health: {health}")
        
        print("Step 3: Testing document table...")
        # Check if documents table exists
        table_check = await db_manager.fetch_all("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_name = 'documents'
        """)
        
        if table_check:
            print("   ✅ Documents table exists")
            
            # Check table structure
            columns = await db_manager.fetch_all("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'documents'
                ORDER BY ordinal_position
            """)
            
            print("   Table columns:")
            for col in columns:
                print(f"      - {col['column_name']}: {col['data_type']}")
        else:
            print("   ❌ Documents table does not exist")
            return False
        
        print("Step 4: Creating test user and document...")
        # First create a test user
        import uuid
        test_user_id = uuid.uuid4()
        test_doc_id = uuid.uuid4()

        # Create test user
        await db_manager.execute_command("""
            INSERT INTO users (id, username, email, created_at, updated_at)
            VALUES ($1, $2, $3, NOW(), NOW())
            ON CONFLICT (id) DO NOTHING
        """, test_user_id, "test_user", "test@example.com")

        # Insert a test document using correct schema
        await db_manager.execute_command("""
            INSERT INTO documents (id, user_id, title, content, source_url, source_type, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, NOW(), NOW())
            ON CONFLICT (id) DO UPDATE SET
                content = EXCLUDED.content,
                updated_at = NOW()
        """, test_doc_id, test_user_id, "Test Document", "This is a test document for debugging agent communication.", "test_source", "debug")
        
        print("   ✅ Test document added")
        
        print("Step 5: Testing direct database search...")
        # Test direct database query
        results = await db_manager.fetch_all("""
            SELECT id, title, content FROM documents 
            WHERE content ILIKE $1 
            LIMIT 3
        """, "%test%")
        
        print(f"   Direct query results: {len(results)} documents found")
        for doc in results:
            print(f"      - {doc['title']} (ID: {doc['id']})")
        
        print("Step 6: Testing agent document search...")
        # Test agent search method
        search_result = await agent._search_documents({
            "query": "test document",
            "limit": 3
        })
        
        print(f"   Agent search result: {search_result}")
        
        if search_result and search_result.get('total_found', 0) > 0:
            print("   ✅ Document search: FULLY WORKING")
            return True
        else:
            print("   ❌ Document search: NO RESULTS")
            return False
            
    except Exception as e:
        print(f"   ❌ Document search failed: {e}")
        print("   Full traceback:")
        traceback.print_exc()
        return False

async def debug_communication_system():
    """Debug communication system"""
    print("\n🔍 DEBUGGING COMMUNICATION SYSTEM")
    print("=" * 40)
    
    try:
        print("Step 1: Testing communication imports...")
        from agents.communication_system import MultiAgentCommunicationSystem
        from agents.base_agent import AgentMessage, MessageType
        print("   ✅ Communication imports successful")
        
        print("Step 2: Testing communication system creation...")
        comm_system = MultiAgentCommunicationSystem()
        print("   ✅ Communication system created")
        
        print("Step 3: Testing communication system initialization...")
        success = await comm_system.initialize()
        print(f"   Communication system initialization: {success}")
        
        if success:
            print("   ✅ Communication system: WORKING")
            return True
        else:
            print("   ❌ Communication system: FAILED")
            return False
            
    except Exception as e:
        print(f"   ❌ Communication system failed: {e}")
        print("   Full traceback:")
        traceback.print_exc()
        return False

async def main():
    """Run complete debugging"""
    print("🚨 DEBUGGING AGENT FAILURES")
    print("=" * 50)
    
    # Debug agent creation
    agent = await debug_agent_creation()
    
    # Debug document search
    doc_search_working = await debug_document_search(agent)
    
    # Debug communication system
    comm_working = await debug_communication_system()
    
    print("\n" + "=" * 50)
    print("🔍 DEBUGGING SUMMARY")
    print("=" * 50)
    print(f"Agent Creation: {'✅ WORKING' if agent else '❌ FAILED'}")
    print(f"Document Search: {'✅ WORKING' if doc_search_working else '❌ FAILED'}")
    print(f"Communication System: {'✅ WORKING' if comm_working else '❌ FAILED'}")
    
    if agent and doc_search_working and comm_working:
        print("\n🎉 ALL SYSTEMS WORKING!")
        return True
    else:
        print("\n❌ ISSUES FOUND - NEED FIXES")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
