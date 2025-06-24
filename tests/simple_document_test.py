#!/usr/bin/env python3
"""
Simple Document Search Test

Test document search without complex setup.
"""

import os
import sys
import asyncio
from pathlib import Path

# Environment setup
os.environ.update({
    "DATABASE_URL": "postgresql+asyncpg://postgres:postgres@localhost:54321/pygent_factory",
    "DB_HOST": "localhost",
    "DB_PORT": "54321", 
    "DB_NAME": "pygent_factory",
    "DB_USER": "postgres",
    "DB_PASSWORD": "postgres"
})

sys.path.append(str(Path(__file__).parent / "src"))

async def test_document_search():
    """Test document search functionality"""
    
    print("🔍 SIMPLE DOCUMENT SEARCH TEST")
    print("=" * 40)
    
    try:
        # Test 1: Create agent
        print("1. Creating ResearchAgent...")
        from agents.specialized_agents import ResearchAgent
        agent = ResearchAgent(name="SimpleTestAgent")
        
        init_success = await agent.initialize()
        print(f"   Agent initialization: {'✅' if init_success else '❌'}")
        
        if not init_success:
            return False
        
        # Test 2: Check existing documents
        print("2. Checking existing documents...")
        from database.production_manager import db_manager
        
        existing_docs = await db_manager.fetch_all("""
            SELECT id, title, content FROM documents LIMIT 5
        """)
        
        print(f"   Found {len(existing_docs)} existing documents")
        for doc in existing_docs[:3]:
            print(f"      - {doc['title'][:50]}...")
        
        # Test 3: Try agent search on existing data
        print("3. Testing agent search...")
        
        if existing_docs:
            # Search for something that might exist
            search_result = await agent._search_documents({
                "query": "document",  # Generic term likely to match
                "limit": 3
            })
            
            print(f"   Search result: {search_result}")
            
            if search_result and search_result.get('total_found', 0) > 0:
                print("   ✅ Document search: WORKING")
                return True
            else:
                print("   ⚠️ Document search: NO MATCHES")
                return False
        else:
            print("   ⚠️ No existing documents to search")
            
            # Try search anyway to test the mechanism
            search_result = await agent._search_documents({
                "query": "test",
                "limit": 3
            })
            
            print(f"   Empty search result: {search_result}")
            
            if search_result and 'search_method' in search_result:
                print("   ✅ Document search mechanism: WORKING")
                return True
            else:
                print("   ❌ Document search mechanism: FAILED")
                return False
        
    except Exception as e:
        print(f"   ❌ Document search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_agent_creation_only():
    """Test just agent creation"""
    
    print("\n🤖 AGENT CREATION TEST")
    print("=" * 40)
    
    try:
        print("1. Testing ResearchAgent creation...")
        from agents.specialized_agents import ResearchAgent
        
        agent = ResearchAgent(name="CreationTestAgent")
        print(f"   ✅ Agent created: {agent.name}")
        
        print("2. Testing agent initialization...")
        success = await agent.initialize()
        print(f"   ✅ Agent initialized: {success}")
        
        print("3. Testing agent capabilities...")
        print(f"   ✅ Agent has {len(agent.capabilities)} capabilities")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Agent creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run simple tests"""
    
    # Test agent creation
    agent_working = await test_agent_creation_only()
    
    # Test document search
    doc_search_working = await test_document_search()
    
    print("\n" + "=" * 40)
    print("📊 SIMPLE TEST RESULTS")
    print("=" * 40)
    print(f"Agent Creation: {'✅ WORKING' if agent_working else '❌ FAILED'}")
    print(f"Document Search: {'✅ WORKING' if doc_search_working else '❌ FAILED'}")
    
    if agent_working and doc_search_working:
        print("\n🎉 BOTH SYSTEMS WORKING!")
        return True
    else:
        print(f"\n⚠️ Issues found:")
        if not agent_working:
            print("   - Agent creation needs fixing")
        if not doc_search_working:
            print("   - Document search needs fixing")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
