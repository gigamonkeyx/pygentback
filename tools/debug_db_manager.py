#!/usr/bin/env python3
"""
Debug Database Manager State

Test to understand the database manager initialization sequence.
"""

import os
import sys
import asyncio
from pathlib import Path

# Set environment variables
os.environ["DB_HOST"] = "localhost"
os.environ["DB_PORT"] = "54321"
os.environ["DB_NAME"] = "pygent_factory"
os.environ["DB_USER"] = "postgres"
os.environ["DB_PASSWORD"] = "postgres"
os.environ["DATABASE_URL"] = "postgresql+asyncpg://postgres:postgres@localhost:54321/pygent_factory"

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

async def debug_db_manager():
    """Debug database manager state"""
    print("🔍 DEBUGGING DATABASE MANAGER STATE")
    print("=" * 50)
    
    # Step 1: Import and check initial state
    print("1. Importing database manager...")
    from database.production_manager import db_manager, initialize_database
    
    print(f"   db_manager exists: {db_manager is not None}")
    print(f"   db_manager.is_initialized: {db_manager.is_initialized}")
    print(f"   db_manager.engine: {db_manager.engine}")
    
    # Step 2: Initialize database
    print("\n2. Initializing database...")
    success = await initialize_database()
    print(f"   Initialization success: {success}")
    print(f"   db_manager.is_initialized: {db_manager.is_initialized}")
    print(f"   db_manager.engine: {db_manager.engine is not None}")
    
    # Step 3: Test agent import
    print("\n3. Testing agent import...")
    from agents.specialized_agents import ResearchAgent
    from agents.specialized_agents import db_manager as agent_db_manager
    
    print(f"   Agent db_manager exists: {agent_db_manager is not None}")
    print(f"   Agent db_manager.is_initialized: {agent_db_manager.is_initialized}")
    print(f"   Same instance: {db_manager is agent_db_manager}")
    
    # Step 4: Test agent execution
    print("\n4. Testing agent task execution...")
    agent = ResearchAgent(name="debug_agent")
    await agent.initialize()
    
    try:
        result = await agent._search_documents({"query": "test", "limit": 3})
        print(f"   Agent search result: {result.get('search_method', 'No method')}")
        print("   ✅ Agent task execution successful")
    except Exception as e:
        print(f"   ❌ Agent task execution failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 DEBUG COMPLETE")

if __name__ == "__main__":
    asyncio.run(debug_db_manager())
