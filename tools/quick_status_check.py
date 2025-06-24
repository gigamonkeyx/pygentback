#!/usr/bin/env python3
"""
Quick Status Check

Check the current status of agent system fixes.
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
    "DB_PASSWORD": "postgres",
    "REDIS_HOST": "localhost",
    "REDIS_PORT": "6379",
    "REDIS_DB": "0"
})

sys.path.append(str(Path(__file__).parent / "src"))

async def quick_check():
    """Quick status check"""
    
    print("🔍 QUICK STATUS CHECK")
    print("=" * 30)
    
    # Test 1: Basic imports
    print("1. Testing imports...")
    try:
        from agents.specialized_agents import ResearchAgent
        print("   ✅ ResearchAgent import: OK")
    except Exception as e:
        print(f"   ❌ ResearchAgent import: {e}")
        return False
    
    # Test 2: Agent creation
    print("2. Testing agent creation...")
    try:
        agent = ResearchAgent(name="QuickTestAgent")
        print("   ✅ Agent creation: OK")
    except Exception as e:
        print(f"   ❌ Agent creation: {e}")
        return False
    
    # Test 3: Agent initialization
    print("3. Testing agent initialization...")
    try:
        success = await agent.initialize()
        if success:
            print("   ✅ Agent initialization: OK")
        else:
            print("   ❌ Agent initialization: FAILED")
            return False
    except Exception as e:
        print(f"   ❌ Agent initialization: {e}")
        return False
    
    # Test 4: Database access
    print("4. Testing database access...")
    try:
        result = await agent._search_documents({"query": "test", "limit": 1})
        print(f"   ✅ Database access: OK - {result.get('search_method', 'unknown')}")
    except Exception as e:
        print(f"   ❌ Database access: {e}")
        return False
    
    print("\n✅ ALL QUICK CHECKS PASSED!")
    return True

if __name__ == "__main__":
    success = asyncio.run(quick_check())
    print(f"\nStatus: {'SUCCESS' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
