#!/usr/bin/env python3
"""
Test Agent Instantiation

Test creating real agent instances to identify the exact issues.
"""

import os
import sys
import asyncio
from pathlib import Path

# Environment setup
os.environ.update({
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

async def test_base_agent():
    """Test BaseAgent instantiation"""
    print("1. Testing BaseAgent instantiation...")
    
    try:
        from agents.base_agent import BaseAgent, AgentType
        
        # Try to create BaseAgent directly (should fail - it's abstract)
        try:
            agent = BaseAgent(agent_type=AgentType.RESEARCH, name="TestAgent")
            print("   ❌ BaseAgent should not be instantiable (it's abstract)")
            return False
        except TypeError as e:
            print(f"   ✅ BaseAgent correctly abstract: {e}")
            return True
            
    except Exception as e:
        print(f"   ❌ BaseAgent import failed: {e}")
        return False

async def test_research_agent():
    """Test ResearchAgent instantiation"""
    print("2. Testing ResearchAgent instantiation...")
    
    try:
        from agents.specialized_agents import ResearchAgent
        
        # Create ResearchAgent
        agent = ResearchAgent(name="TestResearchAgent")
        print(f"   ✅ ResearchAgent created: {agent.name}")
        
        # Test initialization
        success = await agent.initialize()
        print(f"   ✅ ResearchAgent initialized: {success}")
        print(f"   ✅ Agent status: {agent.status}")
        print(f"   ✅ Agent ID: {agent.agent_id}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ ResearchAgent creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_analysis_agent():
    """Test AnalysisAgent instantiation"""
    print("3. Testing AnalysisAgent instantiation...")
    
    try:
        from agents.specialized_agents import AnalysisAgent
        
        # Create AnalysisAgent
        agent = AnalysisAgent(name="TestAnalysisAgent")
        print(f"   ✅ AnalysisAgent created: {agent.name}")
        
        # Test initialization
        success = await agent.initialize()
        print(f"   ✅ AnalysisAgent initialized: {success}")
        print(f"   ✅ Agent status: {agent.status}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ AnalysisAgent creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_generation_agent():
    """Test GenerationAgent instantiation"""
    print("4. Testing GenerationAgent instantiation...")
    
    try:
        from agents.specialized_agents import GenerationAgent
        
        # Create GenerationAgent
        agent = GenerationAgent(name="TestGenerationAgent")
        print(f"   ✅ GenerationAgent created: {agent.name}")
        
        # Test initialization
        success = await agent.initialize()
        print(f"   ✅ GenerationAgent initialized: {success}")
        print(f"   ✅ Agent status: {agent.status}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ GenerationAgent creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_agent_capabilities():
    """Test agent capabilities"""
    print("5. Testing Agent Capabilities...")
    
    try:
        from agents.specialized_agents import ResearchAgent
        
        agent = ResearchAgent(name="CapabilityTestAgent")
        await agent.initialize()
        
        print(f"   ✅ Agent has {len(agent.capabilities)} capabilities:")
        for cap in agent.capabilities:
            print(f"      - {cap.name}: {cap.description}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Capability test failed: {e}")
        return False

async def main():
    """Run agent instantiation tests"""
    print("🧪 TESTING AGENT INSTANTIATION")
    print("=" * 40)
    
    tests = [
        test_base_agent,
        test_research_agent,
        test_analysis_agent,
        test_generation_agent,
        test_agent_capabilities
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            result = await test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 40)
    print("📊 AGENT INSTANTIATION TEST SUMMARY")
    print("=" * 40)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL AGENT TESTS PASSED!")
        print("✅ Agents can be instantiated successfully")
        return True
    else:
        print(f"\n⚠️ Some agent tests failed ({passed}/{total})")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
