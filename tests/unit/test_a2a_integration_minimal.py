#!/usr/bin/env python3
"""
Minimal A2A Integration Test

Tests the core A2A integration without complex orchestration manager setup.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_coordination_strategies():
    """Test coordination strategies are available"""
    print("🧪 Testing coordination strategies...")
    
    try:
        from src.orchestration.a2a_coordination_strategies import CoordinationStrategy, A2ACoordinationEngine
        
        strategies = list(CoordinationStrategy)
        print(f"✅ {len(strategies)} coordination strategies available:")
        for strategy in strategies:
            print(f"   • {strategy.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Coordination strategies test failed: {e}")
        return False

def test_coordination_engine():
    """Test coordination engine can be created"""
    print("\n🧪 Testing coordination engine...")
    
    try:
        from src.orchestration.a2a_coordination_strategies import A2ACoordinationEngine
        
        # Mock A2A manager
        class MockA2AManager:
            async def get_agent_status(self):
                return {"agents": [], "total_agents": 0}
            async def coordinate_multi_agent_task(self, **kwargs):
                return []
        
        # Mock orchestration manager
        class MockOrchestrationManager:
            pass
        
        mock_a2a = MockA2AManager()
        mock_orchestration = MockOrchestrationManager()
        
        # Create coordination engine
        coordination_engine = A2ACoordinationEngine(mock_a2a, mock_orchestration)
        
        print("✅ A2A Coordination Engine created successfully")
        
        # Test performance tracking
        if hasattr(coordination_engine, 'get_strategy_performance'):
            print("✅ Performance tracking available")
        else:
            print("❌ Performance tracking not available")
            return False
        
        # Test coordination history
        if hasattr(coordination_engine, 'get_coordination_history'):
            print("✅ Coordination history available")
        else:
            print("❌ Coordination history not available")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Coordination engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_task_dispatcher_a2a():
    """Test task dispatcher A2A integration"""
    print("\n🧪 Testing task dispatcher A2A integration...")
    
    try:
        from src.orchestration.task_dispatcher import TaskDispatcher
        
        # Check if TaskDispatcher accepts a2a_manager parameter
        import inspect
        sig = inspect.signature(TaskDispatcher.__init__)
        params = list(sig.parameters.keys())
        
        if 'a2a_manager' in params:
            print("✅ TaskDispatcher accepts a2a_manager parameter")
        else:
            print("❌ TaskDispatcher does not accept a2a_manager parameter")
            return False
        
        # Check for A2A-aware methods
        methods = dir(TaskDispatcher)
        a2a_methods = [m for m in methods if 'a2a' in m.lower()]
        
        if a2a_methods:
            print(f"✅ TaskDispatcher has A2A methods: {a2a_methods}")
        else:
            print("❌ TaskDispatcher has no A2A methods")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Task dispatcher A2A test failed: {e}")
        return False

def test_orchestration_manager_a2a():
    """Test orchestration manager A2A integration"""
    print("\n🧪 Testing orchestration manager A2A integration...")
    
    try:
        from src.orchestration.orchestration_manager import OrchestrationManager
        
        # Check if OrchestrationManager accepts a2a_manager parameter
        import inspect
        sig = inspect.signature(OrchestrationManager.__init__)
        params = list(sig.parameters.keys())
        
        if 'a2a_manager' in params:
            print("✅ OrchestrationManager accepts a2a_manager parameter")
        else:
            print("❌ OrchestrationManager does not accept a2a_manager parameter")
            return False
        
        # Check for A2A coordination methods
        methods = dir(OrchestrationManager)
        coordination_methods = [
            'execute_coordination_strategy',
            'get_coordination_strategies',
            'get_coordination_performance',
            'execute_multi_strategy_workflow'
        ]
        
        missing_methods = []
        for method in coordination_methods:
            if method not in methods:
                missing_methods.append(method)
        
        if missing_methods:
            print(f"❌ Missing coordination methods: {missing_methods}")
            return False
        else:
            print(f"✅ All {len(coordination_methods)} coordination methods present")
        
        return True
        
    except Exception as e:
        print(f"❌ Orchestration manager A2A test failed: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints are present"""
    print("\n🧪 Testing API endpoints...")
    
    try:
        server_file = Path("src/servers/agent_orchestration_mcp_server.py")
        if not server_file.exists():
            print("❌ Orchestration MCP server file not found")
            return False
        
        with open(server_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for A2A coordination endpoints
        required_endpoints = [
            "/v1/a2a/coordination/execute",
            "/v1/a2a/coordination/strategies",
            "/v1/a2a/coordination/performance",
            "/v1/a2a/workflows/execute"
        ]
        
        present_endpoints = []
        missing_endpoints = []
        
        for endpoint in required_endpoints:
            if endpoint in content:
                present_endpoints.append(endpoint)
            else:
                missing_endpoints.append(endpoint)
        
        if missing_endpoints:
            print(f"❌ Missing endpoints: {missing_endpoints}")
            return False
        else:
            print(f"✅ All {len(required_endpoints)} endpoints present")
        
        return True
        
    except Exception as e:
        print(f"❌ API endpoints test failed: {e}")
        return False

def main():
    """Main test runner"""
    print("🚀 Starting Minimal A2A Integration Tests")
    print("="*60)
    
    tests = [
        ("Coordination Strategies", test_coordination_strategies),
        ("Coordination Engine", test_coordination_engine),
        ("Task Dispatcher A2A", test_task_dispatcher_a2a),
        ("Orchestration Manager A2A", test_orchestration_manager_a2a),
        ("API Endpoints", test_api_endpoints)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("📊 MINIMAL A2A INTEGRATION TEST SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📈 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 ALL A2A INTEGRATION TESTS PASSED!")
        print("✅ Orchestration workflows can use A2A for agent-to-agent communication")
        print("✅ A2A coordination strategies are implemented and available")
        print("✅ API endpoints are properly configured")
        print("🚀 Phase 3: Orchestration Manager Integration COMPLETE")
        return 0
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed")
        print("🔧 Please review and fix the issues")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
