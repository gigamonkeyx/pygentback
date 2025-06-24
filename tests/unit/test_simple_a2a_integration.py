#!/usr/bin/env python3
"""
Simple A2A Integration Test

Quick test to validate A2A integration components.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test basic imports"""
    print("🧪 Testing imports...")
    
    try:
        from src.orchestration.orchestration_manager import OrchestrationManager
        print("✅ OrchestrationManager import successful")
    except Exception as e:
        print(f"❌ OrchestrationManager import failed: {e}")
        return False
    
    try:
        from src.orchestration.a2a_coordination_strategies import A2ACoordinationEngine, CoordinationStrategy
        print("✅ A2A coordination strategies import successful")
    except Exception as e:
        print(f"❌ A2A coordination strategies import failed: {e}")
        return False
    
    return True

def test_coordination_strategies():
    """Test coordination strategies"""
    print("\n🧪 Testing coordination strategies...")
    
    try:
        from src.orchestration.a2a_coordination_strategies import CoordinationStrategy
        
        strategies = [
            CoordinationStrategy.SEQUENTIAL,
            CoordinationStrategy.PARALLEL,
            CoordinationStrategy.HIERARCHICAL,
            CoordinationStrategy.PIPELINE,
            CoordinationStrategy.CONSENSUS,
            CoordinationStrategy.AUCTION,
            CoordinationStrategy.SWARM
        ]
        
        print(f"✅ {len(strategies)} coordination strategies available:")
        for strategy in strategies:
            print(f"   • {strategy.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Coordination strategies test failed: {e}")
        return False

def test_orchestration_manager():
    """Test orchestration manager creation"""
    print("\n🧪 Testing orchestration manager...")
    
    try:
        from src.orchestration.orchestration_manager import OrchestrationManager
        
        # Import and create real A2A manager
        from src.a2a_protocol.manager import A2AManager

        real_a2a = A2AManager()
        
        # Create mock config
        class MockConfig:
            def __init__(self):
                self.max_concurrent_tasks = 10
                self.task_timeout_seconds = 300
                self.metrics_collection_interval = 30
                self.health_check_interval = 60
                self.cleanup_interval = 3600
                self.max_task_history = 1000
                self.enable_performance_optimization = True
                self.enable_adaptive_scheduling = True
                self.enable_load_balancing = True
                self.enable_fault_tolerance = True
                self.enable_metrics_collection = True
                self.enable_health_monitoring = True
                self.enable_documentation_orchestration = True
                self.enable_research_orchestration = True
                self.performance_alerts_enabled = True
                self.alert_thresholds = {
                    'high_cpu_usage': 80.0,
                    'high_memory_usage': 85.0,
                    'low_success_rate': 0.8,
                    'high_error_rate': 0.1,
                    'high_response_time': 5.0
                }

            def validate(self):
                return True
        
        mock_config = MockConfig()

        # Create orchestration manager with real A2A manager
        orchestration_manager = OrchestrationManager(
            config=mock_config,
            a2a_manager=real_a2a
        )
        
        print("✅ OrchestrationManager created successfully")
        
        # Check A2A integration
        if hasattr(orchestration_manager, 'a2a_manager'):
            print("✅ A2A manager integrated")
        else:
            print("❌ A2A manager not integrated")
            return False
        
        # Check coordination engine
        if hasattr(orchestration_manager, 'a2a_coordination_engine'):
            print("✅ A2A coordination engine available")
        else:
            print("❌ A2A coordination engine not available")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ OrchestrationManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_endpoints():
    """Test API endpoints"""
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
        
        missing_endpoints = []
        for endpoint in required_endpoints:
            if endpoint not in content:
                missing_endpoints.append(endpoint)
        
        if missing_endpoints:
            print(f"❌ Missing endpoints: {missing_endpoints}")
            return False
        else:
            print(f"✅ All {len(required_endpoints)} endpoints present")
        
        # Check for endpoint functions
        required_functions = [
            "execute_coordination_strategy",
            "get_coordination_strategies",
            "get_coordination_performance",
            "execute_multi_strategy_workflow"
        ]
        
        missing_functions = []
        for function in required_functions:
            if f"async def {function}" not in content:
                missing_functions.append(function)
        
        if missing_functions:
            print(f"❌ Missing functions: {missing_functions}")
            return False
        else:
            print(f"✅ All {len(required_functions)} functions present")
        
        return True
        
    except Exception as e:
        print(f"❌ API endpoints test failed: {e}")
        return False

def main():
    """Main test runner"""
    print("🚀 Starting Simple A2A Integration Tests")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Coordination Strategies", test_coordination_strategies),
        ("Orchestration Manager", test_orchestration_manager),
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
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📈 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL TESTS PASSED!")
        print("✅ A2A integration is working correctly")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
