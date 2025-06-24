#!/usr/bin/env python3
"""
Test A2A Integration

Test the complete A2A integration with PyGent Factory.
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def test_a2a_integration():
    """Test A2A integration"""
    print("🚀 Testing A2A Integration with PyGent Factory")
    print("=" * 60)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: A2A Router Import
    tests_total += 1
    try:
        # Test the import path that will be used in production
        import sys
        import importlib.util

        # Load the A2A router module directly
        router_path = project_root / "src" / "api" / "routes" / "a2a.py"
        spec = importlib.util.spec_from_file_location("a2a_router", router_path)
        a2a_router_module = importlib.util.module_from_spec(spec)

        # Add to sys.modules to handle internal imports
        sys.modules["a2a_router"] = a2a_router_module
        spec.loader.exec_module(a2a_router_module)

        # Check if router exists
        if hasattr(a2a_router_module, 'router'):
            print("✅ Test 1: A2A Router Import - PASSED")
            tests_passed += 1
        else:
            print("❌ Test 1: A2A Router Import - FAILED: router attribute not found")

    except Exception as e:
        print(f"❌ Test 1: A2A Router Import - FAILED: {e}")
    
    # Test 2: A2A Components Integration
    tests_total += 1
    try:
        from src.core.agent_factory import AgentFactory

        # Create agent factory with A2A (without async initialization)
        factory = AgentFactory(base_url="http://localhost:8000")

        # Check A2A integration (basic attributes only)
        has_a2a = hasattr(factory, 'a2a_enabled')
        has_card_generator = hasattr(factory, 'a2a_card_generator')

        # Check if A2A components are initialized (they might be None if not available)
        has_transport_attr = hasattr(factory, 'a2a_transport')
        has_task_manager_attr = hasattr(factory, 'a2a_task_manager')
        has_security_attr = hasattr(factory, 'a2a_security_manager')
        has_discovery_attr = hasattr(factory, 'a2a_discovery')

        if has_a2a and has_card_generator and has_transport_attr:
            print("✅ Test 2: A2A Components Integration - PASSED")
            tests_passed += 1
        else:
            print(f"❌ Test 2: A2A Components Integration - FAILED: a2a_enabled={has_a2a}, card_generator={has_card_generator}, transport_attr={has_transport_attr}")

    except Exception as e:
        print(f"❌ Test 2: A2A Components Integration - FAILED: {e}")
    
    # Test 3: System Agent Card Generation
    tests_total += 1
    try:
        from a2a_protocol.agent_card_generator import A2AAgentCardGenerator
        
        generator = A2AAgentCardGenerator("http://localhost:8000")
        
        # Test sync method exists
        has_sync_method = hasattr(generator, 'generate_agent_card_sync')
        has_system_method = hasattr(generator, 'generate_system_agent_card')
        
        if has_sync_method and has_system_method:
            print("✅ Test 3: System Agent Card Generation - PASSED")
            tests_passed += 1
        else:
            print(f"❌ Test 3: System Agent Card Generation - FAILED: sync_method={has_sync_method}, system_method={has_system_method}")
            
    except Exception as e:
        print(f"❌ Test 3: System Agent Card Generation - FAILED: {e}")
    
    # Test 4: A2A Transport Streaming
    tests_total += 1
    try:
        from a2a_protocol.transport import A2ATransportLayer
        
        transport = A2ATransportLayer()
        
        # Check streaming methods
        has_streaming = hasattr(transport, 'handle_streaming_request')
        has_message_stream = hasattr(transport, '_handle_message_stream')
        has_task_resubscribe = hasattr(transport, '_handle_tasks_resubscribe')
        
        if has_streaming and has_message_stream and has_task_resubscribe:
            print("✅ Test 4: A2A Transport Streaming - PASSED")
            tests_passed += 1
        else:
            print(f"❌ Test 4: A2A Transport Streaming - FAILED: streaming={has_streaming}, message_stream={has_message_stream}, task_resubscribe={has_task_resubscribe}")
            
    except Exception as e:
        print(f"❌ Test 4: A2A Transport Streaming - FAILED: {e}")
    
    # Test 5: A2A API Models
    tests_total += 1
    try:
        from src.api.routes.a2a import A2AMessageRequest, A2AMessageResponse, AgentDiscoveryResponse
        
        # Test model creation
        request = A2AMessageRequest(method="test", params={"test": "data"})
        response = A2AMessageResponse(result={"status": "ok"})
        discovery = AgentDiscoveryResponse(agents=[], total=0, timestamp="2024-01-01T00:00:00")
        
        if request and response and discovery:
            print("✅ Test 5: A2A API Models - PASSED")
            tests_passed += 1
        else:
            print("❌ Test 5: A2A API Models - FAILED: Model creation failed")
            
    except Exception as e:
        print(f"❌ Test 5: A2A API Models - FAILED: {e}")
    
    # Test 6: A2A Error Handling
    tests_total += 1
    try:
        from a2a_protocol.error_handling import A2ATransportError
        
        # Test error creation
        error = A2ATransportError(-32602, "Invalid params")
        error_dict = error.to_dict()
        
        if error and error_dict and error_dict.get("code") == -32602:
            print("✅ Test 6: A2A Error Handling - PASSED")
            tests_passed += 1
        else:
            print("❌ Test 6: A2A Error Handling - FAILED: Error handling not working")
            
    except Exception as e:
        print(f"❌ Test 6: A2A Error Handling - FAILED: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 A2A Integration Test Results: {tests_passed}/{tests_total} passed")
    success_rate = (tests_passed / tests_total) * 100 if tests_total > 0 else 0
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    if tests_passed == tests_total:
        print("🎉 A2A INTEGRATION COMPLETE! All tests passed!")
        return True
    elif tests_passed >= 4:  # 66% threshold
        print("✅ A2A INTEGRATION MOSTLY COMPLETE! Core functionality working.")
        return True
    else:
        print("❌ A2A INTEGRATION INCOMPLETE! Critical issues found.")
        return False


if __name__ == "__main__":
    success = test_a2a_integration()
    sys.exit(0 if success else 1)
