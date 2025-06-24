#!/usr/bin/env python3
"""
Quick A2A Test

A simple test to verify the A2A implementation is working.
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def test_a2a_components():
    """Test A2A components"""
    print("🚀 Testing A2A Implementation")
    print("=" * 50)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: A2A Standard Types
    tests_total += 1
    try:
        from a2a_standard import AgentProvider, AgentCard, AgentCapabilities, AgentSkill, TaskState
        
        # Test creating AgentProvider with organization
        provider = AgentProvider(
            name="Test Provider",
            organization="Test Org",
            description="Test provider",
            url="https://test.com"
        )
        
        print("✅ Test 1: A2A Standard Types - PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 1: A2A Standard Types - FAILED: {e}")
    
    # Test 2: Agent Card Generator
    tests_total += 1
    try:
        from a2a_protocol.agent_card_generator import A2AAgentCardGenerator
        
        generator = A2AAgentCardGenerator(base_url="http://localhost:8000")
        
        # Test sync card generation
        agent_card = generator.generate_agent_card_sync(
            agent_id="test_agent_123",
            agent_name="Test Agent",
            agent_type="general",
            capabilities=["reasoning", "analysis"],
            skills=["problem_solving"],
            enable_authentication=True
        )
        
        # Validate card structure
        required_fields = ["name", "description", "url", "capabilities", "skills", "provider"]
        missing_fields = [field for field in required_fields if field not in agent_card]
        
        if len(missing_fields) == 0 and agent_card.get("name") == "Test Agent":
            print("✅ Test 2: Agent Card Generator - PASSED")
            tests_passed += 1
        else:
            print(f"❌ Test 2: Agent Card Generator - FAILED: Missing fields {missing_fields}")
            
    except Exception as e:
        print(f"❌ Test 2: Agent Card Generator - FAILED: {e}")
    
    # Test 3: Task Manager
    tests_total += 1
    try:
        from a2a_protocol.task_manager import A2ATaskManager, TaskState
        
        task_manager = A2ATaskManager()
        
        # Create a task
        task = task_manager.create_task_sync(
            task_id="test_task_123",
            context_id="test_context",
            message_content="Test task content"
        )
        
        # Update task status
        success = task_manager.update_task_status_sync(
            task_id="test_task_123",
            state=TaskState.WORKING,
            message="Task is working"
        )
        
        # Get task
        retrieved_task = task_manager.get_task_sync("test_task_123")
        
        if task and success and retrieved_task and retrieved_task.id == "test_task_123":
            print("✅ Test 3: Task Manager - PASSED")
            tests_passed += 1
        else:
            print("❌ Test 3: Task Manager - FAILED: Task operations failed")
            
    except Exception as e:
        print(f"❌ Test 3: Task Manager - FAILED: {e}")
    
    # Test 4: Security Manager
    tests_total += 1
    try:
        from a2a_protocol.security import A2ASecurityManager
        
        security_manager = A2ASecurityManager()
        
        # Test JWT token generation
        payload = {"user_id": "test_user", "scope": "agent:read"}
        token = security_manager.generate_jwt_token(payload)
        
        # Test token validation (add small delay to avoid timing issues)
        import time
        time.sleep(0.1)
        validation_result = security_manager.validate_jwt_token(token)
        
        # Test API key generation
        api_key, api_key_obj = security_manager.generate_api_key("test_user")
        
        # Test API key validation
        api_validation = security_manager.validate_api_key(api_key)
        
        if (token and validation_result and validation_result.success and
            api_key and api_validation and api_validation.success):
            print("✅ Test 4: Security Manager - PASSED")
            tests_passed += 1
        else:
            print(f"❌ Test 4: Security Manager - FAILED: token={bool(token)}, validation={bool(validation_result)}, validation_success={validation_result.success if validation_result else None}, api_key={bool(api_key)}, api_validation={bool(api_validation)}, api_success={api_validation.success if api_validation else None}")
            
    except Exception as e:
        print(f"❌ Test 4: Security Manager - FAILED: {e}")
    
    # Test 5: Error Handling
    tests_total += 1
    try:
        from a2a_protocol.error_handling import A2AErrorHandler, A2AError, A2ATransportError
        
        error_handler = A2AErrorHandler()
        
        # Test creating A2A error
        error = A2AError(code=-32001, message="Test error message")
        
        # Test transport error
        transport_error = A2ATransportError(-32602, "Invalid params")
        
        # Test error handling
        handled_error = error_handler.handle_error(error)
        handled_transport_error = error_handler.handle_transport_error(transport_error)
        
        if (error and transport_error and 
            handled_error and handled_transport_error):
            print("✅ Test 5: Error Handling - PASSED")
            tests_passed += 1
        else:
            print("❌ Test 5: Error Handling - FAILED: Error handling operations failed")
            
    except Exception as e:
        print(f"❌ Test 5: Error Handling - FAILED: {e}")
    
    # Test 6: Short-lived Optimization
    tests_total += 1
    try:
        from a2a_protocol.short_lived_optimization import ShortLivedAgentOptimizer, OptimizationConfig
        
        optimizer = ShortLivedAgentOptimizer()
        
        # Test configuration
        config = OptimizationConfig(
            enable_memory_optimization=True,
            enable_fast_startup=True,
            enable_task_pooling=True,
            enable_resource_monitoring=True,
            enable_auto_shutdown=True
        )
        
        optimizer.configure(config)
        
        # Test creating optimized agent
        optimized_agent = optimizer.create_optimized_agent_sync(
            agent_id="test_optimized_agent",
            agent_type="general"
        )
        
        # Test task execution
        task_data = {
            "id": "test_task",
            "type": "analysis",
            "data": "test content"
        }
        
        result = optimized_agent.execute_task_sync(task_data)
        
        if optimized_agent and result:
            print("✅ Test 6: Short-lived Optimization - PASSED")
            tests_passed += 1
        else:
            print("❌ Test 6: Short-lived Optimization - FAILED: Optimization operations failed")
            
    except Exception as e:
        print(f"❌ Test 6: Short-lived Optimization - FAILED: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{tests_total} passed")
    success_rate = (tests_passed / tests_total) * 100 if tests_total > 0 else 0
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    if tests_passed == tests_total:
        print("🎉 ALL TESTS PASSED! A2A Implementation is working correctly!")
        return True
    else:
        print("💥 Some tests failed. Check the errors above.")
        return False


if __name__ == "__main__":
    success = test_a2a_components()
    sys.exit(0 if success else 1)
