#!/usr/bin/env python3
"""
Robust A2A Implementation Validation Script

A PROPER validation script that actually works and tests real functionality.
No weak bullshit - this tests the actual A2A implementation.
"""

import asyncio
import json
import logging
import sys
import time
import os
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

# Add src to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RobustA2AValidator:
    """Robust A2A Implementation Validator - No Mock Bullshit"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_results": [],
            "summary": {}
        }
        self.test_agents = []
    
    def record_test_result(self, test_name: str, success: bool, details: Dict[str, Any] = None):
        """Record a test result"""
        self.results["total_tests"] += 1
        if success:
            self.results["passed_tests"] += 1
            logger.info(f"✅ {test_name}: PASSED")
        else:
            self.results["failed_tests"] += 1
            logger.error(f"❌ {test_name}: FAILED")
        
        self.results["test_results"].append({
            "test_name": test_name,
            "success": success,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details or {}
        })
    
    def test_a2a_imports(self):
        """Test 1: A2A Component Imports"""
        test_name = "A2A Component Imports"
        try:
            # Test core imports
            from a2a_protocol.agent_card_generator import A2AAgentCardGenerator
            from a2a_protocol.transport import A2ATransport
            from a2a_protocol.task_manager import A2ATaskManager
            from a2a_protocol.security import A2ASecurityManager
            from a2a_protocol.discovery import A2AAgentDiscovery
            from a2a_protocol.error_handling import A2AErrorHandler
            from a2a_protocol.short_lived_optimization import ShortLivedAgentOptimizer
            from a2a_standard import AgentCard, AgentProvider, AgentCapabilities, AgentSkill
            
            details = {
                "imports_successful": [
                    "A2AAgentCardGenerator",
                    "A2ATransport", 
                    "A2ATaskManager",
                    "A2ASecurityManager",
                    "A2AAgentDiscovery",
                    "A2AErrorHandler",
                    "ShortLivedAgentOptimizer",
                    "AgentCard",
                    "AgentProvider",
                    "AgentCapabilities",
                    "AgentSkill"
                ]
            }
            
            self.record_test_result(test_name, True, details)
            
        except ImportError as e:
            self.record_test_result(test_name, False, {"import_error": str(e)})
        except Exception as e:
            self.record_test_result(test_name, False, {"error": str(e)})
    
    def test_a2a_standard_types(self):
        """Test 2: A2A Standard Types"""
        test_name = "A2A Standard Types"
        try:
            from a2a_standard import AgentCard, AgentProvider, AgentCapabilities, AgentSkill, TaskState, Task
            
            # Test creating agent card
            provider = AgentProvider(
                name="Test Provider",
                organization="Test Org",
                description="Test provider",
                url="https://test.com"
            )
            
            capabilities = AgentCapabilities(
                streaming=True,
                push_notifications=False,
                multi_turn=True,
                file_upload=False,
                file_download=False,
                structured_data=True
            )
            
            skill = AgentSkill(
                id="test_skill",
                name="test_skill",
                description="Test skill",
                input_modalities=["text"],
                output_modalities=["text"],
                tags=["test"],
                examples=["Test example"]
            )
            
            agent_card = AgentCard(
                name="Test Agent",
                description="Test agent card",
                version="1.0.0",
                url="https://test.com/agent",
                defaultInputModes=["text"],
                defaultOutputModes=["text"],
                provider=provider,
                capabilities=capabilities,
                skills=[skill]
            )
            
            details = {
                "agent_card_created": True,
                "agent_name": agent_card.name,
                "provider_name": agent_card.provider.name,
                "capabilities_streaming": agent_card.capabilities.streaming,
                "skills_count": len(agent_card.skills)
            }
            
            self.record_test_result(test_name, True, details)
            
        except Exception as e:
            import traceback
            self.record_test_result(test_name, False, {
                "error": str(e),
                "traceback": traceback.format_exc()
            })
    
    def test_agent_card_generator(self):
        """Test 3: Agent Card Generator"""
        test_name = "Agent Card Generator"
        try:
            from a2a_protocol.agent_card_generator import A2AAgentCardGenerator
            
            generator = A2AAgentCardGenerator(base_url="http://localhost:8000")
            
            # Test synchronous card generation
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
            
            details = {
                "generator_created": True,
                "card_generated": True,
                "card_fields": list(agent_card.keys()),
                "missing_fields": missing_fields,
                "has_security": "securitySchemes" in agent_card,
                "agent_name": agent_card.get("name"),
                "agent_url": agent_card.get("url")
            }
            
            success = len(missing_fields) == 0 and agent_card.get("name") == "Test Agent"
            self.record_test_result(test_name, success, details)
            
        except Exception as e:
            import traceback
            self.record_test_result(test_name, False, {
                "error": str(e),
                "traceback": traceback.format_exc()
            })
    
    def test_task_manager(self):
        """Test 4: Task Manager"""
        test_name = "Task Manager"
        try:
            from a2a_protocol.task_manager import A2ATaskManager
            from a2a_standard import TaskState
            
            task_manager = A2ATaskManager()
            
            # Create a task
            task_id = "test_task_123"
            task = task_manager.create_task_sync(
                task_id=task_id,
                context_id="test_context",
                message_content="Test task content"
            )
            
            # Update task status
            task_manager.update_task_status_sync(
                task_id=task_id,
                state=TaskState.WORKING,
                message="Task is working"
            )
            
            # Get task
            retrieved_task = task_manager.get_task_sync(task_id)
            
            details = {
                "task_manager_created": True,
                "task_created": task is not None,
                "task_id": task.id if task else None,
                "task_status_updated": True,
                "task_retrieved": retrieved_task is not None,
                "task_state": str(retrieved_task.status.state) if retrieved_task else None
            }
            
            success = (task is not None and 
                      retrieved_task is not None and 
                      retrieved_task.id == task_id)
            
            self.record_test_result(test_name, success, details)
            
        except Exception as e:
            import traceback
            self.record_test_result(test_name, False, {
                "error": str(e),
                "traceback": traceback.format_exc()
            })
    
    def test_security_manager(self):
        """Test 5: Security Manager"""
        test_name = "Security Manager"
        try:
            from a2a_protocol.security import A2ASecurityManager
            
            security_manager = A2ASecurityManager()
            
            # Test JWT token generation
            payload = {"user_id": "test_user", "scope": "agent:read"}
            token = security_manager.generate_jwt_token(payload)
            
            # Test token validation
            validation_result = security_manager.validate_jwt_token(token)
            
            # Test API key generation
            api_key, api_key_obj = security_manager.generate_api_key("test_user")
            
            # Test API key validation
            api_validation = security_manager.validate_api_key(api_key)
            
            details = {
                "security_manager_created": True,
                "jwt_token_generated": token is not None,
                "jwt_validation_success": validation_result.success if validation_result else False,
                "api_key_generated": api_key is not None,
                "api_key_validation_success": api_validation.success if api_validation else False,
                "token_length": len(token) if token else 0
            }
            
            success = (token is not None and 
                      validation_result and validation_result.success and
                      api_key is not None and
                      api_validation and api_validation.success)
            
            self.record_test_result(test_name, success, details)
            
        except Exception as e:
            import traceback
            self.record_test_result(test_name, False, {
                "error": str(e),
                "traceback": traceback.format_exc()
            })
    
    def test_short_lived_optimization(self):
        """Test 6: Short-lived Agent Optimization"""
        test_name = "Short-lived Agent Optimization"
        try:
            from a2a_protocol.short_lived_optimization import ShortLivedAgentOptimizer, OptimizationConfig
            
            optimizer = ShortLivedAgentOptimizer()
            
            # Test configuration
            config = OptimizationConfig(
                enable_memory_optimization=True,
                enable_startup_optimization=True,
                enable_resource_pooling=True,
                max_pool_size=5,
                pool_cleanup_interval=300
            )
            
            optimizer.configure(config)
            
            # Test creating optimized agent
            agent_id = "test_optimized_agent"
            optimized_agent = optimizer.create_optimized_agent_sync(
                agent_id=agent_id,
                agent_type="general"
            )
            
            # Test task execution
            task_data = {
                "id": "test_task",
                "type": "analysis",
                "data": "test content"
            }
            
            if optimized_agent:
                result = optimized_agent.execute_task_sync(task_data)
            else:
                result = None
            
            details = {
                "optimizer_created": True,
                "config_applied": True,
                "optimized_agent_created": optimized_agent is not None,
                "task_executed": result is not None,
                "agent_id": optimized_agent.agent_id if optimized_agent else None,
                "task_result": result if result else None
            }
            
            success = optimized_agent is not None and result is not None
            self.record_test_result(test_name, success, details)
            
        except Exception as e:
            import traceback
            self.record_test_result(test_name, False, {
                "error": str(e),
                "traceback": traceback.format_exc()
            })
    
    def test_error_handling(self):
        """Test 7: Error Handling"""
        test_name = "Error Handling"
        try:
            from a2a_protocol.error_handling import A2AErrorHandler, A2AError, A2ATransportError
            
            error_handler = A2AErrorHandler()
            
            # Test creating A2A error
            error = A2AError("Test error message", error_code="TEST_ERROR")
            
            # Test transport error
            transport_error = A2ATransportError(-32602, "Invalid params")
            
            # Test error handling
            handled_error = error_handler.handle_error(error)
            handled_transport_error = error_handler.handle_transport_error(transport_error)
            
            details = {
                "error_handler_created": True,
                "a2a_error_created": error is not None,
                "transport_error_created": transport_error is not None,
                "error_handled": handled_error is not None,
                "transport_error_handled": handled_transport_error is not None,
                "error_message": str(error),
                "transport_error_code": transport_error.code
            }
            
            success = (error is not None and 
                      transport_error is not None and
                      handled_error is not None and
                      handled_transport_error is not None)
            
            self.record_test_result(test_name, success, details)
            
        except Exception as e:
            import traceback
            self.record_test_result(test_name, False, {
                "error": str(e),
                "traceback": traceback.format_exc()
            })
    
    def run_all_tests(self):
        """Run all validation tests"""
        logger.info("🚀 Starting Robust A2A Implementation Validation")
        logger.info("=" * 80)
        
        # Run all tests
        self.test_a2a_imports()
        self.test_a2a_standard_types()
        self.test_agent_card_generator()
        self.test_task_manager()
        self.test_security_manager()
        self.test_short_lived_optimization()
        self.test_error_handling()
        
        # Generate summary
        self.results["summary"] = {
            "success_rate": (self.results["passed_tests"] / self.results["total_tests"]) * 100 if self.results["total_tests"] > 0 else 0,
            "critical_failures": [
                result for result in self.results["test_results"] 
                if not result["success"] and result["test_name"] in [
                    "A2A Component Imports", 
                    "A2A Standard Types",
                    "Agent Card Generator"
                ]
            ]
        }
        
        # Print results
        self.print_results()
        
        return self.results["failed_tests"] == 0
    
    def print_results(self):
        """Print validation results"""
        print("\n" + "="*80)
        print("🔍 ROBUST A2A IMPLEMENTATION VALIDATION RESULTS")
        print("="*80)
        
        print(f"📊 Total Tests: {self.results['total_tests']}")
        print(f"✅ Passed: {self.results['passed_tests']}")
        print(f"❌ Failed: {self.results['failed_tests']}")
        print(f"📈 Success Rate: {self.results['summary']['success_rate']:.1f}%")
        
        print("\n📋 Test Details:")
        for result in self.results["test_results"]:
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            print(f"  {status} - {result['test_name']}")
            if not result["success"] and "error" in result["details"]:
                print(f"    Error: {result['details']['error']}")
        
        if self.results["summary"]["critical_failures"]:
            print("\n🚨 Critical Failures:")
            for failure in self.results["summary"]["critical_failures"]:
                print(f"  - {failure['test_name']}")
        
        print("\n" + "="*80)
        
        # Save results to file
        with open("a2a_robust_validation_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        print("💾 Results saved to: a2a_robust_validation_results.json")


def main():
    """Main validation function"""
    validator = RobustA2AValidator()
    success = validator.run_all_tests()
    
    if success:
        print("\n🎉 A2A Implementation Validation: SUCCESS")
        sys.exit(0)
    else:
        print("\n💥 A2A Implementation Validation: FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
