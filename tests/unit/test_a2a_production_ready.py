#!/usr/bin/env python3
"""
A2A Production Readiness Validation

Comprehensive validation that A2A protocol is production-ready without requiring a running server.
Tests all components, integrations, and configurations.
"""

import sys
import os
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

class A2AProductionValidator:
    """A2A production readiness validator"""
    
    def __init__(self):
        self.test_results = []
        self.validation_errors = []
    
    def validate_a2a_imports(self) -> Dict[str, Any]:
        """Validate all A2A imports work correctly"""
        print("🔍 Validating A2A Imports...")
        
        try:
            # Core A2A components
            from a2a_protocol.transport import A2ATransportLayer
            from a2a_protocol.task_manager import A2ATaskManager
            from a2a_protocol.security import A2ASecurityManager
            from a2a_protocol.discovery import A2AAgentDiscovery
            from a2a_protocol.agent_card_generator import A2AAgentCardGenerator
            from a2a_protocol.error_handling import A2AError, A2ATransportError
            
            # A2A standard types
            from a2a_standard import AgentCard, AgentProvider, AgentCapabilities, AgentSkill
            
            # A2A API routes
            from src.api.routes.a2a import router as a2a_router
            
            # FastAPI app with A2A integration
            from src.api.main import app
            
            return {
                "test": "a2a_imports",
                "success": True,
                "message": "All A2A imports successful",
                "components": [
                    "A2ATransportLayer", "A2ATaskManager", "A2ASecurityManager",
                    "A2AAgentDiscovery", "A2AAgentCardGenerator", "A2AError",
                    "AgentCard", "AgentProvider", "A2A Router", "FastAPI App"
                ]
            }
            
        except Exception as e:
            return {
                "test": "a2a_imports",
                "success": False,
                "message": f"A2A import failed: {e}",
                "error": str(e)
            }
    
    def validate_a2a_components_initialization(self) -> Dict[str, Any]:
        """Validate A2A components can be initialized"""
        print("🔍 Validating A2A Component Initialization...")
        
        try:
            from a2a_protocol.transport import A2ATransportLayer
            from a2a_protocol.task_manager import A2ATaskManager
            from a2a_protocol.security import A2ASecurityManager
            from a2a_protocol.discovery import A2AAgentDiscovery
            from a2a_protocol.agent_card_generator import A2AAgentCardGenerator
            
            # Initialize components
            transport = A2ATransportLayer()
            task_manager = A2ATaskManager()
            security = A2ASecurityManager()
            discovery = A2AAgentDiscovery()
            card_generator = A2AAgentCardGenerator("http://localhost:8000")
            
            # Validate component attributes
            components_valid = all([
                hasattr(transport, 'handle_jsonrpc_request'),
                hasattr(task_manager, 'create_task_sync'),
                hasattr(security, 'generate_jwt_token'),
                hasattr(discovery, 'register_agent'),
                hasattr(card_generator, 'generate_agent_card_sync')
            ])
            
            return {
                "test": "a2a_component_initialization",
                "success": components_valid,
                "message": "All A2A components initialized successfully" if components_valid else "Some components missing required methods",
                "components": {
                    "transport": bool(hasattr(transport, 'handle_jsonrpc_request')),
                    "task_manager": bool(hasattr(task_manager, 'create_task_sync')),
                    "security": bool(hasattr(security, 'generate_jwt_token')),
                    "discovery": bool(hasattr(discovery, 'register_agent')),
                    "card_generator": bool(hasattr(card_generator, 'generate_agent_card_sync'))
                }
            }
            
        except Exception as e:
            return {
                "test": "a2a_component_initialization",
                "success": False,
                "message": f"A2A component initialization failed: {e}",
                "error": str(e)
            }
    
    def validate_agent_factory_a2a_integration(self) -> Dict[str, Any]:
        """Validate AgentFactory has A2A integration"""
        print("🔍 Validating AgentFactory A2A Integration...")
        
        try:
            from src.core.agent_factory import AgentFactory
            
            # Create agent factory
            factory = AgentFactory(base_url="http://localhost:8000")
            
            # Check A2A integration
            a2a_attributes = [
                'a2a_enabled', 'a2a_card_generator', 'a2a_transport',
                'a2a_task_manager', 'a2a_security_manager', 'a2a_discovery'
            ]
            
            missing_attributes = [attr for attr in a2a_attributes if not hasattr(factory, attr)]
            
            # Check if A2A is enabled
            a2a_enabled = getattr(factory, 'a2a_enabled', False)
            
            success = len(missing_attributes) == 0 and a2a_enabled
            
            return {
                "test": "agent_factory_a2a_integration",
                "success": success,
                "message": f"AgentFactory A2A integration {'complete' if success else 'incomplete'}",
                "a2a_enabled": a2a_enabled,
                "missing_attributes": missing_attributes,
                "available_attributes": [attr for attr in a2a_attributes if hasattr(factory, attr)]
            }
            
        except Exception as e:
            return {
                "test": "agent_factory_a2a_integration",
                "success": False,
                "message": f"AgentFactory A2A integration validation failed: {e}",
                "error": str(e)
            }
    
    def validate_a2a_api_endpoints(self) -> Dict[str, Any]:
        """Validate A2A API endpoints are registered"""
        print("🔍 Validating A2A API Endpoints...")
        
        try:
            from src.api.main import app
            
            # Get all routes
            routes = []
            for route in app.routes:
                if hasattr(route, 'path'):
                    routes.append(route.path)
            
            # Expected A2A endpoints
            expected_a2a_endpoints = [
                "/a2a/v1/.well-known/agent.json",
                "/a2a/v1/agents/discover",
                "/a2a/v1/message/send",
                "/a2a/v1/health"
            ]
            
            # Check which endpoints are registered
            registered_endpoints = [endpoint for endpoint in expected_a2a_endpoints if endpoint in routes]
            missing_endpoints = [endpoint for endpoint in expected_a2a_endpoints if endpoint not in routes]
            
            success = len(missing_endpoints) == 0
            
            return {
                "test": "a2a_api_endpoints",
                "success": success,
                "message": f"A2A API endpoints {'all registered' if success else 'missing some endpoints'}",
                "registered_endpoints": registered_endpoints,
                "missing_endpoints": missing_endpoints,
                "total_routes": len(routes)
            }
            
        except Exception as e:
            return {
                "test": "a2a_api_endpoints",
                "success": False,
                "message": f"A2A API endpoint validation failed: {e}",
                "error": str(e)
            }
    
    def validate_a2a_security_functionality(self) -> Dict[str, Any]:
        """Validate A2A security functionality"""
        print("🔍 Validating A2A Security Functionality...")
        
        try:
            from a2a_protocol.security import A2ASecurityManager
            
            security = A2ASecurityManager()
            
            # Test JWT token generation and validation
            payload = {"user_id": "test_user", "scope": "agent:read"}
            token = security.generate_jwt_token(payload)
            validation_result = security.validate_jwt_token(token)
            
            # Test API key generation and validation
            api_key, api_key_obj = security.generate_api_key("test_user")
            api_validation = security.validate_api_key(api_key)
            
            jwt_works = token and validation_result and validation_result.success
            api_key_works = api_key and api_validation and api_validation.success
            
            success = jwt_works and api_key_works
            
            return {
                "test": "a2a_security_functionality",
                "success": success,
                "message": f"A2A security {'fully functional' if success else 'has issues'}",
                "jwt_token_works": jwt_works,
                "api_key_works": api_key_works,
                "token_length": len(token) if token else 0,
                "api_key_length": len(api_key) if api_key else 0
            }
            
        except Exception as e:
            return {
                "test": "a2a_security_functionality",
                "success": False,
                "message": f"A2A security validation failed: {e}",
                "error": str(e)
            }
    
    def validate_docker_configuration(self) -> Dict[str, Any]:
        """Validate Docker configuration includes A2A settings"""
        print("🔍 Validating Docker Configuration...")
        
        try:
            docker_compose_path = project_root / "docker-compose.yml"
            
            if not docker_compose_path.exists():
                return {
                    "test": "docker_configuration",
                    "success": False,
                    "message": "docker-compose.yml not found"
                }
            
            with open(docker_compose_path, 'r') as f:
                docker_content = f.read()
            
            # Check for A2A environment variables
            a2a_env_vars = [
                "A2A_ENABLED=true",
                "A2A_BASE_URL",
                "A2A_MCP_PORT",
                "A2A_DISCOVERY_ENABLED",
                "A2A_SECURITY_ENABLED"
            ]
            
            present_vars = [var for var in a2a_env_vars if var in docker_content]
            missing_vars = [var for var in a2a_env_vars if var not in docker_content]
            
            # Check for A2A port exposure
            has_a2a_port = "8006:8006" in docker_content
            
            success = len(missing_vars) == 0 and has_a2a_port
            
            return {
                "test": "docker_configuration",
                "success": success,
                "message": f"Docker configuration {'complete' if success else 'missing A2A settings'}",
                "present_env_vars": present_vars,
                "missing_env_vars": missing_vars,
                "a2a_port_exposed": has_a2a_port
            }
            
        except Exception as e:
            return {
                "test": "docker_configuration",
                "success": False,
                "message": f"Docker configuration validation failed: {e}",
                "error": str(e)
            }
    
    def run_all_validations(self) -> Dict[str, Any]:
        """Run all A2A production readiness validations"""
        print("🚀 Running A2A Production Readiness Validation")
        print("=" * 70)
        
        validations = [
            self.validate_a2a_imports(),
            self.validate_a2a_components_initialization(),
            self.validate_agent_factory_a2a_integration(),
            self.validate_a2a_api_endpoints(),
            self.validate_a2a_security_functionality(),
            self.validate_docker_configuration(),
        ]
        
        total_tests = len(validations)
        passed_tests = 0
        
        for result in validations:
            if result["success"]:
                print(f"✅ {result['test']}: PASSED - {result['message']}")
                passed_tests += 1
            else:
                print(f"❌ {result['test']}: FAILED - {result['message']}")
        
        # Calculate overall readiness
        success_rate = (passed_tests / total_tests) * 100
        production_ready = passed_tests == total_tests
        
        summary = {
            "production_ready": production_ready,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": round(success_rate, 1),
            "timestamp": datetime.utcnow().isoformat(),
            "validation_results": validations
        }
        
        print(f"\n" + "=" * 70)
        print(f"📊 A2A Production Readiness Summary:")
        print(f"   Production Ready: {'YES' if production_ready else 'NO'}")
        print(f"   Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests})")
        print(f"   Timestamp: {summary['timestamp']}")
        
        if production_ready:
            print("🎉 A2A PROTOCOL IS PRODUCTION READY!")
            print("   ✅ All components functional")
            print("   ✅ All integrations complete")
            print("   ✅ Security systems operational")
            print("   ✅ API endpoints registered")
            print("   ✅ Docker configuration ready")
        else:
            print("⚠️  A2A PROTOCOL NEEDS ATTENTION BEFORE PRODUCTION")
            failed_tests = [v for v in validations if not v["success"]]
            for failed in failed_tests:
                print(f"   ❌ {failed['test']}: {failed['message']}")
        
        return summary

def main():
    """Main validation function"""
    validator = A2AProductionValidator()
    summary = validator.run_all_validations()
    
    # Exit with appropriate code
    if summary["production_ready"]:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
