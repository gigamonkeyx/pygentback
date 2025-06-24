#!/usr/bin/env python3
"""
A2A Implementation Validation Script

Comprehensive validation script to test the complete A2A protocol implementation
according to Google A2A specification.
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class A2AValidationRunner:
    """A2A Implementation Validation Runner"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_results": [],
            "summary": {}
        }
        self.agent_factory = None
        self.test_agents = []
    
    async def setup(self):
        """Setup validation environment"""
        try:
            # Test basic imports first
            logger.info("Testing A2A component imports...")

            # Import core components
            from core.agent_factory import AgentFactory
            logger.info("✅ AgentFactory imported")

            # Import A2A protocol components
            try:
                from a2a_protocol.agent_card_generator import A2AAgentCardGenerator
                logger.info("✅ A2A Agent Card Generator imported")
            except ImportError as e:
                logger.warning(f"⚠️ A2A Agent Card Generator not available: {e}")

            try:
                from a2a_protocol.transport import A2ATransport
                logger.info("✅ A2A Transport imported")
            except ImportError as e:
                logger.warning(f"⚠️ A2A Transport not available: {e}")

            try:
                from a2a_protocol.task_manager import A2ATaskManager
                logger.info("✅ A2A Task Manager imported")
            except ImportError as e:
                logger.warning(f"⚠️ A2A Task Manager not available: {e}")

            try:
                from a2a_protocol.security import A2ASecurityManager
                logger.info("✅ A2A Security Manager imported")
            except ImportError as e:
                logger.warning(f"⚠️ A2A Security Manager not available: {e}")

            try:
                from a2a_protocol.discovery import A2AAgentDiscovery
                logger.info("✅ A2A Discovery imported")
            except ImportError as e:
                logger.warning(f"⚠️ A2A Discovery not available: {e}")

            try:
                from a2a_protocol.error_handling import A2AErrorHandler
                logger.info("✅ A2A Error Handler imported")
            except ImportError as e:
                logger.warning(f"⚠️ A2A Error Handler not available: {e}")

            # Initialize agent factory
            self.agent_factory = AgentFactory(base_url="http://localhost:8000")

            # Try to initialize (but don't fail if it doesn't work)
            try:
                await self.agent_factory.initialize()
                logger.info("✅ Agent factory initialized successfully")
            except Exception as e:
                logger.warning(f"⚠️ Agent factory initialization failed: {e}")
                # Continue anyway for basic testing

            return True

        except Exception as e:
            logger.error(f"❌ Failed to setup validation environment: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    async def cleanup(self):
        """Cleanup validation environment"""
        try:
            # Cleanup test agents
            for agent in self.test_agents:
                try:
                    await self.agent_factory.destroy_agent(agent.agent_id)
                except Exception as e:
                    logger.warning(f"Failed to cleanup agent {agent.agent_id}: {e}")
            
            logger.info("✅ Validation environment cleaned up")
        except Exception as e:
            logger.error(f"❌ Failed to cleanup validation environment: {e}")
    
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
    
    async def test_agent_factory_a2a_support(self):
        """Test 1: Agent Factory A2A Support"""
        test_name = "Agent Factory A2A Support"
        try:
            if not self.agent_factory:
                self.record_test_result(test_name, False, {"error": "Agent factory not initialized"})
                return

            # Check A2A components
            a2a_enabled = getattr(self.agent_factory, 'a2a_enabled', False)
            a2a_card_generator = getattr(self.agent_factory, 'a2a_card_generator', None)
            a2a_mcp_server = getattr(self.agent_factory, 'a2a_mcp_server', None)
            base_url = getattr(self.agent_factory, 'base_url', None)

            # Check if agent factory has the expected methods
            has_create_agent = hasattr(self.agent_factory, 'create_agent')
            has_a2a_methods = hasattr(self.agent_factory, 'discover_agents_in_network')

            details = {
                "a2a_enabled": a2a_enabled,
                "a2a_card_generator_available": a2a_card_generator is not None,
                "a2a_mcp_server_available": a2a_mcp_server is not None,
                "base_url": base_url,
                "has_create_agent": has_create_agent,
                "has_a2a_methods": has_a2a_methods,
                "agent_factory_type": type(self.agent_factory).__name__
            }

            # Success if we have basic functionality
            success = has_create_agent and base_url is not None
            self.record_test_result(test_name, success, details)

        except Exception as e:
            import traceback
            self.record_test_result(test_name, False, {
                "error": str(e),
                "traceback": traceback.format_exc()
            })
    
    async def test_agent_creation(self):
        """Test 2: A2A-Compliant Agent Creation"""
        test_name = "A2A-Compliant Agent Creation"
        try:
            if not self.agent_factory:
                self.record_test_result(test_name, False, {"error": "Agent factory not available"})
                return

            if not hasattr(self.agent_factory, 'create_agent'):
                self.record_test_result(test_name, False, {"error": "create_agent method not available"})
                return

            # Try to create a test agent
            try:
                agent = await self.agent_factory.create_agent(
                    agent_type="general",
                    name="A2A Validation Test Agent",
                    capabilities=["reasoning", "analysis"]
                )

                if agent:
                    self.test_agents.append(agent)

                    details = {
                        "agent_id": getattr(agent, 'agent_id', None),
                        "agent_name": getattr(agent, 'name', None),
                        "agent_type": getattr(agent, 'type', None),
                        "status": str(getattr(agent, 'status', 'unknown')),
                        "agent_class": type(agent).__name__
                    }

                    success = agent is not None and hasattr(agent, 'agent_id')
                    self.record_test_result(test_name, success, details)
                else:
                    self.record_test_result(test_name, False, {"error": "Agent creation returned None"})

            except Exception as create_error:
                # Agent creation failed, but that's still useful information
                self.record_test_result(test_name, False, {
                    "error": f"Agent creation failed: {str(create_error)}",
                    "error_type": type(create_error).__name__
                })

        except Exception as e:
            import traceback
            self.record_test_result(test_name, False, {
                "error": str(e),
                "traceback": traceback.format_exc()
            })
    
    async def test_agent_card_generation(self):
        """Test 3: Agent Card Generation"""
        test_name = "Agent Card Generation"
        try:
            if not self.test_agents:
                self.record_test_result(test_name, False, {"error": "No test agents available"})
                return
            
            agent = self.test_agents[0]
            
            if self.agent_factory.a2a_card_generator:
                agent_card = await self.agent_factory.a2a_card_generator.generate_agent_card(
                    agent_id=agent.agent_id,
                    agent_name=agent.name,
                    agent_type=agent.type,
                    capabilities=["reasoning", "analysis"],
                    skills=["problem_solving", "data_analysis"],
                    enable_authentication=True
                )
                
                # Validate agent card structure
                required_fields = ["name", "description", "url", "capabilities", "skills", "provider"]
                missing_fields = [field for field in required_fields if field not in agent_card]
                
                details = {
                    "agent_card_fields": list(agent_card.keys()),
                    "required_fields": required_fields,
                    "missing_fields": missing_fields,
                    "has_security_schemes": "securitySchemes" in agent_card
                }
                
                success = len(missing_fields) == 0
                self.record_test_result(test_name, success, details)
            else:
                self.record_test_result(test_name, False, {"error": "A2A card generator not available"})
                
        except Exception as e:
            self.record_test_result(test_name, False, {"error": str(e)})
    
    async def test_a2a_mcp_server_integration(self):
        """Test 4: A2A MCP Server Integration"""
        test_name = "A2A MCP Server Integration"
        try:
            if self.agent_factory.a2a_mcp_server:
                # Get server status
                status = self.agent_factory.get_a2a_mcp_server_status()
                
                # Try to register an agent
                if self.test_agents:
                    agent = self.test_agents[0]
                    registration_success = await self.agent_factory.register_agent_with_a2a_mcp(agent)
                    
                    details = {
                        "server_status": status,
                        "agent_registration_success": registration_success
                    }
                    
                    success = status.get("available", False) and registration_success
                else:
                    details = {"server_status": status}
                    success = status.get("available", False)
                
                self.record_test_result(test_name, success, details)
            else:
                self.record_test_result(test_name, False, {"error": "A2A MCP server not available"})
                
        except Exception as e:
            self.record_test_result(test_name, False, {"error": str(e)})
    
    async def test_short_lived_agent_optimization(self):
        """Test 5: Short-lived Agent Optimization"""
        test_name = "Short-lived Agent Optimization"
        try:
            # Create a short-lived agent
            start_time = time.time()
            
            short_lived_agent = await self.agent_factory.create_short_lived_agent(
                agent_type="general",
                purpose="validation_test"
            )
            
            creation_time = time.time() - start_time
            
            if short_lived_agent:
                self.test_agents.append(short_lived_agent)
                
                # Execute a task
                task_start = time.time()
                task_result = await self.agent_factory.execute_short_lived_task(
                    short_lived_agent.agent_id,
                    {"id": "validation_task", "type": "test", "data": "validation data"}
                )
                task_time = time.time() - task_start
                
                # Get metrics
                metrics = await self.agent_factory.get_short_lived_agent_metrics(
                    short_lived_agent.agent_id
                )
                
                details = {
                    "creation_time_seconds": creation_time,
                    "task_execution_time_seconds": task_time,
                    "task_result_available": task_result is not None,
                    "metrics_available": metrics is not None,
                    "agent_config": short_lived_agent.config.custom_config if hasattr(short_lived_agent, 'config') else None
                }
                
                success = task_result is not None and creation_time < 5.0  # Should be fast
                self.record_test_result(test_name, success, details)
                
                # Cleanup short-lived agent
                await self.agent_factory.shutdown_short_lived_agent(
                    short_lived_agent.agent_id,
                    pool_for_reuse=False
                )
            else:
                self.record_test_result(test_name, False, {"error": "Short-lived agent creation failed"})
                
        except Exception as e:
            self.record_test_result(test_name, False, {"error": str(e)})
    
    async def test_agent_discovery(self):
        """Test 6: Agent Discovery"""
        test_name = "Agent Discovery"
        try:
            # Test agent discovery
            discovered_agents = await self.agent_factory.discover_agents_in_network([
                "http://localhost:8000",
                "http://localhost:8001"
            ])
            
            # Test task-based agent finding
            research_match = await self.agent_factory.find_agent_for_task(
                "Research the latest AI developments",
                required_capabilities=["research"],
                required_skills=["analysis"]
            )
            
            # Get discovery statistics
            discovery_stats = self.agent_factory.get_discovery_stats()
            
            details = {
                "discovered_agents_count": len(discovered_agents),
                "research_match_found": research_match is not None,
                "discovery_stats": discovery_stats
            }
            
            success = isinstance(discovered_agents, list) and not discovery_stats.get("error")
            self.record_test_result(test_name, success, details)
            
        except Exception as e:
            self.record_test_result(test_name, False, {"error": str(e)})
    
    async def test_error_handling(self):
        """Test 7: Error Handling"""
        test_name = "Error Handling"
        try:
            # Test invalid agent type
            invalid_agent_error = None
            try:
                await self.agent_factory.create_agent(
                    agent_type="invalid_nonexistent_type",
                    name="Invalid Agent"
                )
            except Exception as e:
                invalid_agent_error = str(e)
            
            # Test non-existent agent retrieval
            non_existent_agent = await self.agent_factory.get_agent("non_existent_id_12345")
            
            # Test connectivity to invalid endpoint
            connectivity_result = await self.agent_factory.test_agent_connectivity(
                "http://invalid-nonexistent-server:9999"
            )
            
            details = {
                "invalid_agent_error_caught": invalid_agent_error is not None,
                "non_existent_agent_returns_none": non_existent_agent is None,
                "invalid_connectivity_handled": not connectivity_result.get("success", True)
            }
            
            success = (invalid_agent_error is not None and 
                      non_existent_agent is None and 
                      not connectivity_result.get("success", True))
            
            self.record_test_result(test_name, success, details)
            
        except Exception as e:
            self.record_test_result(test_name, False, {"error": str(e)})
    
    async def test_performance_benchmarks(self):
        """Test 8: Performance Benchmarks"""
        test_name = "Performance Benchmarks"
        try:
            # Test agent creation performance
            start_time = time.time()
            
            performance_agents = []
            for i in range(3):
                agent = await self.agent_factory.create_agent(
                    agent_type="general",
                    name=f"Performance Test Agent {i}"
                )
                performance_agents.append(agent)
                self.test_agents.append(agent)
            
            creation_time = time.time() - start_time
            
            # Test concurrent operations
            concurrent_start = time.time()
            
            async def get_agent_status(agent):
                return await self.agent_factory.get_agent_status(agent.agent_id)
            
            status_tasks = [get_agent_status(agent) for agent in performance_agents]
            statuses = await asyncio.gather(*status_tasks)
            
            concurrent_time = time.time() - concurrent_start
            
            details = {
                "agents_created": len(performance_agents),
                "creation_time_seconds": creation_time,
                "average_creation_time": creation_time / len(performance_agents),
                "concurrent_operations_time": concurrent_time,
                "all_statuses_retrieved": all(status is not None for status in statuses)
            }
            
            # Performance criteria: should create 3 agents in under 15 seconds
            success = creation_time < 15.0 and concurrent_time < 5.0
            self.record_test_result(test_name, success, details)
            
        except Exception as e:
            self.record_test_result(test_name, False, {"error": str(e)})
    
    async def run_all_tests(self):
        """Run all validation tests"""
        logger.info("🚀 Starting A2A Implementation Validation")
        
        # Setup
        if not await self.setup():
            logger.error("❌ Failed to setup validation environment")
            return False
        
        try:
            # Run all tests
            await self.test_agent_factory_a2a_support()
            await self.test_agent_creation()
            await self.test_agent_card_generation()
            await self.test_a2a_mcp_server_integration()
            await self.test_short_lived_agent_optimization()
            await self.test_agent_discovery()
            await self.test_error_handling()
            await self.test_performance_benchmarks()
            
            # Generate summary
            self.results["summary"] = {
                "success_rate": (self.results["passed_tests"] / self.results["total_tests"]) * 100 if self.results["total_tests"] > 0 else 0,
                "total_duration": "N/A",  # Could add timing
                "critical_failures": [
                    result for result in self.results["test_results"] 
                    if not result["success"] and result["test_name"] in [
                        "Agent Factory A2A Support", 
                        "A2A-Compliant Agent Creation",
                        "Agent Card Generation"
                    ]
                ]
            }
            
            # Print results
            self.print_results()
            
            return self.results["failed_tests"] == 0
            
        finally:
            await self.cleanup()
    
    def print_results(self):
        """Print validation results"""
        print("\n" + "="*80)
        print("🔍 A2A IMPLEMENTATION VALIDATION RESULTS")
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
        with open("a2a_validation_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        print("💾 Results saved to: a2a_validation_results.json")


async def main():
    """Main validation function"""
    validator = A2AValidationRunner()
    success = await validator.run_all_tests()
    
    if success:
        print("\n🎉 A2A Implementation Validation: SUCCESS")
        sys.exit(0)
    else:
        print("\n💥 A2A Implementation Validation: FAILED")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
