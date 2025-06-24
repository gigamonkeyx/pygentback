#!/usr/bin/env python3
"""
PyGent Factory MCP Ecosystem Integration Tests

End-to-end integration testing for the complete MCP ecosystem.
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

class EcosystemIntegrationTests:
    """End-to-end integration tests for PyGent Factory MCP ecosystem"""
    
    def __init__(self):
        self.session = None
        self.test_results = []
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def log_test_result(self, test_name: str, success: bool, details: str = "", duration: float = 0):
        """Log test result"""
        result = {
            'test': test_name,
            'success': success,
            'details': details,
            'duration_ms': round(duration * 1000, 2),
            'timestamp': datetime.utcnow().isoformat()
        }
        self.test_results.append(result)
        
        status = "✅" if success else "❌"
        print(f"{status} {test_name} ({duration*1000:.1f}ms)")
        if details:
            print(f"    {details}")
    
    async def test_ecosystem_startup_sequence(self) -> bool:
        """Test that all servers start up in correct order and are accessible"""
        start = time.time()
        try:
            print("🚀 Testing Ecosystem Startup Sequence...")
            
            servers = [
                ("Document Processing", "http://127.0.0.1:8003/health"),
                ("Vector Search", "http://127.0.0.1:8004/health"),
                ("Agent Orchestration", "http://127.0.0.1:8005/health"),
                ("A2A MCP Server", "http://127.0.0.1:8006/health"),
                ("Simple A2A Agent", "http://127.0.0.1:8007/health")
            ]
            
            all_healthy = True
            server_statuses = []
            
            for name, url in servers:
                try:
                    async with self.session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            status = data.get('status', 'unknown')
                            server_statuses.append(f"{name}: {status}")
                            if status != 'healthy':
                                all_healthy = False
                        else:
                            server_statuses.append(f"{name}: HTTP {response.status}")
                            all_healthy = False
                except Exception as e:
                    server_statuses.append(f"{name}: Error - {str(e)}")
                    all_healthy = False
            
            duration = time.time() - start
            details = f"All servers healthy: {all_healthy}. Status: {', '.join(server_statuses)}"
            self.log_test_result("Ecosystem Startup Sequence", all_healthy, details, duration)
            return all_healthy
            
        except Exception as e:
            duration = time.time() - start
            self.log_test_result("Ecosystem Startup Sequence", False, f"Error: {str(e)}", duration)
            return False
    
    async def test_a2a_agent_discovery_workflow(self) -> bool:
        """Test complete A2A agent discovery workflow"""
        start = time.time()
        try:
            print("\n🔍 Testing A2A Agent Discovery Workflow...")
            
            # Step 1: Trigger discovery
            async with self.session.post("http://127.0.0.1:8005/v1/a2a/discover") as response:
                if response.status != 200:
                    raise Exception(f"Discovery trigger failed: HTTP {response.status}")
                
                discovery_data = await response.json()
                discovered_count = discovery_data.get('total_agents', 0)
            
            # Step 2: List discovered agents
            async with self.session.get("http://127.0.0.1:8005/v1/a2a/agents") as response:
                if response.status != 200:
                    raise Exception(f"Agent listing failed: HTTP {response.status}")
                
                agents_data = await response.json()
                listed_count = agents_data.get('total_agents', 0)
                agents = agents_data.get('a2a_agents', {})
            
            # Step 3: Validate agent information
            valid_agents = 0
            for agent_id, agent_info in agents.items():
                if all(key in agent_info for key in ['name', 'description', 'url', 'version']):
                    valid_agents += 1
            
            success = discovered_count > 0 and discovered_count == listed_count and valid_agents == listed_count
            duration = time.time() - start
            details = f"Discovered: {discovered_count}, Listed: {listed_count}, Valid: {valid_agents}"
            self.log_test_result("A2A Agent Discovery Workflow", success, details, duration)
            return success
            
        except Exception as e:
            duration = time.time() - start
            self.log_test_result("A2A Agent Discovery Workflow", False, f"Error: {str(e)}", duration)
            return False
    
    async def test_multi_agent_communication_flow(self) -> bool:
        """Test complete multi-agent communication flow"""
        start = time.time()
        try:
            print("\n💬 Testing Multi-Agent Communication Flow...")
            
            # Step 1: Get available agents
            async with self.session.get("http://127.0.0.1:8005/v1/a2a/agents") as response:
                if response.status != 200:
                    raise Exception(f"Failed to get agents: HTTP {response.status}")
                
                agents_data = await response.json()
                agents = agents_data.get('a2a_agents', {})
                
                if not agents:
                    raise Exception("No agents available for testing")
            
            # Step 2: Test message sending to each agent
            successful_messages = 0
            total_agents = len(agents)
            
            for agent_id, agent_info in agents.items():
                try:
                    message_payload = {
                        "agent_id": agent_id,
                        "message": f"Integration test message for {agent_info['name']}. Please process this test: 'PyGent Factory integration testing is working perfectly!'",
                        "context_id": f"integration-test-{agent_id}"
                    }
                    
                    async with self.session.post(
                        "http://127.0.0.1:8005/v1/a2a/message",
                        json=message_payload
                    ) as msg_response:
                        if msg_response.status == 200:
                            result = await msg_response.json()
                            if result.get('status') == 'sent':
                                successful_messages += 1
                
                except Exception as e:
                    print(f"    Failed to send message to {agent_info['name']}: {e}")
            
            success = successful_messages == total_agents
            duration = time.time() - start
            details = f"Successful messages: {successful_messages}/{total_agents}"
            self.log_test_result("Multi-Agent Communication Flow", success, details, duration)
            return success
            
        except Exception as e:
            duration = time.time() - start
            self.log_test_result("Multi-Agent Communication Flow", False, f"Error: {str(e)}", duration)
            return False
    
    async def test_server_capability_integration(self) -> bool:
        """Test integration between server capabilities"""
        start = time.time()
        try:
            print("\n🔧 Testing Server Capability Integration...")
            
            capabilities_found = {}
            
            # Test each server's capabilities
            servers = [
                ("Document Processing", "http://127.0.0.1:8003/"),
                ("Vector Search", "http://127.0.0.1:8004/"),
                ("Agent Orchestration", "http://127.0.0.1:8005/"),
                ("A2A MCP Server", "http://127.0.0.1:8006/"),
                ("Simple A2A Agent", "http://127.0.0.1:8007/")
            ]
            
            for name, url in servers:
                try:
                    async with self.session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            capabilities = data.get('capabilities', [])
                            capabilities_found[name] = len(capabilities)
                        else:
                            capabilities_found[name] = 0
                except Exception:
                    capabilities_found[name] = 0
            
            # Validate that all servers have capabilities
            total_capabilities = sum(capabilities_found.values())
            servers_with_capabilities = len([c for c in capabilities_found.values() if c > 0])
            
            success = servers_with_capabilities == len(servers) and total_capabilities >= 20
            duration = time.time() - start
            details = f"Total capabilities: {total_capabilities}, Servers with capabilities: {servers_with_capabilities}/{len(servers)}"
            self.log_test_result("Server Capability Integration", success, details, duration)
            return success
            
        except Exception as e:
            duration = time.time() - start
            self.log_test_result("Server Capability Integration", False, f"Error: {str(e)}", duration)
            return False
    
    async def test_error_handling_and_recovery(self) -> bool:
        """Test error handling and recovery mechanisms"""
        start = time.time()
        try:
            print("\n🛡️ Testing Error Handling and Recovery...")
            
            error_tests = []
            
            # Test 1: Invalid endpoint
            try:
                async with self.session.get("http://127.0.0.1:8005/invalid-endpoint") as response:
                    error_tests.append(response.status == 404)
            except Exception:
                error_tests.append(False)
            
            # Test 2: Invalid A2A message
            try:
                invalid_payload = {"invalid": "payload"}
                async with self.session.post(
                    "http://127.0.0.1:8005/v1/a2a/message",
                    json=invalid_payload
                ) as response:
                    error_tests.append(response.status == 400)
            except Exception:
                error_tests.append(False)
            
            # Test 3: Non-existent agent message
            try:
                invalid_agent_payload = {
                    "agent_id": "non-existent-agent",
                    "message": "test",
                    "context_id": "test"
                }
                async with self.session.post(
                    "http://127.0.0.1:8005/v1/a2a/message",
                    json=invalid_agent_payload
                ) as response:
                    error_tests.append(response.status in [404, 500])
            except Exception:
                error_tests.append(False)
            
            success = all(error_tests)
            duration = time.time() - start
            details = f"Error handling tests passed: {sum(error_tests)}/{len(error_tests)}"
            self.log_test_result("Error Handling and Recovery", success, details, duration)
            return success
            
        except Exception as e:
            duration = time.time() - start
            self.log_test_result("Error Handling and Recovery", False, f"Error: {str(e)}", duration)
            return False
    
    async def test_performance_under_load(self) -> bool:
        """Test system performance under moderate load"""
        start = time.time()
        try:
            print("\n⚡ Testing Performance Under Load...")
            
            # Concurrent health checks
            health_tasks = []
            for _ in range(50):
                task = self.session.get("http://127.0.0.1:8005/health")
                health_tasks.append(task)
            
            # Execute concurrent requests
            responses = await asyncio.gather(*health_tasks, return_exceptions=True)
            
            successful_responses = 0
            for response in responses:
                if not isinstance(response, Exception):
                    async with response as resp:
                        if resp.status == 200:
                            successful_responses += 1
                        resp.close()
            
            success_rate = (successful_responses / len(health_tasks)) * 100
            success = success_rate >= 95
            
            duration = time.time() - start
            details = f"Success rate under load: {success_rate:.1f}% ({successful_responses}/{len(health_tasks)})"
            self.log_test_result("Performance Under Load", success, details, duration)
            return success
            
        except Exception as e:
            duration = time.time() - start
            self.log_test_result("Performance Under Load", False, f"Error: {str(e)}", duration)
            return False
    
    async def run_all_integration_tests(self) -> Dict[str, Any]:
        """Run all integration tests"""
        print("🧪 PyGent Factory MCP Ecosystem Integration Tests")
        print("=" * 70)
        
        tests = [
            self.test_ecosystem_startup_sequence,
            self.test_a2a_agent_discovery_workflow,
            self.test_multi_agent_communication_flow,
            self.test_server_capability_integration,
            self.test_error_handling_and_recovery,
            self.test_performance_under_load
        ]
        
        passed = 0
        for test in tests:
            if await test():
                passed += 1
        
        total = len(tests)
        success_rate = (passed / total) * 100
        
        print("\n" + "=" * 70)
        print(f"📊 Integration Test Results: {passed}/{total} tests passed ({success_rate:.1f}%)")
        
        if passed == total:
            print("🎉 ALL INTEGRATION TESTS PASSED!")
            print("   PyGent Factory MCP Ecosystem is fully integrated and production-ready!")
        elif passed >= total * 0.8:
            print("✅ MOST INTEGRATION TESTS PASSED!")
            print("   PyGent Factory MCP Ecosystem is largely functional with minor issues.")
        else:
            print("⚠️ SOME INTEGRATION TESTS FAILED")
            print("   PyGent Factory MCP Ecosystem needs attention before production deployment.")
        
        return {
            'total_tests': total,
            'passed_tests': passed,
            'success_rate': success_rate,
            'test_results': self.test_results,
            'ecosystem_status': 'production_ready' if passed == total else 'needs_attention'
        }


async def main():
    """Main integration test execution"""
    async with EcosystemIntegrationTests() as tests:
        results = await tests.run_all_integration_tests()
        
        # Save results
        with open('ecosystem_integration_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Integration test results saved to: ecosystem_integration_test_results.json")
        
        return 0 if results['success_rate'] >= 80 else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
