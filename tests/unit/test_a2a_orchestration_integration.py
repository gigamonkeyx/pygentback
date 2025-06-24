#!/usr/bin/env python3
"""
Test A2A Orchestration Integration

Test the integration between the Agent Orchestration MCP Server and A2A MCP Server.
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

class A2AOrchestrationTester:
    """Test A2A integration with Agent Orchestration"""
    
    def __init__(self):
        self.orchestration_url = "http://127.0.0.1:8005"
        self.a2a_url = "http://127.0.0.1:8006"
        self.session = None
        self.results = []
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def log_result(self, test_name: str, success: bool, details: str = "", duration: float = 0):
        """Log test result"""
        result = {
            'test': test_name,
            'success': success,
            'details': details,
            'duration_ms': round(duration * 1000, 2),
            'timestamp': datetime.utcnow().isoformat()
        }
        self.results.append(result)
        
        status = "✅" if success else "❌"
        print(f"{status} {test_name} ({duration*1000:.1f}ms)")
        if details:
            print(f"    {details}")
    
    async def test_orchestration_health(self) -> bool:
        """Test orchestration server health"""
        start = time.time()
        try:
            print("🏥 Testing Agent Orchestration Server Health...")
            
            async with self.session.get(f"{self.orchestration_url}/health") as response:
                duration = time.time() - start
                
                if response.status == 200:
                    data = await response.json()
                    status = data.get('status', 'unknown')
                    
                    if status == 'healthy':
                        a2a_agents = data.get('performance', {}).get('a2a_agents_discovered', 0)
                        details = f"Orchestration healthy, A2A agents: {a2a_agents}"
                        self.log_result("Orchestration Health", True, details, duration)
                        return True
                    else:
                        details = f"Orchestration status: {status}"
                        self.log_result("Orchestration Health", False, details, duration)
                        return False
                else:
                    details = f"HTTP {response.status}"
                    self.log_result("Orchestration Health", False, details, duration)
                    return False
                    
        except Exception as e:
            duration = time.time() - start
            self.log_result("Orchestration Health", False, f"Error: {str(e)}", duration)
            return False
    
    async def test_a2a_server_health(self) -> bool:
        """Test A2A server health"""
        start = time.time()
        try:
            print("\n🔗 Testing A2A Server Health...")
            
            async with self.session.get(f"{self.a2a_url}/health") as response:
                duration = time.time() - start
                
                if response.status == 200:
                    data = await response.json()
                    status = data.get('status', 'unknown')
                    
                    if status == 'healthy':
                        uptime = data.get('details', {}).get('uptime_seconds', 0)
                        details = f"A2A server healthy, uptime: {uptime:.1f}s"
                        self.log_result("A2A Server Health", True, details, duration)
                        return True
                    else:
                        details = f"A2A status: {status}"
                        self.log_result("A2A Server Health", False, details, duration)
                        return False
                else:
                    details = f"HTTP {response.status}"
                    self.log_result("A2A Server Health", False, details, duration)
                    return False
                    
        except Exception as e:
            duration = time.time() - start
            self.log_result("A2A Server Health", False, f"Error: {str(e)}", duration)
            return False
    
    async def test_a2a_agent_discovery(self) -> bool:
        """Test A2A agent discovery from orchestration server"""
        start = time.time()
        try:
            print("\n🔍 Testing A2A Agent Discovery...")
            
            async with self.session.get(f"{self.orchestration_url}/v1/a2a/agents") as response:
                duration = time.time() - start
                
                if response.status == 200:
                    data = await response.json()
                    total_agents = data.get('total_agents', 0)
                    a2a_agents = data.get('a2a_agents', {})
                    
                    details = f"Discovered {total_agents} A2A agents"
                    if a2a_agents:
                        agent_names = [agent['name'] for agent in a2a_agents.values()]
                        details += f": {', '.join(agent_names)}"
                    
                    self.log_result("A2A Agent Discovery", True, details, duration)
                    return True
                else:
                    details = f"HTTP {response.status}"
                    self.log_result("A2A Agent Discovery", False, details, duration)
                    return False
                    
        except Exception as e:
            duration = time.time() - start
            self.log_result("A2A Agent Discovery", False, f"Error: {str(e)}", duration)
            return False
    
    async def test_a2a_message_sending(self) -> bool:
        """Test sending A2A message from orchestration server"""
        start = time.time()
        try:
            print("\n💬 Testing A2A Message Sending...")
            
            # First, check if we have any A2A agents
            async with self.session.get(f"{self.orchestration_url}/v1/a2a/agents") as response:
                if response.status != 200:
                    self.log_result("A2A Message Sending", False, "No A2A agents available", time.time() - start)
                    return False
                
                data = await response.json()
                a2a_agents = data.get('a2a_agents', {})
                
                if not a2a_agents:
                    # Try to discover agents first
                    async with self.session.post(f"{self.orchestration_url}/v1/a2a/discover") as discover_response:
                        if discover_response.status == 200:
                            # Check again
                            async with self.session.get(f"{self.orchestration_url}/v1/a2a/agents") as retry_response:
                                if retry_response.status == 200:
                                    retry_data = await retry_response.json()
                                    a2a_agents = retry_data.get('a2a_agents', {})
                
                if not a2a_agents:
                    self.log_result("A2A Message Sending", False, "No A2A agents available for testing", time.time() - start)
                    return False
            
            # Get the first available agent
            agent_id = list(a2a_agents.keys())[0]
            agent_name = a2a_agents[agent_id]['name']
            
            # Send a test message
            message_payload = {
                "agent_id": agent_id,
                "message": "Hello from Agent Orchestration! This is a test message.",
                "context_id": "test-context-001"
            }
            
            async with self.session.post(
                f"{self.orchestration_url}/v1/a2a/message",
                json=message_payload
            ) as response:
                duration = time.time() - start
                
                if response.status == 200:
                    result = await response.json()
                    details = f"Message sent to {agent_name}, response received"
                    self.log_result("A2A Message Sending", True, details, duration)
                    return True
                else:
                    error_text = await response.text()
                    details = f"HTTP {response.status}: {error_text}"
                    self.log_result("A2A Message Sending", False, details, duration)
                    return False
                    
        except Exception as e:
            duration = time.time() - start
            self.log_result("A2A Message Sending", False, f"Error: {str(e)}", duration)
            return False
    
    async def test_orchestration_capabilities(self) -> bool:
        """Test orchestration server capabilities"""
        start = time.time()
        try:
            print("\n📊 Testing Orchestration Capabilities...")
            
            async with self.session.get(f"{self.orchestration_url}/") as response:
                duration = time.time() - start
                
                if response.status == 200:
                    data = await response.json()
                    capabilities = data.get('capabilities', [])
                    
                    # Check for A2A integration capability
                    has_a2a = "A2A protocol integration" in capabilities
                    
                    if has_a2a:
                        details = f"A2A integration capability present, {len(capabilities)} total capabilities"
                        self.log_result("Orchestration Capabilities", True, details, duration)
                        return True
                    else:
                        details = f"A2A integration capability missing from {capabilities}"
                        self.log_result("Orchestration Capabilities", False, details, duration)
                        return False
                else:
                    details = f"HTTP {response.status}"
                    self.log_result("Orchestration Capabilities", False, details, duration)
                    return False
                    
        except Exception as e:
            duration = time.time() - start
            self.log_result("Orchestration Capabilities", False, f"Error: {str(e)}", duration)
            return False
    
    async def run_all_tests(self) -> dict:
        """Run all A2A orchestration integration tests"""
        print("🚀 A2A Orchestration Integration Test Suite")
        print("=" * 60)
        
        tests = [
            ("Orchestration Health", self.test_orchestration_health),
            ("A2A Server Health", self.test_a2a_server_health),
            ("A2A Agent Discovery", self.test_a2a_agent_discovery),
            ("A2A Message Sending", self.test_a2a_message_sending),
            ("Orchestration Capabilities", self.test_orchestration_capabilities)
        ]
        
        passed = 0
        for test_name, test_func in tests:
            if await test_func():
                passed += 1
        
        total = len(tests)
        
        print("\n" + "=" * 60)
        print(f"📊 A2A Integration Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print(f"\n🎉 ALL TESTS PASSED!")
            print(f"   A2A integration with Agent Orchestration is fully functional.")
        elif passed >= total * 0.75:
            print(f"\n✅ MOST TESTS PASSED!")
            print(f"   A2A integration is largely functional with minor issues.")
        else:
            print(f"\n⚠️ SOME TESTS FAILED")
            print(f"   A2A integration needs attention before production use.")
        
        return {
            'total_tests': total,
            'passed_tests': passed,
            'success_rate': (passed / total) * 100,
            'test_results': self.results
        }


async def main():
    """Main test execution"""
    async with A2AOrchestrationTester() as tester:
        results = await tester.run_all_tests()
        
        # Save results
        with open('a2a_orchestration_integration_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Results saved to: a2a_orchestration_integration_results.json")
        
        return 0 if results['success_rate'] >= 75 else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
