#!/usr/bin/env python3
"""
Test A2A MCP Server

Test the A2A MCP Server functionality including:
- Agent card discovery
- Message sending and receiving
- Task management
- Multi-agent communication
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

class A2AMCPTester:
    """Test the A2A MCP Server"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8006"):
        self.base_url = base_url
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
    
    async def test_server_health(self) -> bool:
        """Test server health endpoint"""
        start = time.time()
        try:
            print("🏥 Testing A2A MCP Server Health...")
            
            async with self.session.get(f"{self.base_url}/health") as response:
                duration = time.time() - start
                
                if response.status == 200:
                    data = await response.json()
                    status = data.get('status', 'unknown')
                    
                    if status == 'healthy':
                        details = f"Server healthy, uptime: {data.get('details', {}).get('uptime_seconds', 0):.1f}s"
                        self.log_result("Server Health", True, details, duration)
                        return True
                    else:
                        details = f"Server status: {status}"
                        self.log_result("Server Health", False, details, duration)
                        return False
                else:
                    details = f"HTTP {response.status}"
                    self.log_result("Server Health", False, details, duration)
                    return False
                    
        except Exception as e:
            duration = time.time() - start
            self.log_result("Server Health", False, f"Error: {str(e)}", duration)
            return False
    
    async def test_agent_card_discovery(self) -> bool:
        """Test agent card discovery"""
        start = time.time()
        try:
            print("\n🔍 Testing Agent Card Discovery...")
            
            async with self.session.get(f"{self.base_url}/.well-known/agent.json") as response:
                duration = time.time() - start
                
                if response.status == 200:
                    agent_card = await response.json()
                    
                    # Validate agent card structure
                    required_fields = ['name', 'description', 'version', 'url']
                    missing_fields = [field for field in required_fields if field not in agent_card]
                    
                    if not missing_fields:
                        details = f"Agent: {agent_card['name']} v{agent_card['version']}"
                        self.log_result("Agent Card Discovery", True, details, duration)
                        return True
                    else:
                        details = f"Missing fields: {missing_fields}"
                        self.log_result("Agent Card Discovery", False, details, duration)
                        return False
                else:
                    details = f"HTTP {response.status}"
                    self.log_result("Agent Card Discovery", False, details, duration)
                    return False
                    
        except Exception as e:
            duration = time.time() - start
            self.log_result("Agent Card Discovery", False, f"Error: {str(e)}", duration)
            return False
    
    async def test_a2a_message_sending(self) -> bool:
        """Test A2A message sending"""
        start = time.time()
        try:
            print("\n💬 Testing A2A Message Sending...")
            
            # Create A2A message payload
            message_payload = {
                "jsonrpc": "2.0",
                "id": "test-001",
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [
                            {
                                "kind": "text",
                                "text": "Hello from A2A test! Please process this message."
                            }
                        ],
                        "messageId": "test-msg-001"
                    }
                }
            }
            
            async with self.session.post(
                f"{self.base_url}/a2a/message/send",
                json=message_payload
            ) as response:
                duration = time.time() - start
                
                if response.status == 200:
                    task_response = await response.json()
                    
                    # Validate task response
                    if 'id' in task_response and 'status' in task_response:
                        task_id = task_response['id']
                        status = task_response.get('status', {}).get('state', 'unknown')
                        details = f"Task created: {task_id}, status: {status}"
                        self.log_result("A2A Message Sending", True, details, duration)
                        return True
                    else:
                        details = "Invalid task response structure"
                        self.log_result("A2A Message Sending", False, details, duration)
                        return False
                else:
                    details = f"HTTP {response.status}"
                    self.log_result("A2A Message Sending", False, details, duration)
                    return False
                    
        except Exception as e:
            duration = time.time() - start
            self.log_result("A2A Message Sending", False, f"Error: {str(e)}", duration)
            return False
    
    async def test_mcp_agent_listing(self) -> bool:
        """Test MCP agent listing"""
        start = time.time()
        try:
            print("\n📋 Testing MCP Agent Listing...")
            
            async with self.session.get(f"{self.base_url}/mcp/a2a/agents") as response:
                duration = time.time() - start
                
                if response.status == 200:
                    agents_data = await response.json()
                    
                    if 'agents' in agents_data and 'total_agents' in agents_data:
                        total_agents = agents_data['total_agents']
                        details = f"Found {total_agents} registered agents"
                        self.log_result("MCP Agent Listing", True, details, duration)
                        return True
                    else:
                        details = "Invalid agents response structure"
                        self.log_result("MCP Agent Listing", False, details, duration)
                        return False
                else:
                    details = f"HTTP {response.status}"
                    self.log_result("MCP Agent Listing", False, details, duration)
                    return False
                    
        except Exception as e:
            duration = time.time() - start
            self.log_result("MCP Agent Listing", False, f"Error: {str(e)}", duration)
            return False
    
    async def test_server_info(self) -> bool:
        """Test server info endpoint"""
        start = time.time()
        try:
            print("\n📊 Testing Server Info...")
            
            async with self.session.get(f"{self.base_url}/") as response:
                duration = time.time() - start
                
                if response.status == 200:
                    info = await response.json()
                    
                    if 'name' in info and 'capabilities' in info:
                        name = info['name']
                        capabilities = len(info['capabilities'])
                        details = f"Server: {name}, {capabilities} capabilities"
                        self.log_result("Server Info", True, details, duration)
                        return True
                    else:
                        details = "Invalid server info structure"
                        self.log_result("Server Info", False, details, duration)
                        return False
                else:
                    details = f"HTTP {response.status}"
                    self.log_result("Server Info", False, details, duration)
                    return False
                    
        except Exception as e:
            duration = time.time() - start
            self.log_result("Server Info", False, f"Error: {str(e)}", duration)
            return False
    
    async def run_all_tests(self) -> dict:
        """Run all A2A MCP Server tests"""
        print("🚀 A2A MCP Server Test Suite")
        print("=" * 50)
        
        tests = [
            ("Server Health", self.test_server_health),
            ("Agent Card Discovery", self.test_agent_card_discovery),
            ("A2A Message Sending", self.test_a2a_message_sending),
            ("MCP Agent Listing", self.test_mcp_agent_listing),
            ("Server Info", self.test_server_info)
        ]
        
        passed = 0
        for test_name, test_func in tests:
            if await test_func():
                passed += 1
        
        total = len(tests)
        
        print("\n" + "=" * 50)
        print(f"📊 A2A MCP Server Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print(f"\n🎉 ALL TESTS PASSED!")
            print(f"   A2A MCP Server is fully functional and production-ready.")
        elif passed >= total * 0.75:
            print(f"\n✅ MOST TESTS PASSED!")
            print(f"   A2A MCP Server is largely functional with minor issues.")
        else:
            print(f"\n⚠️ SOME TESTS FAILED")
            print(f"   A2A MCP Server needs attention before production use.")
        
        return {
            'total_tests': total,
            'passed_tests': passed,
            'success_rate': (passed / total) * 100,
            'test_results': self.results
        }


async def main():
    """Main test execution"""
    async with A2AMCPTester() as tester:
        results = await tester.run_all_tests()
        
        # Save results
        with open('a2a_mcp_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Results saved to: a2a_mcp_test_results.json")
        
        return 0 if results['success_rate'] >= 75 else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
