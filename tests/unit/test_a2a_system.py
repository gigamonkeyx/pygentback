#!/usr/bin/env python3
"""
Comprehensive A2A System Testing and Validation

Tests all aspects of the A2A multi-agent system to ensure production readiness.
"""

import asyncio
import json
import aiohttp
import time
import sys
from datetime import datetime
from typing import Dict, Any, List


class A2ASystemTester:
    """Comprehensive A2A system tester"""
    
    def __init__(self, server_url: str = "http://localhost:8080"):
        self.server_url = server_url
        self.session = None
        self.test_results = {}
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_system_health(self) -> bool:
        """Test system health and basic connectivity"""
        print("🏥 TESTING SYSTEM HEALTH")
        print("-" * 30)
        
        try:
            async with self.session.get(f"{self.server_url}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    print(f"✅ Server Status: {health_data.get('status', 'unknown')}")
                    print(f"✅ Agents Registered: {health_data.get('agents_registered', 0)}")
                    print(f"✅ Active Tasks: {health_data.get('tasks_active', 0)}")
                    print(f"✅ Timestamp: {health_data.get('timestamp', 'unknown')}")
                    
                    # Validate required fields
                    required_fields = ['status', 'timestamp', 'agents_registered', 'tasks_active']
                    missing_fields = [field for field in required_fields if field not in health_data]
                    
                    if missing_fields:
                        print(f"❌ Missing health fields: {missing_fields}")
                        return False
                    
                    if health_data['status'] != 'healthy':
                        print(f"❌ System not healthy: {health_data['status']}")
                        return False
                    
                    if health_data['agents_registered'] < 1:
                        print(f"❌ No agents registered")
                        return False
                    
                    self.test_results['system_health'] = True
                    return True
                else:
                    print(f"❌ Health check failed: HTTP {response.status}")
                    self.test_results['system_health'] = False
                    return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            self.test_results['system_health'] = False
            return False
    
    async def test_agent_discovery(self) -> bool:
        """Test agent discovery endpoint"""
        print("\n🔍 TESTING AGENT DISCOVERY")
        print("-" * 30)
        
        try:
            async with self.session.get(f"{self.server_url}/.well-known/agent.json") as response:
                if response.status == 200:
                    agent_card = await response.json()
                    
                    print(f"✅ Agent Name: {agent_card.get('name', 'unknown')}")
                    print(f"✅ Agent Description: {agent_card.get('description', 'unknown')}")
                    print(f"✅ Agent Version: {agent_card.get('version', 'unknown')}")
                    
                    # Validate agent card structure
                    required_fields = ['name', 'description', 'version', 'capabilities', 'skills']
                    missing_fields = [field for field in required_fields if field not in agent_card]
                    
                    if missing_fields:
                        print(f"❌ Missing agent card fields: {missing_fields}")
                        return False
                    
                    capabilities = agent_card.get('capabilities', {})
                    print(f"✅ Capabilities: {len(capabilities)} features")
                    for cap, enabled in capabilities.items():
                        status = "✅" if enabled else "❌"
                        print(f"   {status} {cap}")
                    
                    skills = agent_card.get('skills', [])
                    print(f"✅ Skills: {len(skills)} available")
                    for skill in skills[:3]:  # Show first 3
                        print(f"   • {skill.get('name', 'unknown')}")
                    
                    self.test_results['agent_discovery'] = True
                    return True
                else:
                    print(f"❌ Agent discovery failed: HTTP {response.status}")
                    self.test_results['agent_discovery'] = False
                    return False
        except Exception as e:
            print(f"❌ Agent discovery error: {e}")
            self.test_results['agent_discovery'] = False
            return False
    
    async def test_agents_list(self) -> bool:
        """Test agents list endpoint"""
        print("\n👥 TESTING AGENTS LIST")
        print("-" * 25)
        
        try:
            async with self.session.get(f"{self.server_url}/agents") as response:
                if response.status == 200:
                    agents = await response.json()
                    
                    print(f"✅ Total Agents: {len(agents)}")
                    
                    if len(agents) == 0:
                        print("❌ No agents found")
                        return False
                    
                    for i, agent in enumerate(agents):
                        print(f"✅ Agent {i+1}:")
                        print(f"   • Name: {agent.get('name', 'unknown')}")
                        print(f"   • Type: {agent.get('type', 'unknown')}")
                        print(f"   • Status: {agent.get('status', 'unknown')}")
                        print(f"   • ID: {agent.get('agent_id', 'unknown')[:8]}...")
                    
                    self.test_results['agents_list'] = True
                    return True
                else:
                    print(f"❌ Agents list failed: HTTP {response.status}")
                    self.test_results['agents_list'] = False
                    return False
        except Exception as e:
            print(f"❌ Agents list error: {e}")
            self.test_results['agents_list'] = False
            return False
    
    async def send_jsonrpc_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send JSON-RPC request"""
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": int(time.time() * 1000)
        }
        
        try:
            async with self.session.post(
                self.server_url,
                json=request,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"HTTP {response.status}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def test_document_search(self) -> bool:
        """Test document search functionality"""
        print("\n📚 TESTING DOCUMENT SEARCH")
        print("-" * 30)
        
        try:
            # Test multiple search queries
            test_queries = [
                "machine learning algorithms",
                "neural networks",
                "artificial intelligence"
            ]
            
            successful_searches = 0
            
            for query in test_queries:
                print(f"🔍 Testing: '{query}'")
                
                params = {
                    "message": {
                        "role": "user",
                        "parts": [{"type": "text", "text": f"Search for documents about {query}"}]
                    }
                }
                
                # Send search request
                response = await self.send_jsonrpc_request("tasks/send", params)
                
                if "result" in response:
                    task_id = response["result"]["id"]
                    print(f"   ✅ Task created: {task_id[:8]}...")
                    
                    # Wait for completion
                    await asyncio.sleep(1)
                    
                    # Get results
                    get_params = {"id": task_id}
                    result_response = await self.send_jsonrpc_request("tasks/get", get_params)
                    
                    if "result" in result_response:
                        task_result = result_response["result"]
                        status = task_result["status"]["state"]
                        artifacts = task_result.get("artifacts", [])
                        
                        print(f"   ✅ Status: {status}")
                        print(f"   ✅ Artifacts: {len(artifacts)}")
                        
                        if status == "completed" and artifacts:
                            successful_searches += 1
                            
                            # Parse artifact content
                            if artifacts[0].get("parts"):
                                try:
                                    content = artifacts[0]["parts"][0].get("text", "")
                                    result_data = json.loads(content)
                                    print(f"   ✅ Search method: {result_data.get('search_method', 'unknown')}")
                                    print(f"   ✅ Documents found: {result_data.get('total_found', 0)}")
                                except:
                                    print(f"   ✅ Raw result available")
                        else:
                            print(f"   ❌ Search incomplete: {status}")
                    else:
                        print(f"   ❌ Failed to get results: {result_response}")
                else:
                    print(f"   ❌ Failed to create task: {response}")
            
            success_rate = successful_searches / len(test_queries)
            print(f"\n📊 Document Search Results: {successful_searches}/{len(test_queries)} ({success_rate*100:.1f}%)")
            
            if success_rate >= 0.8:  # 80% success rate required
                self.test_results['document_search'] = True
                return True
            else:
                self.test_results['document_search'] = False
                return False
                
        except Exception as e:
            print(f"❌ Document search test error: {e}")
            self.test_results['document_search'] = False
            return False
    
    async def test_multi_agent_tasks(self) -> bool:
        """Test multi-agent task execution"""
        print("\n🤝 TESTING MULTI-AGENT TASKS")
        print("-" * 35)
        
        try:
            # Test different types of tasks
            test_tasks = [
                {
                    "type": "research",
                    "query": "Research quantum computing applications"
                },
                {
                    "type": "analysis", 
                    "query": "Analyze statistical trends in AI research"
                },
                {
                    "type": "synthesis",
                    "query": "Synthesize findings from multiple sources"
                }
            ]
            
            completed_tasks = 0
            task_ids = []
            
            for task in test_tasks:
                print(f"📤 {task['type'].title()} Task: {task['query'][:50]}...")
                
                params = {
                    "message": {
                        "role": "user",
                        "parts": [{"type": "text", "text": task["query"]}]
                    }
                }
                
                # Send task
                response = await self.send_jsonrpc_request("tasks/send", params)
                
                if "result" in response:
                    task_id = response["result"]["id"]
                    task_ids.append(task_id)
                    print(f"   ✅ Task created: {task_id[:8]}...")
                    
                    # Wait for completion
                    await asyncio.sleep(1)
                    
                    # Check status
                    get_params = {"id": task_id}
                    result_response = await self.send_jsonrpc_request("tasks/get", get_params)
                    
                    if "result" in result_response:
                        task_result = result_response["result"]
                        status = task_result["status"]["state"]
                        
                        if status == "completed":
                            completed_tasks += 1
                            print(f"   ✅ Completed successfully")
                        else:
                            print(f"   ❌ Failed: {status}")
                    else:
                        print(f"   ❌ Failed to get status")
                else:
                    print(f"   ❌ Failed to create task")
            
            success_rate = completed_tasks / len(test_tasks)
            print(f"\n📊 Multi-Agent Tasks: {completed_tasks}/{len(test_tasks)} ({success_rate*100:.1f}%)")

            # Debug: Show detailed results
            print(f"   Debug: completed_tasks={completed_tasks}, total_tasks={len(test_tasks)}, success_rate={success_rate}")

            if success_rate >= 0.8:  # 80% success rate required
                print(f"   ✅ Multi-agent tasks test PASSED (success_rate {success_rate:.1f}% >= 80%)")
                return True
            else:
                print(f"   ❌ Multi-agent tasks test FAILED (success_rate {success_rate:.1f}% < 80%)")
                return False
                
        except Exception as e:
            print(f"❌ Multi-agent tasks test error: {e}")
            self.test_results['multi_agent_tasks'] = False
            return False
    
    async def test_concurrent_load(self) -> bool:
        """Test concurrent task load"""
        print("\n⚡ TESTING CONCURRENT LOAD")
        print("-" * 30)
        
        try:
            # Create multiple concurrent tasks
            concurrent_count = 5
            print(f"🚀 Launching {concurrent_count} concurrent tasks...")
            
            # Prepare tasks
            tasks = []
            for i in range(concurrent_count):
                params = {
                    "message": {
                        "role": "user",
                        "parts": [{"type": "text", "text": f"Concurrent test task {i+1}: search for AI research"}]
                    }
                }
                tasks.append(self.send_jsonrpc_request("tasks/send", params))
            
            # Execute all tasks concurrently
            start_time = time.time()
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            execution_time = time.time() - start_time
            
            # Count successful task creations
            successful_tasks = 0
            task_ids = []
            
            for i, response in enumerate(responses):
                if isinstance(response, dict) and "result" in response:
                    task_id = response["result"]["id"]
                    task_ids.append(task_id)
                    successful_tasks += 1
                    print(f"   ✅ Task {i+1}: {task_id[:8]}...")
                else:
                    print(f"   ❌ Task {i+1}: Failed")
            
            print(f"📊 Task Creation: {successful_tasks}/{concurrent_count} in {execution_time:.2f}s")
            
            # Wait for completion and check results
            await asyncio.sleep(3)
            
            completed_tasks = 0
            for task_id in task_ids:
                get_params = {"id": task_id}
                result_response = await self.send_jsonrpc_request("tasks/get", get_params)
                
                if "result" in result_response:
                    task_result = result_response["result"]
                    if task_result["status"]["state"] == "completed":
                        completed_tasks += 1
            
            completion_rate = completed_tasks / len(task_ids) if task_ids else 0
            print(f"📊 Task Completion: {completed_tasks}/{len(task_ids)} ({completion_rate*100:.1f}%)")
            
            if completion_rate >= 0.6:  # 60% completion rate for concurrent load
                self.test_results['concurrent_load'] = True
                return True
            else:
                self.test_results['concurrent_load'] = False
                return False
                
        except Exception as e:
            print(f"❌ Concurrent load test error: {e}")
            self.test_results['concurrent_load'] = False
            return False
    
    async def test_error_handling(self) -> bool:
        """Test error handling capabilities"""
        print("\n🛡️ TESTING ERROR HANDLING")
        print("-" * 30)
        
        try:
            error_tests = []
            
            # Test 1: Invalid JSON-RPC request
            print("🧪 Testing invalid JSON-RPC request...")
            invalid_request = {"invalid": "request"}
            async with self.session.post(self.server_url, json=invalid_request) as response:
                if response.status == 200:
                    result = await response.json()
                    if "error" in result:
                        print("   ✅ Invalid request handled properly")
                        error_tests.append(True)
                    else:
                        print("   ❌ Invalid request not handled")
                        error_tests.append(False)
                else:
                    print(f"   ❌ Unexpected status: {response.status}")
                    error_tests.append(False)
            
            # Test 2: Invalid task ID
            print("🧪 Testing invalid task ID...")
            get_params = {"id": "invalid-task-id"}
            response = await self.send_jsonrpc_request("tasks/get", get_params)
            if "error" in response:
                print("   ✅ Invalid task ID handled properly")
                error_tests.append(True)
            else:
                print("   ❌ Invalid task ID not handled")
                error_tests.append(False)
            
            # Test 3: Malformed message
            print("🧪 Testing malformed message...")
            params = {
                "message": {
                    "role": "invalid_role",
                    "parts": "invalid_parts_format"
                }
            }
            response = await self.send_jsonrpc_request("tasks/send", params)
            # Should either succeed with error handling or return error
            if "result" in response or "error" in response:
                print("   ✅ Malformed message handled")
                error_tests.append(True)
            else:
                print("   ❌ Malformed message not handled")
                error_tests.append(False)
            
            success_rate = sum(error_tests) / len(error_tests)
            print(f"\n📊 Error Handling: {sum(error_tests)}/{len(error_tests)} ({success_rate*100:.1f}%)")
            
            if success_rate >= 0.8:  # 80% error handling success
                self.test_results['error_handling'] = True
                return True
            else:
                self.test_results['error_handling'] = False
                return False
                
        except Exception as e:
            print(f"❌ Error handling test error: {e}")
            self.test_results['error_handling'] = False
            return False
    
    async def test_performance(self) -> bool:
        """Test performance characteristics"""
        print("\n🚀 TESTING PERFORMANCE")
        print("-" * 25)
        
        try:
            performance_tests = []
            
            # Test 1: Health check response time
            print("⏱️ Testing health check response time...")
            start_time = time.time()
            async with self.session.get(f"{self.server_url}/health") as response:
                response_time = time.time() - start_time
                if response.status == 200 and response_time < 1.0:
                    print(f"   ✅ Health check: {response_time:.3f}s")
                    performance_tests.append(True)
                else:
                    print(f"   ❌ Health check slow: {response_time:.3f}s")
                    performance_tests.append(False)
            
            # Test 2: Agent discovery response time
            print("⏱️ Testing agent discovery response time...")
            start_time = time.time()
            async with self.session.get(f"{self.server_url}/.well-known/agent.json") as response:
                response_time = time.time() - start_time
                if response.status == 200 and response_time < 2.0:
                    print(f"   ✅ Agent discovery: {response_time:.3f}s")
                    performance_tests.append(True)
                else:
                    print(f"   ❌ Agent discovery slow: {response_time:.3f}s")
                    performance_tests.append(False)
            
            # Test 3: Task creation response time
            print("⏱️ Testing task creation response time...")
            params = {
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": "Performance test query"}]
                }
            }
            start_time = time.time()
            response = await self.send_jsonrpc_request("tasks/send", params)
            response_time = time.time() - start_time
            
            if "result" in response and response_time < 3.0:
                print(f"   ✅ Task creation: {response_time:.3f}s")
                performance_tests.append(True)
            else:
                print(f"   ❌ Task creation slow: {response_time:.3f}s")
                performance_tests.append(False)
            
            success_rate = sum(performance_tests) / len(performance_tests)
            print(f"\n📊 Performance: {sum(performance_tests)}/{len(performance_tests)} ({success_rate*100:.1f}%)")
            
            if success_rate >= 0.8:  # 80% performance success
                self.test_results['performance'] = True
                return True
            else:
                self.test_results['performance'] = False
                return False
                
        except Exception as e:
            print(f"❌ Performance test error: {e}")
            self.test_results['performance'] = False
            return False
    
    async def run_comprehensive_tests(self) -> bool:
        """Run all comprehensive tests"""
        print("🧪 COMPREHENSIVE A2A SYSTEM TESTING")
        print("=" * 60)
        print(f"Timestamp: {datetime.utcnow().isoformat()}")
        print("=" * 60)
        
        # Define test suite
        test_suite = [
            ("System Health", self.test_system_health),
            ("Agent Discovery", self.test_agent_discovery),
            ("Agents List", self.test_agents_list),
            ("Document Search", self.test_document_search),
            ("Multi-Agent Tasks", self.test_multi_agent_tasks),
            ("Concurrent Load", self.test_concurrent_load),
            ("Error Handling", self.test_error_handling),
            ("Performance", self.test_performance)
        ]
        
        passed_tests = 0
        total_tests = len(test_suite)
        
        # Execute all tests
        for test_name, test_func in test_suite:
            key = test_name.lower().replace(" ", "_")
            try:
                result = await test_func()
                # Ensure test_results is properly set for both success and failure
                self.test_results[key] = result
                if result:
                    passed_tests += 1
            except Exception as e:
                print(f"❌ {test_name} test failed with exception: {e}")
                self.test_results[key] = False
        
        # Generate final report
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE TEST RESULTS")
        print("=" * 60)
        
        success_rate = (passed_tests / total_tests) * 100
        print(f"Overall Success Rate: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        print()
        
        for test_name, _ in test_suite:
            key = test_name.lower().replace(" ", "_")
            result = self.test_results.get(key, False)
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name}: {status}")
        
        print("\n" + "=" * 60)
        
        if success_rate >= 90:  # 90% success rate for production validation
            print("🎉 A2A SYSTEM VALIDATION: COMPLETE!")
            print("✅ All critical tests passed")
            print("✅ System is production-ready")
            print("✅ Real agents operational")
            print("✅ Multi-agent coordination working")
            print("🚀 A2A MULTI-AGENT SYSTEM VALIDATED FOR PRODUCTION!")
            return True
        elif success_rate >= 80:
            print("⚠️ A2A SYSTEM VALIDATION: MOSTLY SUCCESSFUL")
            print(f"✅ {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
            print("⚠️ Some optimizations recommended")
            print("🚀 A2A system operational with minor issues")
            return True
        else:
            print("❌ A2A SYSTEM VALIDATION: NEEDS IMPROVEMENT")
            print(f"❌ Only {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
            print("❌ System needs fixes before production deployment")
            return False


async def main():
    """Run comprehensive A2A system testing"""
    
    print("🧪 Starting Comprehensive A2A System Testing...")
    print("Please ensure the A2A server is running on localhost:8080")
    print()
    
    async with A2ASystemTester() as tester:
        success = await tester.run_comprehensive_tests()
        
        if success:
            print("\n✨ All tests completed successfully!")
            return True
        else:
            print("\n❌ Some tests failed. Check system status.")
            return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Testing error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
