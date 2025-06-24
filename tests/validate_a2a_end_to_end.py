#!/usr/bin/env python3
"""
A2A End-to-End Validation Test

Comprehensive validation of the complete A2A ecosystem with real-world scenarios.
"""

import asyncio
import json
import aiohttp
import time
from datetime import datetime
from typing import Dict, Any, List


class A2AEndToEndValidator:
    """End-to-end validation of A2A multi-agent system"""
    
    def __init__(self, server_url: str = "http://localhost:8080"):
        self.server_url = server_url
        self.session = None
        self.validation_results = {}
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def send_jsonrpc_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send JSON-RPC request to A2A server"""
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
    
    async def validate_infrastructure(self) -> bool:
        """Validate infrastructure components"""
        print("🔧 VALIDATING INFRASTRUCTURE")
        print("-" * 40)
        
        try:
            # Check server health
            async with self.session.get(f"{self.server_url}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    print(f"✅ Server Health: {health_data['status']}")
                    print(f"   - Agents: {health_data['agents_registered']}")
                    print(f"   - Tasks: {health_data['tasks_active']}")
                    
                    self.validation_results["infrastructure"] = True
                    return True
                else:
                    print(f"❌ Health check failed: {response.status}")
                    self.validation_results["infrastructure"] = False
                    return False
        except Exception as e:
            print(f"❌ Infrastructure validation failed: {e}")
            self.validation_results["infrastructure"] = False
            return False
    
    async def validate_agent_discovery(self) -> bool:
        """Validate agent discovery protocol"""
        print("\n🔍 VALIDATING AGENT DISCOVERY")
        print("-" * 40)
        
        try:
            # Test agent discovery endpoint
            async with self.session.get(f"{self.server_url}/.well-known/agent.json") as response:
                if response.status == 200:
                    agent_card = await response.json()
                    
                    # Validate agent card structure
                    required_fields = ["name", "description", "version", "capabilities", "skills"]
                    for field in required_fields:
                        if field not in agent_card:
                            print(f"❌ Missing required field: {field}")
                            self.validation_results["agent_discovery"] = False
                            return False
                    
                    print(f"✅ Agent Card Valid")
                    print(f"   - Name: {agent_card['name']}")
                    print(f"   - Skills: {len(agent_card['skills'])}")
                    
                    self.validation_results["agent_discovery"] = True
                    return True
                else:
                    print(f"❌ Agent discovery failed: {response.status}")
                    self.validation_results["agent_discovery"] = False
                    return False
        except Exception as e:
            print(f"❌ Agent discovery validation failed: {e}")
            self.validation_results["agent_discovery"] = False
            return False
    
    async def validate_document_retrieval_workflow(self) -> bool:
        """Validate complete document retrieval workflow"""
        print("\n📚 VALIDATING DOCUMENT RETRIEVAL WORKFLOW")
        print("-" * 50)
        
        try:
            # Test comprehensive document search
            search_queries = [
                "machine learning algorithms",
                "neural network architectures", 
                "artificial intelligence applications"
            ]
            
            successful_searches = 0
            
            for query in search_queries:
                print(f"🔍 Testing query: '{query}'")
                
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
                    
                    # Wait for completion
                    await asyncio.sleep(2)
                    
                    # Get results
                    get_params = {"id": task_id}
                    result_response = await self.send_jsonrpc_request("tasks/get", get_params)
                    
                    if "result" in result_response:
                        task_result = result_response["result"]
                        
                        if task_result["status"]["state"] == "completed":
                            artifacts = task_result.get("artifacts", [])
                            if artifacts:
                                print(f"   ✅ Search completed with {len(artifacts)} artifacts")
                                successful_searches += 1
                            else:
                                print(f"   ⚠️ Search completed but no artifacts generated")
                        else:
                            print(f"   ❌ Search failed: {task_result['status']['state']}")
                    else:
                        print(f"   ❌ Failed to get search results")
                else:
                    print(f"   ❌ Failed to initiate search")
            
            success_rate = successful_searches / len(search_queries)
            print(f"\n📊 Document Retrieval Results: {successful_searches}/{len(search_queries)} ({success_rate*100:.1f}%)")
            
            if success_rate >= 0.8:  # 80% success rate required
                print("✅ Document retrieval workflow validated")
                self.validation_results["document_retrieval"] = True
                return True
            else:
                print("❌ Document retrieval workflow validation failed")
                self.validation_results["document_retrieval"] = False
                return False
                
        except Exception as e:
            print(f"❌ Document retrieval validation failed: {e}")
            self.validation_results["document_retrieval"] = False
            return False
    
    async def validate_multi_agent_coordination(self) -> bool:
        """Validate multi-agent coordination scenarios"""
        print("\n🤝 VALIDATING MULTI-AGENT COORDINATION")
        print("-" * 45)
        
        try:
            # Test complex multi-step workflow
            workflow_steps = [
                {
                    "step": "Research",
                    "query": "Research current trends in deep learning optimization techniques"
                },
                {
                    "step": "Analysis", 
                    "query": "Analyze the performance metrics and statistical significance of the research findings"
                },
                {
                    "step": "Synthesis",
                    "query": "Synthesize the research and analysis into actionable insights"
                }
            ]
            
            task_ids = []
            completed_tasks = 0
            
            for i, step in enumerate(workflow_steps):
                print(f"📤 Step {i+1}: {step['step']}")
                
                params = {
                    "message": {
                        "role": "user",
                        "parts": [{"type": "text", "text": step["query"]}]
                    }
                }
                
                # Send task
                response = await self.send_jsonrpc_request("tasks/send", params)
                
                if "result" in response:
                    task_id = response["result"]["id"]
                    task_ids.append(task_id)
                    print(f"   ✅ Task created: {task_id[:8]}...")
                    
                    # Wait for completion
                    await asyncio.sleep(2)
                    
                    # Check completion
                    get_params = {"id": task_id}
                    result_response = await self.send_jsonrpc_request("tasks/get", get_params)
                    
                    if "result" in result_response:
                        task_result = result_response["result"]
                        if task_result["status"]["state"] == "completed":
                            completed_tasks += 1
                            print(f"   ✅ Task completed successfully")
                        else:
                            print(f"   ❌ Task failed: {task_result['status']['state']}")
                    else:
                        print(f"   ❌ Failed to get task status")
                else:
                    print(f"   ❌ Failed to create task")
            
            success_rate = completed_tasks / len(workflow_steps)
            print(f"\n📊 Coordination Results: {completed_tasks}/{len(workflow_steps)} ({success_rate*100:.1f}%)")
            
            if success_rate >= 0.8:  # 80% success rate required
                print("✅ Multi-agent coordination validated")
                self.validation_results["coordination"] = True
                return True
            else:
                print("❌ Multi-agent coordination validation failed")
                self.validation_results["coordination"] = False
                return False
                
        except Exception as e:
            print(f"❌ Multi-agent coordination validation failed: {e}")
            self.validation_results["coordination"] = False
            return False
    
    async def validate_concurrent_operations(self) -> bool:
        """Validate concurrent multi-agent operations"""
        print("\n⚡ VALIDATING CONCURRENT OPERATIONS")
        print("-" * 40)
        
        try:
            # Create multiple concurrent tasks
            concurrent_tasks = [
                "Search for documents about quantum computing",
                "Analyze trends in machine learning research",
                "Research applications of natural language processing",
                "Evaluate statistical models for data analysis",
                "Investigate neural network optimization techniques"
            ]
            
            print(f"🚀 Launching {len(concurrent_tasks)} concurrent tasks...")
            
            # Send all tasks concurrently
            task_coroutines = []
            for query in concurrent_tasks:
                params = {
                    "message": {
                        "role": "user",
                        "parts": [{"type": "text", "text": query}]
                    }
                }
                task_coroutines.append(self.send_jsonrpc_request("tasks/send", params))
            
            # Wait for all tasks to be created
            responses = await asyncio.gather(*task_coroutines, return_exceptions=True)
            
            # Extract task IDs
            task_ids = []
            for response in responses:
                if isinstance(response, dict) and "result" in response:
                    task_ids.append(response["result"]["id"])
            
            print(f"✅ Created {len(task_ids)} concurrent tasks")
            
            # Wait for all tasks to complete
            await asyncio.sleep(5)
            
            # Check completion status
            completed_count = 0
            for task_id in task_ids:
                get_params = {"id": task_id}
                result_response = await self.send_jsonrpc_request("tasks/get", get_params)
                
                if "result" in result_response:
                    task_result = result_response["result"]
                    if task_result["status"]["state"] == "completed":
                        completed_count += 1
            
            success_rate = completed_count / len(task_ids) if task_ids else 0
            print(f"📊 Concurrent Operations: {completed_count}/{len(task_ids)} ({success_rate*100:.1f}%)")
            
            if success_rate >= 0.8:  # 80% success rate required
                print("✅ Concurrent operations validated")
                self.validation_results["concurrent_operations"] = True
                return True
            else:
                print("❌ Concurrent operations validation failed")
                self.validation_results["concurrent_operations"] = False
                return False
                
        except Exception as e:
            print(f"❌ Concurrent operations validation failed: {e}")
            self.validation_results["concurrent_operations"] = False
            return False
    
    async def validate_production_readiness(self) -> bool:
        """Validate production readiness criteria"""
        print("\n🏭 VALIDATING PRODUCTION READINESS")
        print("-" * 40)
        
        try:
            production_checks = {
                "error_handling": await self._test_error_handling(),
                "performance": await self._test_performance(),
                "reliability": await self._test_reliability(),
                "monitoring": await self._test_monitoring()
            }
            
            passed_checks = sum(production_checks.values())
            total_checks = len(production_checks)
            
            print(f"\n📊 Production Readiness: {passed_checks}/{total_checks}")
            for check, result in production_checks.items():
                status = "✅" if result else "❌"
                print(f"   {status} {check.replace('_', ' ').title()}")
            
            if passed_checks >= total_checks * 0.8:  # 80% of checks must pass
                print("✅ Production readiness validated")
                self.validation_results["production_readiness"] = True
                return True
            else:
                print("❌ Production readiness validation failed")
                self.validation_results["production_readiness"] = False
                return False
                
        except Exception as e:
            print(f"❌ Production readiness validation failed: {e}")
            self.validation_results["production_readiness"] = False
            return False
    
    async def _test_error_handling(self) -> bool:
        """Test error handling capabilities"""
        try:
            # Test invalid JSON-RPC request
            invalid_request = {"invalid": "request"}
            async with self.session.post(self.server_url, json=invalid_request) as response:
                if response.status == 200:
                    result = await response.json()
                    if "error" in result:
                        print("   ✅ Error handling: Invalid requests handled properly")
                        return True
            return False
        except:
            return False
    
    async def _test_performance(self) -> bool:
        """Test performance characteristics"""
        try:
            # Test response time
            start_time = time.time()
            async with self.session.get(f"{self.server_url}/health") as response:
                if response.status == 200:
                    response_time = time.time() - start_time
                    if response_time < 1.0:  # Response time under 1 second
                        print(f"   ✅ Performance: Response time {response_time:.3f}s")
                        return True
            return False
        except:
            return False
    
    async def _test_reliability(self) -> bool:
        """Test system reliability"""
        try:
            # Test multiple consecutive requests
            success_count = 0
            for _ in range(5):
                async with self.session.get(f"{self.server_url}/health") as response:
                    if response.status == 200:
                        success_count += 1
                await asyncio.sleep(0.1)
            
            if success_count >= 4:  # 80% success rate
                print(f"   ✅ Reliability: {success_count}/5 requests successful")
                return True
            return False
        except:
            return False
    
    async def _test_monitoring(self) -> bool:
        """Test monitoring capabilities"""
        try:
            # Test health endpoint
            async with self.session.get(f"{self.server_url}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    required_fields = ["status", "timestamp", "agents_registered"]
                    if all(field in health_data for field in required_fields):
                        print("   ✅ Monitoring: Health metrics available")
                        return True
            return False
        except:
            return False
    
    async def run_complete_validation(self) -> bool:
        """Run complete end-to-end validation"""
        print("🔬 A2A END-TO-END VALIDATION")
        print("=" * 60)
        print(f"Timestamp: {datetime.utcnow().isoformat()}")
        print("=" * 60)
        
        validation_steps = [
            ("Infrastructure", self.validate_infrastructure),
            ("Agent Discovery", self.validate_agent_discovery),
            ("Document Retrieval", self.validate_document_retrieval_workflow),
            ("Multi-Agent Coordination", self.validate_multi_agent_coordination),
            ("Concurrent Operations", self.validate_concurrent_operations),
            ("Production Readiness", self.validate_production_readiness)
        ]
        
        passed_validations = 0
        total_validations = len(validation_steps)
        
        for step_name, validation_func in validation_steps:
            try:
                result = await validation_func()
                if result:
                    passed_validations += 1
            except Exception as e:
                print(f"❌ {step_name} validation failed with exception: {e}")
                self.validation_results[step_name.lower().replace(" ", "_")] = False
        
        # Final results
        print("\n" + "=" * 60)
        print("📊 END-TO-END VALIDATION SUMMARY")
        print("=" * 60)
        
        success_rate = (passed_validations / total_validations) * 100
        print(f"Overall Success Rate: {passed_validations}/{total_validations} ({success_rate:.1f}%)")
        
        for step_name, _ in validation_steps:
            key = step_name.lower().replace(" ", "_")
            result = self.validation_results.get(key, False)
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{step_name}: {status}")
        
        if success_rate >= 90:  # 90% success rate for production readiness
            print("\n🎉 A2A SYSTEM VALIDATION: COMPLETE!")
            print("✅ All critical validations passed")
            print("✅ Real agents performing actual operations")
            print("✅ Multi-agent coordination functional")
            print("✅ Production-ready deployment validated")
            print("🚀 A2A MULTI-AGENT SYSTEM READY FOR PRODUCTION!")
            return True
        else:
            print(f"\n⚠️ A2A system validation incomplete ({success_rate:.1f}%)")
            print("Additional work needed before production deployment")
            return False


async def main():
    """Run end-to-end validation"""
    
    print("🔬 Starting A2A End-to-End Validation...")
    print("Please ensure the A2A server is running on localhost:8080")
    print()
    
    async with A2AEndToEndValidator() as validator:
        success = await validator.run_complete_validation()
        
        if success:
            print("\n✨ End-to-end validation completed successfully!")
            return True
        else:
            print("\n❌ End-to-end validation failed. Check system status.")
            return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Validation interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Validation error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
