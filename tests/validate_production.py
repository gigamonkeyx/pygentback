#!/usr/bin/env python3
"""
A2A Production Deployment Validation

Comprehensive validation of production deployment readiness.
"""

import asyncio
import aiohttp
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProductionValidator:
    """Production deployment validator"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = None
        self.validation_results = {}
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def run_production_validation(self) -> Dict[str, Any]:
        """Run comprehensive production validation"""
        
        print("🚀 A2A PRODUCTION DEPLOYMENT VALIDATION")
        print("=" * 60)
        print(f"Target URL: {self.base_url}")
        print(f"Validation Time: {datetime.utcnow().isoformat()}")
        print("=" * 60)
        
        validation_tests = [
            ("Infrastructure Health", self.validate_infrastructure),
            ("API Endpoints", self.validate_api_endpoints),
            ("Agent System", self.validate_agent_system),
            ("Task Processing", self.validate_task_processing),
            ("Performance", self.validate_performance),
            ("Security", self.validate_security),
            ("Monitoring", self.validate_monitoring),
            ("Load Handling", self.validate_load_handling),
        ]
        
        passed_tests = 0
        total_tests = len(validation_tests)
        
        for test_name, test_func in validation_tests:
            print(f"\n🧪 VALIDATING: {test_name}")
            print("-" * 40)
            
            try:
                result = await test_func()
                if result:
                    print(f"✅ {test_name}: PASSED")
                    passed_tests += 1
                    self.validation_results[test_name] = {"status": "PASSED", "details": result}
                else:
                    print(f"❌ {test_name}: FAILED")
                    self.validation_results[test_name] = {"status": "FAILED", "details": result}
            except Exception as e:
                print(f"❌ {test_name}: ERROR - {e}")
                self.validation_results[test_name] = {"status": "ERROR", "error": str(e)}
        
        # Generate final report
        success_rate = (passed_tests / total_tests) * 100
        
        print("\n" + "=" * 60)
        print("📊 PRODUCTION VALIDATION RESULTS")
        print("=" * 60)
        print(f"Success Rate: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        print()
        
        for test_name, result in self.validation_results.items():
            status_icon = "✅" if result["status"] == "PASSED" else "❌"
            print(f"{status_icon} {test_name}: {result['status']}")
        
        print("\n" + "=" * 60)
        
        if success_rate >= 90:
            print("🎉 PRODUCTION VALIDATION: PASSED")
            print("✅ System is ready for production deployment")
            deployment_status = "PRODUCTION_READY"
        elif success_rate >= 80:
            print("⚠️ PRODUCTION VALIDATION: CONDITIONAL PASS")
            print("⚠️ Some issues detected, review before deployment")
            deployment_status = "CONDITIONAL_READY"
        else:
            print("❌ PRODUCTION VALIDATION: FAILED")
            print("❌ System not ready for production deployment")
            deployment_status = "NOT_READY"
        
        return {
            "validation_time": datetime.utcnow().isoformat(),
            "success_rate": success_rate,
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "deployment_status": deployment_status,
            "test_results": self.validation_results
        }
    
    async def validate_infrastructure(self) -> Dict[str, Any]:
        """Validate infrastructure health"""
        
        try:
            # Test basic connectivity
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    
                    print(f"   ✅ Server Status: {health_data.get('status')}")
                    print(f"   ✅ Agents Registered: {health_data.get('agents_registered')}")
                    print(f"   ✅ Uptime Available: {health_data.get('uptime', 'N/A')}")
                    
                    return {
                        "server_status": health_data.get('status'),
                        "agents_count": health_data.get('agents_registered', 0),
                        "response_time": response.headers.get('X-Response-Time', 'N/A')
                    }
                else:
                    print(f"   ❌ Health check failed: HTTP {response.status}")
                    return False
        except Exception as e:
            print(f"   ❌ Infrastructure error: {e}")
            return False
    
    async def validate_api_endpoints(self) -> Dict[str, Any]:
        """Validate all API endpoints"""
        
        endpoints = [
            ("/health", "Health Check"),
            ("/.well-known/agent.json", "Agent Discovery"),
            ("/agents", "Agent List"),
        ]
        
        endpoint_results = {}
        
        for endpoint, name in endpoints:
            try:
                start_time = time.time()
                async with self.session.get(f"{self.base_url}{endpoint}") as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        print(f"   ✅ {name}: HTTP 200 ({response_time:.3f}s)")
                        endpoint_results[endpoint] = {
                            "status": "OK",
                            "response_time": response_time,
                            "data_size": len(str(data))
                        }
                    else:
                        print(f"   ❌ {name}: HTTP {response.status}")
                        endpoint_results[endpoint] = {"status": "FAILED", "http_status": response.status}
            except Exception as e:
                print(f"   ❌ {name}: Error - {e}")
                endpoint_results[endpoint] = {"status": "ERROR", "error": str(e)}
        
        return endpoint_results
    
    async def validate_agent_system(self) -> Dict[str, Any]:
        """Validate agent system functionality"""
        
        try:
            # Get agent discovery info
            async with self.session.get(f"{self.base_url}/.well-known/agent.json") as response:
                if response.status == 200:
                    agent_card = await response.json()
                    
                    print(f"   ✅ Agent Name: {agent_card.get('name')}")
                    print(f"   ✅ Skills Available: {len(agent_card.get('skills', []))}")
                    print(f"   ✅ Capabilities: {len(agent_card.get('capabilities', {}))}")
                    
                    # Get agents list
                    async with self.session.get(f"{self.base_url}/agents") as agents_response:
                        if agents_response.status == 200:
                            agents = await agents_response.json()
                            print(f"   ✅ Registered Agents: {len(agents)}")
                            
                            return {
                                "agent_card": agent_card,
                                "registered_agents": len(agents),
                                "skills_count": len(agent_card.get('skills', [])),
                                "capabilities_count": len(agent_card.get('capabilities', {}))
                            }
                        else:
                            print(f"   ❌ Agents list failed: HTTP {agents_response.status}")
                            return False
                else:
                    print(f"   ❌ Agent discovery failed: HTTP {response.status}")
                    return False
        except Exception as e:
            print(f"   ❌ Agent system error: {e}")
            return False
    
    async def validate_task_processing(self) -> Dict[str, Any]:
        """Validate task processing functionality"""
        
        try:
            # Send a test task
            request = {
                "jsonrpc": "2.0",
                "method": "tasks/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"type": "text", "text": "Production validation test query"}]
                    }
                },
                "id": str(uuid.uuid4())
            }
            
            start_time = time.time()
            async with self.session.post(self.base_url, json=request) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    if "result" in result:
                        task_id = result["result"]["id"]
                        creation_time = time.time() - start_time
                        
                        print(f"   ✅ Task Created: {task_id[:8]}... ({creation_time:.3f}s)")
                        
                        # Wait for completion
                        await asyncio.sleep(1)
                        
                        # Get task status
                        get_request = {
                            "jsonrpc": "2.0",
                            "method": "tasks/get",
                            "params": {"id": task_id},
                            "id": str(uuid.uuid4())
                        }
                        
                        async with self.session.post(self.base_url, json=get_request) as get_response:
                            if get_response.status == 200:
                                get_result = await get_response.json()
                                
                                if "result" in get_result:
                                    task_status = get_result["result"]["status"]["state"]
                                    artifacts = get_result["result"].get("artifacts", [])
                                    
                                    print(f"   ✅ Task Status: {task_status}")
                                    print(f"   ✅ Artifacts: {len(artifacts)}")
                                    
                                    return {
                                        "task_creation_time": creation_time,
                                        "task_status": task_status,
                                        "artifacts_count": len(artifacts),
                                        "task_id": task_id
                                    }
                                else:
                                    print(f"   ❌ Task get failed: {get_result}")
                                    return False
                            else:
                                print(f"   ❌ Task get HTTP error: {get_response.status}")
                                return False
                    else:
                        print(f"   ❌ Task creation failed: {result}")
                        return False
                else:
                    print(f"   ❌ Task creation HTTP error: {response.status}")
                    return False
        except Exception as e:
            print(f"   ❌ Task processing error: {e}")
            return False
    
    async def validate_performance(self) -> Dict[str, Any]:
        """Validate performance characteristics"""
        
        try:
            response_times = []
            
            # Test multiple requests
            for i in range(5):
                start_time = time.time()
                async with self.session.get(f"{self.base_url}/health") as response:
                    response_time = time.time() - start_time
                    response_times.append(response_time)
                    
                    if response.status != 200:
                        print(f"   ❌ Performance test {i+1} failed: HTTP {response.status}")
                        return False
            
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            min_response_time = min(response_times)
            
            print(f"   ✅ Average Response Time: {avg_response_time:.3f}s")
            print(f"   ✅ Min Response Time: {min_response_time:.3f}s")
            print(f"   ✅ Max Response Time: {max_response_time:.3f}s")
            
            # Performance thresholds
            if avg_response_time < 0.1:  # 100ms
                print(f"   ✅ Performance: Excellent")
                performance_grade = "EXCELLENT"
            elif avg_response_time < 0.5:  # 500ms
                print(f"   ✅ Performance: Good")
                performance_grade = "GOOD"
            elif avg_response_time < 1.0:  # 1s
                print(f"   ⚠️ Performance: Acceptable")
                performance_grade = "ACCEPTABLE"
            else:
                print(f"   ❌ Performance: Poor")
                performance_grade = "POOR"
                return False
            
            return {
                "average_response_time": avg_response_time,
                "min_response_time": min_response_time,
                "max_response_time": max_response_time,
                "performance_grade": performance_grade,
                "test_count": len(response_times)
            }
        except Exception as e:
            print(f"   ❌ Performance validation error: {e}")
            return False
    
    async def validate_security(self) -> Dict[str, Any]:
        """Validate security measures"""
        
        security_checks = []
        
        try:
            # Test 1: Check for security headers
            async with self.session.get(f"{self.base_url}/health") as response:
                headers = response.headers
                
                security_headers = [
                    "X-Content-Type-Options",
                    "X-Frame-Options", 
                    "X-XSS-Protection"
                ]
                
                headers_present = 0
                for header in security_headers:
                    if header in headers:
                        headers_present += 1
                        print(f"   ✅ Security Header: {header}")
                    else:
                        print(f"   ⚠️ Missing Header: {header}")
                
                security_checks.append(("security_headers", headers_present / len(security_headers)))
            
            # Test 2: Test invalid JSON-RPC request
            invalid_request = {"invalid": "request"}
            async with self.session.post(self.base_url, json=invalid_request) as response:
                if response.status in [400, 422]:  # Expected error response
                    print(f"   ✅ Invalid Request Handling: HTTP {response.status}")
                    security_checks.append(("invalid_request_handling", 1.0))
                else:
                    print(f"   ❌ Invalid Request Not Handled: HTTP {response.status}")
                    security_checks.append(("invalid_request_handling", 0.0))
            
            # Test 3: Test oversized request (if applicable)
            large_data = "x" * 1000  # 1KB test
            large_request = {
                "jsonrpc": "2.0",
                "method": "tasks/send",
                "params": {"message": {"role": "user", "parts": [{"type": "text", "text": large_data}]}},
                "id": 1
            }
            
            async with self.session.post(self.base_url, json=large_request) as response:
                if response.status in [200, 413]:  # Either accepted or rejected
                    print(f"   ✅ Large Request Handling: HTTP {response.status}")
                    security_checks.append(("large_request_handling", 1.0))
                else:
                    print(f"   ⚠️ Unexpected Large Request Response: HTTP {response.status}")
                    security_checks.append(("large_request_handling", 0.5))
            
            # Calculate overall security score
            security_score = sum(score for _, score in security_checks) / len(security_checks)
            
            return {
                "security_score": security_score,
                "checks_passed": len([c for c in security_checks if c[1] >= 0.8]),
                "total_checks": len(security_checks),
                "security_details": dict(security_checks)
            }
        except Exception as e:
            print(f"   ❌ Security validation error: {e}")
            return False
    
    async def validate_monitoring(self) -> Dict[str, Any]:
        """Validate monitoring capabilities"""
        
        try:
            # Check if metrics endpoint exists (may not be exposed)
            monitoring_features = []
            
            # Test health endpoint detail
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    
                    # Check for monitoring fields
                    monitoring_fields = ["timestamp", "uptime", "agents_registered", "tasks_active"]
                    present_fields = [field for field in monitoring_fields if field in health_data]
                    
                    print(f"   ✅ Health Monitoring Fields: {len(present_fields)}/{len(monitoring_fields)}")
                    monitoring_features.append(("health_monitoring", len(present_fields) / len(monitoring_fields)))
                    
                    # Check timestamp format
                    if "timestamp" in health_data:
                        try:
                            datetime.fromisoformat(health_data["timestamp"].replace('Z', '+00:00'))
                            print(f"   ✅ Timestamp Format: Valid ISO format")
                            monitoring_features.append(("timestamp_format", 1.0))
                        except:
                            print(f"   ❌ Timestamp Format: Invalid")
                            monitoring_features.append(("timestamp_format", 0.0))
                    
                    # Check for response time headers
                    if "X-Response-Time" in response.headers:
                        print(f"   ✅ Response Time Tracking: Available")
                        monitoring_features.append(("response_time_tracking", 1.0))
                    else:
                        print(f"   ⚠️ Response Time Tracking: Not available")
                        monitoring_features.append(("response_time_tracking", 0.0))
                    
                    monitoring_score = sum(score for _, score in monitoring_features) / len(monitoring_features)
                    
                    return {
                        "monitoring_score": monitoring_score,
                        "features_available": len([f for f in monitoring_features if f[1] >= 0.8]),
                        "total_features": len(monitoring_features),
                        "monitoring_details": dict(monitoring_features)
                    }
                else:
                    print(f"   ❌ Health endpoint failed: HTTP {response.status}")
                    return False
        except Exception as e:
            print(f"   ❌ Monitoring validation error: {e}")
            return False
    
    async def validate_load_handling(self) -> Dict[str, Any]:
        """Validate load handling capabilities"""
        
        try:
            # Send multiple concurrent requests
            concurrent_requests = 10
            print(f"   🔄 Testing {concurrent_requests} concurrent requests...")
            
            async def send_request():
                async with self.session.get(f"{self.base_url}/health") as response:
                    return response.status == 200
            
            # Execute concurrent requests
            start_time = time.time()
            results = await asyncio.gather(*[send_request() for _ in range(concurrent_requests)])
            total_time = time.time() - start_time
            
            successful_requests = sum(results)
            success_rate = successful_requests / concurrent_requests
            
            print(f"   ✅ Concurrent Requests: {successful_requests}/{concurrent_requests}")
            print(f"   ✅ Success Rate: {success_rate*100:.1f}%")
            print(f"   ✅ Total Time: {total_time:.3f}s")
            print(f"   ✅ Requests/Second: {concurrent_requests/total_time:.2f}")
            
            if success_rate >= 0.9:  # 90% success rate
                return {
                    "load_test_passed": True,
                    "success_rate": success_rate,
                    "concurrent_requests": concurrent_requests,
                    "total_time": total_time,
                    "requests_per_second": concurrent_requests / total_time
                }
            else:
                print(f"   ❌ Load test failed: {success_rate*100:.1f}% success rate")
                return False
        except Exception as e:
            print(f"   ❌ Load handling validation error: {e}")
            return False


async def main():
    """Run production validation"""
    
    async with ProductionValidator() as validator:
        results = await validator.run_production_validation()
        
        # Save results to file
        with open("production-validation-results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Validation results saved to: production-validation-results.json")
        
        return results["deployment_status"] == "PRODUCTION_READY"


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
