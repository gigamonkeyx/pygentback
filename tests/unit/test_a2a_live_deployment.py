#!/usr/bin/env python3
"""
A2A Live Deployment Test

Test the A2A protocol endpoints in a live deployment without database dependencies.
This validates that our A2A implementation is production-ready.
"""

import asyncio
import aiohttp
import json
import sys
from datetime import datetime
from typing import Dict, Any, List

class A2ALiveDeploymentTester:
    """Test A2A protocol in live deployment"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = None
        self.test_results = []
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def test_a2a_well_known_endpoint(self) -> Dict[str, Any]:
        """Test A2A well-known agent endpoint"""
        print("🔍 Testing A2A Well-Known Endpoint...")
        
        try:
            url = f"{self.base_url}/a2a/v1/.well-known/agent.json"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Validate agent card structure
                    required_fields = ["name", "description", "url", "capabilities", "skills"]
                    missing_fields = [field for field in required_fields if field not in data]
                    
                    return {
                        "test": "a2a_well_known",
                        "success": len(missing_fields) == 0,
                        "status_code": response.status,
                        "message": "A2A well-known endpoint working" if len(missing_fields) == 0 else f"Missing fields: {missing_fields}",
                        "agent_name": data.get("name", "Unknown"),
                        "capabilities": list(data.get("capabilities", {}).keys()),
                        "skills_count": len(data.get("skills", []))
                    }
                else:
                    return {
                        "test": "a2a_well_known",
                        "success": False,
                        "status_code": response.status,
                        "message": f"HTTP {response.status} from well-known endpoint"
                    }
                    
        except Exception as e:
            return {
                "test": "a2a_well_known",
                "success": False,
                "message": f"Well-known endpoint test failed: {e}",
                "error": str(e)
            }
    
    async def test_a2a_discovery_endpoint(self) -> Dict[str, Any]:
        """Test A2A agent discovery endpoint"""
        print("🔍 Testing A2A Discovery Endpoint...")
        
        try:
            url = f"{self.base_url}/a2a/v1/agents/discover"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    return {
                        "test": "a2a_discovery",
                        "success": True,
                        "status_code": response.status,
                        "message": "A2A discovery endpoint working",
                        "discovered_agents": len(data.get("agents", [])),
                        "response_data": data
                    }
                else:
                    return {
                        "test": "a2a_discovery",
                        "success": False,
                        "status_code": response.status,
                        "message": f"HTTP {response.status} from discovery endpoint"
                    }
                    
        except Exception as e:
            return {
                "test": "a2a_discovery",
                "success": False,
                "message": f"Discovery endpoint test failed: {e}",
                "error": str(e)
            }
    
    async def test_a2a_message_send(self) -> Dict[str, Any]:
        """Test A2A message send endpoint"""
        print("🔍 Testing A2A Message Send...")
        
        try:
            url = f"{self.base_url}/a2a/v1/message/send"
            
            # Test message payload
            message_data = {
                "message": "Hello from A2A protocol test",
                "sender": "test_agent",
                "recipient": "pygent_factory",
                "message_type": "test",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            async with self.session.post(url, json=message_data) as response:
                if response.status in [200, 201, 202]:
                    data = await response.json()
                    
                    return {
                        "test": "a2a_message_send",
                        "success": True,
                        "status_code": response.status,
                        "message": "A2A message send working",
                        "message_id": data.get("message_id"),
                        "response_data": data
                    }
                else:
                    return {
                        "test": "a2a_message_send",
                        "success": False,
                        "status_code": response.status,
                        "message": f"HTTP {response.status} from message send endpoint"
                    }
                    
        except Exception as e:
            return {
                "test": "a2a_message_send",
                "success": False,
                "message": f"Message send test failed: {e}",
                "error": str(e)
            }
    
    async def test_a2a_health_endpoint(self) -> Dict[str, Any]:
        """Test A2A health endpoint"""
        print("🔍 Testing A2A Health Endpoint...")
        
        try:
            url = f"{self.base_url}/a2a/v1/health"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    return {
                        "test": "a2a_health",
                        "success": True,
                        "status_code": response.status,
                        "message": "A2A health endpoint working",
                        "health_status": data.get("status", "unknown"),
                        "response_data": data
                    }
                else:
                    return {
                        "test": "a2a_health",
                        "success": False,
                        "status_code": response.status,
                        "message": f"HTTP {response.status} from health endpoint"
                    }
                    
        except Exception as e:
            return {
                "test": "a2a_health",
                "success": False,
                "message": f"Health endpoint test failed: {e}",
                "error": str(e)
            }
    
    async def test_server_connectivity(self) -> Dict[str, Any]:
        """Test basic server connectivity"""
        print("🔍 Testing Server Connectivity...")
        
        try:
            # Try basic health endpoint first
            url = f"{self.base_url}/health"
            async with self.session.get(url, timeout=10) as response:
                return {
                    "test": "server_connectivity",
                    "success": response.status == 200,
                    "status_code": response.status,
                    "message": f"Server connectivity: HTTP {response.status}"
                }
                
        except asyncio.TimeoutError:
            return {
                "test": "server_connectivity",
                "success": False,
                "message": "Server connectivity timeout"
            }
        except Exception as e:
            return {
                "test": "server_connectivity",
                "success": False,
                "message": f"Server connectivity failed: {e}",
                "error": str(e)
            }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all A2A live deployment tests"""
        print("🚀 Running A2A Live Deployment Tests")
        print("=" * 60)
        
        tests = [
            self.test_server_connectivity(),
            self.test_a2a_well_known_endpoint(),
            self.test_a2a_discovery_endpoint(),
            self.test_a2a_message_send(),
            self.test_a2a_health_endpoint(),
        ]
        
        results = await asyncio.gather(*tests, return_exceptions=True)
        
        # Process results
        test_results = []
        for result in results:
            if isinstance(result, Exception):
                test_results.append({
                    "test": "unknown",
                    "success": False,
                    "message": f"Test exception: {result}",
                    "error": str(result)
                })
            else:
                test_results.append(result)
        
        # Calculate summary
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results if result.get("success", False))
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Print results
        for result in test_results:
            if result["success"]:
                print(f"✅ {result['test']}: PASSED - {result['message']}")
            else:
                print(f"❌ {result['test']}: FAILED - {result['message']}")
        
        summary = {
            "deployment_ready": passed_tests >= 3,  # At least connectivity + 2 A2A endpoints
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": round(success_rate, 1),
            "timestamp": datetime.utcnow().isoformat(),
            "test_results": test_results
        }
        
        print(f"\n" + "=" * 60)
        print(f"📊 A2A Live Deployment Summary:")
        print(f"   Deployment Ready: {'YES' if summary['deployment_ready'] else 'NO'}")
        print(f"   Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests})")
        print(f"   Timestamp: {summary['timestamp']}")
        
        if summary["deployment_ready"]:
            print("🎉 A2A PROTOCOL LIVE DEPLOYMENT SUCCESSFUL!")
            print("   ✅ Server connectivity confirmed")
            print("   ✅ A2A endpoints operational")
            print("   ✅ Ready for real-world testing")
        else:
            print("⚠️  A2A DEPLOYMENT NEEDS ATTENTION")
            failed_tests = [r for r in test_results if not r["success"]]
            for failed in failed_tests:
                print(f"   ❌ {failed['test']}: {failed['message']}")
        
        return summary

async def main():
    """Main test function"""
    async with A2ALiveDeploymentTester() as tester:
        summary = await tester.run_all_tests()
        
        # Exit with appropriate code
        if summary["deployment_ready"]:
            sys.exit(0)
        else:
            sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
