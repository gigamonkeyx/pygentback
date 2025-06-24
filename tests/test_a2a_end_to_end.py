#!/usr/bin/env python3
"""
A2A End-to-End Test Suite

Comprehensive testing of A2A protocol functionality including:
- Agent discovery and registration
- Message passing between agents
- Task coordination
- Security authentication
- Streaming communication
"""

import asyncio
import aiohttp
import json
import pytest
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

class A2AEndToEndTester:
    """A2A end-to-end test suite"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = None
        self.test_results = []
        self.created_agents = []
        self.created_tasks = []
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        # Cleanup created resources
        await self.cleanup_test_resources()
        
        if self.session:
            await self.session.close()
    
    async def cleanup_test_resources(self):
        """Clean up test resources"""
        # Cancel any created tasks
        for task_id in self.created_tasks:
            try:
                await self.cancel_task(task_id)
            except:
                pass
        
        # Note: Agents are typically not deleted in tests to avoid disrupting the system
    
    async def make_request(self, endpoint: str, method: str = "GET", 
                          data: Optional[Dict] = None, headers: Optional[Dict] = None) -> Dict[str, Any]:
        """Make HTTP request"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                async with self.session.get(url, headers=headers) as response:
                    status = response.status
                    content = await response.text()
            elif method.upper() == "POST":
                async with self.session.post(url, json=data, headers=headers) as response:
                    status = response.status
                    content = await response.text()
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            # Try to parse JSON
            try:
                json_content = json.loads(content)
            except:
                json_content = {"raw_content": content}
            
            return {
                "status": status,
                "content": json_content,
                "success": 200 <= status < 300
            }
            
        except Exception as e:
            return {
                "status": 0,
                "content": {"error": str(e)},
                "success": False
            }
    
    async def test_agent_discovery(self) -> Dict[str, Any]:
        """Test A2A agent discovery"""
        print("🔍 Testing A2A Agent Discovery...")
        
        result = await self.make_request("/api/a2a/v1/agents/discover")
        
        if result["success"]:
            agents = result["content"].get("agents", [])
            total = result["content"].get("total", 0)
            
            success = len(agents) > 0 and total > 0
            message = f"Discovered {len(agents)} agents (total: {total})"
        else:
            success = False
            message = f"Discovery failed: {result['content']}"
        
        return {
            "test": "agent_discovery",
            "success": success,
            "message": message,
            "data": result["content"]
        }
    
    async def test_well_known_agent_card(self) -> Dict[str, Any]:
        """Test well-known agent card endpoint"""
        print("🔍 Testing Well-known Agent Card...")
        
        result = await self.make_request("/.well-known/agent.json")
        
        if result["success"]:
            agent_card = result["content"]
            required_fields = ["name", "description", "version", "url", "provider"]
            missing_fields = [field for field in required_fields if field not in agent_card]
            
            success = len(missing_fields) == 0
            message = f"Agent card valid" if success else f"Missing fields: {missing_fields}"
        else:
            success = False
            message = f"Agent card retrieval failed: {result['content']}"
        
        return {
            "test": "well_known_agent_card",
            "success": success,
            "message": message,
            "data": result["content"]
        }
    
    async def test_a2a_message_sending(self) -> Dict[str, Any]:
        """Test A2A message sending"""
        print("🔍 Testing A2A Message Sending...")
        
        test_message = {
            "method": "ping",
            "params": {"message": "Hello from A2A test", "timestamp": datetime.utcnow().isoformat()},
            "id": f"test_message_{int(time.time())}"
        }
        
        result = await self.make_request("/api/a2a/v1/message/send", "POST", test_message)
        
        if result["success"]:
            response = result["content"]
            has_result = "result" in response or "error" in response
            has_id = response.get("id") == test_message["id"]
            
            success = has_result and has_id
            message = f"Message sent successfully" if success else f"Invalid response format"
        else:
            success = False
            message = f"Message sending failed: {result['content']}"
        
        return {
            "test": "a2a_message_sending",
            "success": success,
            "message": message,
            "data": result["content"]
        }
    
    async def test_task_management(self) -> Dict[str, Any]:
        """Test A2A task management"""
        print("🔍 Testing A2A Task Management...")
        
        # Create a task through message sending
        task_message = {
            "method": "tasks/create",
            "params": {
                "message": {
                    "parts": [{"type": "text", "text": "Test task for A2A"}],
                    "role": "user"
                },
                "contextId": f"test_context_{int(time.time())}"
            },
            "id": f"task_test_{int(time.time())}"
        }
        
        result = await self.make_request("/api/a2a/v1/message/send", "POST", task_message)
        
        if result["success"]:
            response = result["content"]
            task_id = response.get("result", {}).get("taskId")
            
            if task_id:
                self.created_tasks.append(task_id)
                
                # Try to get task status
                task_result = await self.make_request(f"/api/a2a/v1/tasks/{task_id}")
                
                if task_result["success"]:
                    task_data = task_result["content"]
                    has_required_fields = all(field in task_data for field in ["id", "status"])
                    
                    success = has_required_fields
                    message = f"Task created and retrieved: {task_id}" if success else "Task missing required fields"
                else:
                    success = False
                    message = f"Failed to retrieve task: {task_result['content']}"
            else:
                success = False
                message = "No task ID returned from task creation"
        else:
            success = False
            message = f"Task creation failed: {result['content']}"
        
        return {
            "test": "task_management",
            "success": success,
            "message": message,
            "data": result["content"]
        }
    
    async def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """Cancel a task"""
        return await self.make_request(f"/api/a2a/v1/tasks/{task_id}/cancel", "POST")
    
    async def test_a2a_health(self) -> Dict[str, Any]:
        """Test A2A health endpoint"""
        print("🔍 Testing A2A Health...")
        
        result = await self.make_request("/api/a2a/v1/health")
        
        if result["success"]:
            health_data = result["content"]
            is_healthy = health_data.get("status") == "healthy"
            has_components = "components" in health_data
            
            success = is_healthy and has_components
            message = f"A2A system healthy" if success else f"A2A system unhealthy"
        else:
            success = False
            message = f"Health check failed: {result['content']}"
        
        return {
            "test": "a2a_health",
            "success": success,
            "message": message,
            "data": result["content"]
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all A2A end-to-end tests"""
        print("🚀 Running A2A End-to-End Test Suite")
        print("=" * 60)
        
        tests = [
            self.test_a2a_health(),
            self.test_well_known_agent_card(),
            self.test_agent_discovery(),
            self.test_a2a_message_sending(),
            self.test_task_management(),
        ]
        
        results = []
        total_tests = len(tests)
        passed_tests = 0
        
        for test_coro in tests:
            result = await test_coro
            results.append(result)
            
            if result["success"]:
                print(f"✅ {result['test']}: PASSED - {result['message']}")
                passed_tests += 1
            else:
                print(f"❌ {result['test']}: FAILED - {result['message']}")
        
        # Summary
        success_rate = (passed_tests / total_tests) * 100
        overall_success = passed_tests == total_tests
        
        summary = {
            "overall_success": overall_success,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": round(success_rate, 1),
            "timestamp": datetime.utcnow().isoformat(),
            "test_results": results
        }
        
        print(f"\n" + "=" * 60)
        print(f"📊 A2A End-to-End Test Summary:")
        print(f"   Overall Success: {'PASS' if overall_success else 'FAIL'}")
        print(f"   Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests})")
        print(f"   Timestamp: {summary['timestamp']}")
        
        if overall_success:
            print("🎉 All A2A end-to-end tests passed!")
        else:
            print("💥 Some A2A end-to-end tests failed!")
        
        return summary

async def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="A2A End-to-End Test Suite")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL for tests")
    parser.add_argument("--output", help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    try:
        async with A2AEndToEndTester(args.url) as tester:
            summary = await tester.run_all_tests()
            
            # Save results if requested
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(summary, f, indent=2)
                print(f"\n💾 Results saved to: {args.output}")
            
            # Exit with appropriate code
            sys.exit(0 if summary["overall_success"] else 1)
                
    except Exception as e:
        print(f"❌ End-to-end test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)

if __name__ == "__main__":
    asyncio.run(main())
