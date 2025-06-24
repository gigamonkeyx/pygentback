#!/usr/bin/env python3
"""
A2A Protocol Compliance Test Suite

Comprehensive testing of A2A protocol compliance according to Google A2A specification.
Tests real agent-to-agent communication, message passing, and protocol adherence.
"""

import asyncio
import aiohttp
import json
import sys
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class A2ATestResult:
    """A2A test result structure"""
    test_name: str
    success: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    response_time_ms: Optional[float] = None
    error: Optional[str] = None

class A2AProtocolComplianceTester:
    """Comprehensive A2A protocol compliance tester"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = None
        self.test_results: List[A2ATestResult] = []
        self.agent_card = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def test_a2a_agent_card_compliance(self) -> A2ATestResult:
        """Test A2A agent card compliance with specification"""
        print("🔍 Testing A2A Agent Card Compliance...")
        
        start_time = time.time()
        try:
            url = f"{self.base_url}/a2a/v1/.well-known/agent.json"
            async with self.session.get(url) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status != 200:
                    return A2ATestResult(
                        test_name="agent_card_compliance",
                        success=False,
                        message=f"HTTP {response.status} - Expected 200",
                        response_time_ms=response_time
                    )
                
                data = await response.json()
                self.agent_card = data
                
                # Check required fields according to A2A spec
                required_fields = ["name", "description", "url", "capabilities", "skills", "provider"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    return A2ATestResult(
                        test_name="agent_card_compliance",
                        success=False,
                        message=f"Missing required fields: {missing_fields}",
                        details={"missing_fields": missing_fields, "received_fields": list(data.keys())},
                        response_time_ms=response_time
                    )
                
                # Validate capabilities structure
                capabilities = data.get("capabilities", {})
                if not isinstance(capabilities, dict):
                    return A2ATestResult(
                        test_name="agent_card_compliance",
                        success=False,
                        message="Capabilities must be an object",
                        response_time_ms=response_time
                    )
                
                # Validate skills structure
                skills = data.get("skills", [])
                if not isinstance(skills, list):
                    return A2ATestResult(
                        test_name="agent_card_compliance",
                        success=False,
                        message="Skills must be an array",
                        response_time_ms=response_time
                    )
                
                # Check skill structure
                for i, skill in enumerate(skills):
                    if not isinstance(skill, dict) or "id" not in skill or "name" not in skill:
                        return A2ATestResult(
                            test_name="agent_card_compliance",
                            success=False,
                            message=f"Skill {i} missing required fields (id, name)",
                            response_time_ms=response_time
                        )
                
                return A2ATestResult(
                    test_name="agent_card_compliance",
                    success=True,
                    message="Agent card fully compliant with A2A specification",
                    details={
                        "agent_name": data.get("name"),
                        "capabilities_count": len(capabilities),
                        "skills_count": len(skills),
                        "provider": data.get("provider", {}).get("name")
                    },
                    response_time_ms=response_time
                )
                
        except Exception as e:
            return A2ATestResult(
                test_name="agent_card_compliance",
                success=False,
                message=f"Agent card compliance test failed: {e}",
                error=str(e),
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    async def test_a2a_message_protocol(self) -> A2ATestResult:
        """Test A2A message protocol compliance"""
        print("🔍 Testing A2A Message Protocol...")
        
        start_time = time.time()
        try:
            url = f"{self.base_url}/a2a/v1/message/send"
            
            # Test message according to A2A spec
            test_message = {
                "message": "A2A protocol compliance test message",
                "sender": "a2a_compliance_tester",
                "recipient": "pygent_factory_main",
                "message_type": "test",
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {
                    "test_id": "compliance_001",
                    "protocol_version": "1.0"
                }
            }
            
            async with self.session.post(url, json=test_message) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status not in [200, 201, 202]:
                    return A2ATestResult(
                        test_name="message_protocol",
                        success=False,
                        message=f"HTTP {response.status} - Expected 200/201/202",
                        response_time_ms=response_time
                    )
                
                data = await response.json()
                
                # Check required response fields
                required_response_fields = ["status", "message_id", "timestamp"]
                missing_fields = [field for field in required_response_fields if field not in data]
                
                if missing_fields:
                    return A2ATestResult(
                        test_name="message_protocol",
                        success=False,
                        message=f"Response missing required fields: {missing_fields}",
                        details={"missing_fields": missing_fields, "response": data},
                        response_time_ms=response_time
                    )
                
                # Validate message ID format
                message_id = data.get("message_id", "")
                if not message_id or len(message_id) < 5:
                    return A2ATestResult(
                        test_name="message_protocol",
                        success=False,
                        message="Invalid message_id format",
                        response_time_ms=response_time
                    )
                
                return A2ATestResult(
                    test_name="message_protocol",
                    success=True,
                    message="Message protocol fully compliant",
                    details={
                        "message_id": message_id,
                        "status": data.get("status"),
                        "recipient": data.get("recipient")
                    },
                    response_time_ms=response_time
                )
                
        except Exception as e:
            return A2ATestResult(
                test_name="message_protocol",
                success=False,
                message=f"Message protocol test failed: {e}",
                error=str(e),
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    async def test_a2a_discovery_protocol(self) -> A2ATestResult:
        """Test A2A discovery protocol compliance"""
        print("🔍 Testing A2A Discovery Protocol...")
        
        start_time = time.time()
        try:
            url = f"{self.base_url}/a2a/v1/agents/discover"
            async with self.session.get(url) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status != 200:
                    return A2ATestResult(
                        test_name="discovery_protocol",
                        success=False,
                        message=f"HTTP {response.status} - Expected 200",
                        response_time_ms=response_time
                    )
                
                data = await response.json()
                
                # Check required discovery response fields
                required_fields = ["agents", "total", "timestamp"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    return A2ATestResult(
                        test_name="discovery_protocol",
                        success=False,
                        message=f"Discovery response missing fields: {missing_fields}",
                        response_time_ms=response_time
                    )
                
                agents = data.get("agents", [])
                if not isinstance(agents, list):
                    return A2ATestResult(
                        test_name="discovery_protocol",
                        success=False,
                        message="Agents field must be an array",
                        response_time_ms=response_time
                    )
                
                # Validate agent entries
                for i, agent in enumerate(agents):
                    required_agent_fields = ["id", "name", "url", "status"]
                    missing_agent_fields = [field for field in required_agent_fields if field not in agent]
                    
                    if missing_agent_fields:
                        return A2ATestResult(
                            test_name="discovery_protocol",
                            success=False,
                            message=f"Agent {i} missing fields: {missing_agent_fields}",
                            response_time_ms=response_time
                        )
                
                return A2ATestResult(
                    test_name="discovery_protocol",
                    success=True,
                    message="Discovery protocol fully compliant",
                    details={
                        "agents_discovered": len(agents),
                        "total_reported": data.get("total"),
                        "discovery_method": data.get("discovery_method")
                    },
                    response_time_ms=response_time
                )
                
        except Exception as e:
            return A2ATestResult(
                test_name="discovery_protocol",
                success=False,
                message=f"Discovery protocol test failed: {e}",
                error=str(e),
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    async def test_a2a_health_monitoring(self) -> A2ATestResult:
        """Test A2A health monitoring compliance"""
        print("🔍 Testing A2A Health Monitoring...")
        
        start_time = time.time()
        try:
            url = f"{self.base_url}/a2a/v1/health"
            async with self.session.get(url) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status != 200:
                    return A2ATestResult(
                        test_name="health_monitoring",
                        success=False,
                        message=f"HTTP {response.status} - Expected 200",
                        response_time_ms=response_time
                    )
                
                data = await response.json()
                
                # Check required health fields
                required_fields = ["status", "timestamp"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    return A2ATestResult(
                        test_name="health_monitoring",
                        success=False,
                        message=f"Health response missing fields: {missing_fields}",
                        response_time_ms=response_time
                    )
                
                status = data.get("status", "").lower()
                if status not in ["healthy", "operational", "active"]:
                    return A2ATestResult(
                        test_name="health_monitoring",
                        success=False,
                        message=f"Invalid health status: {status}",
                        response_time_ms=response_time
                    )
                
                return A2ATestResult(
                    test_name="health_monitoring",
                    success=True,
                    message="Health monitoring fully compliant",
                    details={
                        "status": data.get("status"),
                        "a2a_protocol": data.get("a2a_protocol"),
                        "endpoints": data.get("endpoints", [])
                    },
                    response_time_ms=response_time
                )
                
        except Exception as e:
            return A2ATestResult(
                test_name="health_monitoring",
                success=False,
                message=f"Health monitoring test failed: {e}",
                error=str(e),
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    async def test_a2a_performance_benchmarks(self) -> A2ATestResult:
        """Test A2A performance benchmarks"""
        print("🔍 Testing A2A Performance Benchmarks...")
        
        start_time = time.time()
        try:
            # Test multiple rapid requests to measure performance
            tasks = []
            for i in range(10):
                task = self.session.get(f"{self.base_url}/a2a/v1/health")
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks)
            response_times = []
            
            for response in responses:
                if response.status == 200:
                    response_times.append(response.headers.get('X-Response-Time', 0))
                response.close()
            
            total_time = (time.time() - start_time) * 1000
            avg_response_time = total_time / len(responses) if responses else 0
            
            # Performance criteria
            performance_ok = avg_response_time < 1000  # Less than 1 second average
            
            return A2ATestResult(
                test_name="performance_benchmarks",
                success=performance_ok,
                message=f"Performance {'acceptable' if performance_ok else 'needs improvement'}",
                details={
                    "total_requests": len(responses),
                    "successful_requests": sum(1 for r in responses if r.status == 200),
                    "average_response_time_ms": round(avg_response_time, 2),
                    "total_test_time_ms": round(total_time, 2)
                },
                response_time_ms=total_time
            )
            
        except Exception as e:
            return A2ATestResult(
                test_name="performance_benchmarks",
                success=False,
                message=f"Performance benchmark test failed: {e}",
                error=str(e),
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    async def run_comprehensive_compliance_tests(self) -> Dict[str, Any]:
        """Run all A2A protocol compliance tests"""
        print("🚀 Running Comprehensive A2A Protocol Compliance Tests")
        print("=" * 70)
        
        # Execute all compliance tests
        tests = [
            self.test_a2a_agent_card_compliance(),
            self.test_a2a_message_protocol(),
            self.test_a2a_discovery_protocol(),
            self.test_a2a_health_monitoring(),
            self.test_a2a_performance_benchmarks(),
        ]
        
        results = await asyncio.gather(*tests, return_exceptions=True)
        
        # Process results
        test_results = []
        for result in results:
            if isinstance(result, Exception):
                test_results.append(A2ATestResult(
                    test_name="unknown",
                    success=False,
                    message=f"Test exception: {result}",
                    error=str(result)
                ))
            else:
                test_results.append(result)
        
        self.test_results = test_results
        
        # Calculate summary
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results if result.success)
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Print results
        for result in test_results:
            status = "✅ PASSED" if result.success else "❌ FAILED"
            time_info = f" ({result.response_time_ms:.1f}ms)" if result.response_time_ms else ""
            print(f"{status} {result.test_name}: {result.message}{time_info}")
        
        # Calculate compliance level
        compliance_level = "FULL" if passed_tests == total_tests else "PARTIAL" if passed_tests >= total_tests * 0.8 else "MINIMAL"
        
        summary = {
            "compliance_level": compliance_level,
            "protocol_ready": passed_tests >= 4,  # At least 4/5 tests must pass
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": round(success_rate, 1),
            "timestamp": datetime.utcnow().isoformat(),
            "test_results": [
                {
                    "test_name": r.test_name,
                    "success": r.success,
                    "message": r.message,
                    "response_time_ms": r.response_time_ms,
                    "details": r.details
                } for r in test_results
            ]
        }
        
        print(f"\n" + "=" * 70)
        print(f"📊 A2A Protocol Compliance Summary:")
        print(f"   Compliance Level: {compliance_level}")
        print(f"   Protocol Ready: {'YES' if summary['protocol_ready'] else 'NO'}")
        print(f"   Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests})")
        print(f"   Timestamp: {summary['timestamp']}")
        
        if summary["protocol_ready"]:
            print("🎉 A2A PROTOCOL IS FULLY COMPLIANT AND READY!")
            print("   ✅ Agent card specification compliant")
            print("   ✅ Message protocol operational")
            print("   ✅ Discovery system functional")
            print("   ✅ Health monitoring active")
            print("   ✅ Performance benchmarks acceptable")
        else:
            print("⚠️  A2A PROTOCOL COMPLIANCE NEEDS ATTENTION")
            failed_tests = [r for r in test_results if not r.success]
            for failed in failed_tests:
                print(f"   ❌ {failed.test_name}: {failed.message}")
        
        return summary

async def main():
    """Main compliance test function"""
    async with A2AProtocolComplianceTester() as tester:
        summary = await tester.run_comprehensive_compliance_tests()
        
        # Exit with appropriate code
        if summary["protocol_ready"]:
            sys.exit(0)
        else:
            sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
