#!/usr/bin/env python3
"""
A2A Production Health Check Script

Comprehensive health check for A2A protocol in production deployment.
"""

import asyncio
import aiohttp
import json
import sys
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

class A2AHealthChecker:
    """A2A protocol health checker"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip('/')
        self.session = None
        self.results = []
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def check_endpoint(self, endpoint: str, method: str = "GET", 
                           data: Optional[Dict] = None, expected_status: int = 200) -> Dict[str, Any]:
        """Check a single endpoint"""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            if method.upper() == "GET":
                async with self.session.get(url) as response:
                    status = response.status
                    content = await response.text()
            elif method.upper() == "POST":
                async with self.session.post(url, json=data) as response:
                    status = response.status
                    content = await response.text()
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response_time = time.time() - start_time
            
            # Try to parse JSON
            try:
                json_content = json.loads(content)
            except:
                json_content = None
            
            result = {
                "endpoint": endpoint,
                "method": method,
                "status": status,
                "expected_status": expected_status,
                "success": status == expected_status,
                "response_time": round(response_time * 1000, 2),  # ms
                "content_length": len(content),
                "json_content": json_content,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {
                "endpoint": endpoint,
                "method": method,
                "status": 0,
                "expected_status": expected_status,
                "success": False,
                "error": str(e),
                "response_time": round((time.time() - start_time) * 1000, 2),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def check_a2a_health(self) -> Dict[str, Any]:
        """Check A2A health endpoint"""
        return await self.check_endpoint("/api/a2a/v1/health")
    
    async def check_a2a_discovery(self) -> Dict[str, Any]:
        """Check A2A agent discovery"""
        return await self.check_endpoint("/api/a2a/v1/agents/discover")
    
    async def check_well_known_agent(self) -> Dict[str, Any]:
        """Check well-known agent endpoint"""
        return await self.check_endpoint("/.well-known/agent.json")
    
    async def check_a2a_message_send(self) -> Dict[str, Any]:
        """Check A2A message sending"""
        test_message = {
            "method": "ping",
            "params": {"message": "health_check"},
            "id": "health_check_001"
        }
        return await self.check_endpoint("/api/a2a/v1/message/send", "POST", test_message)
    
    async def check_main_api_health(self) -> Dict[str, Any]:
        """Check main API health"""
        return await self.check_endpoint("/health")
    
    async def check_api_docs(self) -> Dict[str, Any]:
        """Check API documentation"""
        return await self.check_endpoint("/docs", expected_status=200)
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        print("🏥 Running A2A Production Health Checks")
        print("=" * 60)
        
        checks = [
            ("Main API Health", self.check_main_api_health()),
            ("API Documentation", self.check_api_docs()),
            ("A2A Health", self.check_a2a_health()),
            ("A2A Discovery", self.check_a2a_discovery()),
            ("Well-known Agent", self.check_well_known_agent()),
            ("A2A Message Send", self.check_a2a_message_send()),
        ]
        
        results = {}
        total_checks = len(checks)
        passed_checks = 0
        
        for check_name, check_coro in checks:
            print(f"\n🔍 Checking: {check_name}")
            result = await check_coro
            results[check_name] = result
            
            if result["success"]:
                print(f"✅ {check_name}: PASSED ({result['response_time']}ms)")
                passed_checks += 1
            else:
                error_msg = result.get("error", f"Status {result['status']}")
                print(f"❌ {check_name}: FAILED - {error_msg}")
        
        # Summary
        success_rate = (passed_checks / total_checks) * 100
        overall_status = "HEALTHY" if passed_checks == total_checks else "DEGRADED" if passed_checks >= total_checks * 0.7 else "UNHEALTHY"
        
        summary = {
            "overall_status": overall_status,
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": total_checks - passed_checks,
            "success_rate": round(success_rate, 1),
            "timestamp": datetime.utcnow().isoformat(),
            "checks": results
        }
        
        print(f"\n" + "=" * 60)
        print(f"📊 Health Check Summary:")
        print(f"   Overall Status: {overall_status}")
        print(f"   Success Rate: {success_rate:.1f}% ({passed_checks}/{total_checks})")
        print(f"   Timestamp: {summary['timestamp']}")
        
        if overall_status == "HEALTHY":
            print("🎉 All A2A systems are healthy and operational!")
        elif overall_status == "DEGRADED":
            print("⚠️  Some A2A systems have issues but core functionality is working.")
        else:
            print("💥 Critical A2A systems are failing!")
        
        return summary

async def main():
    """Main health check function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="A2A Production Health Check")
    parser.add_argument("--url", default="http://localhost:8080", help="Base URL for health checks")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout in seconds")
    
    args = parser.parse_args()
    
    try:
        async with A2AHealthChecker(args.url) as checker:
            # Set timeout
            summary = await asyncio.wait_for(
                checker.run_all_checks(),
                timeout=args.timeout
            )
            
            # Save results if requested
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(summary, f, indent=2)
                print(f"\n💾 Results saved to: {args.output}")
            
            # Exit with appropriate code
            if summary["overall_status"] == "HEALTHY":
                sys.exit(0)
            elif summary["overall_status"] == "DEGRADED":
                sys.exit(1)
            else:
                sys.exit(2)
                
    except asyncio.TimeoutError:
        print(f"❌ Health check timed out after {args.timeout} seconds")
        sys.exit(3)
    except Exception as e:
        print(f"❌ Health check failed with error: {e}")
        sys.exit(4)

if __name__ == "__main__":
    asyncio.run(main())
