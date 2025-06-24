#!/usr/bin/env python3
"""
PyGent Factory MCP Ecosystem Health Monitor

Comprehensive health monitoring and status validation for the complete MCP ecosystem.
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class ServerStatus:
    """Server status information"""
    name: str
    port: int
    url: str
    status: str
    response_time_ms: float
    details: Dict[str, Any]
    error: Optional[str] = None

class EcosystemHealthMonitor:
    """Monitor health of the complete PyGent Factory MCP ecosystem"""
    
    def __init__(self):
        self.servers = {
            "document_processing": {
                "name": "Document Processing MCP Server",
                "port": 8003,
                "url": "http://127.0.0.1:8003",
                "health_endpoint": "/health"
            },
            "vector_search": {
                "name": "Vector Search MCP Server", 
                "port": 8004,
                "url": "http://127.0.0.1:8004",
                "health_endpoint": "/health"
            },
            "agent_orchestration": {
                "name": "Agent Orchestration MCP Server",
                "port": 8005,
                "url": "http://127.0.0.1:8005", 
                "health_endpoint": "/health"
            },
            "a2a_mcp": {
                "name": "A2A MCP Server",
                "port": 8006,
                "url": "http://127.0.0.1:8006",
                "health_endpoint": "/health"
            },
            "simple_a2a_agent": {
                "name": "Simple A2A Agent",
                "port": 8007,
                "url": "http://127.0.0.1:8007",
                "health_endpoint": "/health"
            }
        }
        
        self.session = None
        self.results = []
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def check_server_health(self, server_key: str, server_config: Dict[str, Any]) -> ServerStatus:
        """Check health of a single server"""
        start_time = time.time()
        
        try:
            health_url = f"{server_config['url']}{server_config['health_endpoint']}"
            
            async with self.session.get(health_url) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    health_data = await response.json()
                    
                    return ServerStatus(
                        name=server_config['name'],
                        port=server_config['port'],
                        url=server_config['url'],
                        status="healthy",
                        response_time_ms=round(response_time, 2),
                        details=health_data
                    )
                else:
                    error_text = await response.text()
                    return ServerStatus(
                        name=server_config['name'],
                        port=server_config['port'],
                        url=server_config['url'],
                        status="unhealthy",
                        response_time_ms=round(response_time, 2),
                        details={},
                        error=f"HTTP {response.status}: {error_text}"
                    )
                    
        except asyncio.TimeoutError:
            response_time = (time.time() - start_time) * 1000
            return ServerStatus(
                name=server_config['name'],
                port=server_config['port'],
                url=server_config['url'],
                status="timeout",
                response_time_ms=round(response_time, 2),
                details={},
                error="Request timeout"
            )
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return ServerStatus(
                name=server_config['name'],
                port=server_config['port'],
                url=server_config['url'],
                status="error",
                response_time_ms=round(response_time, 2),
                details={},
                error=str(e)
            )
    
    async def check_ecosystem_health(self) -> Dict[str, Any]:
        """Check health of the entire ecosystem"""
        print("🏥 PyGent Factory MCP Ecosystem Health Check")
        print("=" * 60)
        
        # Check all servers concurrently
        health_tasks = []
        for server_key, server_config in self.servers.items():
            task = self.check_server_health(server_key, server_config)
            health_tasks.append((server_key, task))
        
        # Wait for all health checks to complete
        server_statuses = {}
        for server_key, task in health_tasks:
            server_statuses[server_key] = await task
        
        # Display results
        healthy_count = 0
        total_count = len(server_statuses)
        
        for server_key, status in server_statuses.items():
            status_icon = "✅" if status.status == "healthy" else "❌"
            print(f"{status_icon} {status.name}")
            print(f"    Port: {status.port}")
            print(f"    Status: {status.status}")
            print(f"    Response Time: {status.response_time_ms}ms")
            
            if status.status == "healthy":
                healthy_count += 1
                # Show key metrics if available
                if 'uptime_seconds' in status.details.get('details', {}):
                    uptime = status.details['details']['uptime_seconds']
                    print(f"    Uptime: {uptime:.1f}s")
                if 'performance' in status.details:
                    perf = status.details['performance']
                    if 'success_rate' in perf:
                        print(f"    Success Rate: {perf['success_rate']}%")
            else:
                print(f"    Error: {status.error}")
            
            print()
        
        # Overall ecosystem status
        ecosystem_health = "healthy" if healthy_count == total_count else "degraded"
        if healthy_count == 0:
            ecosystem_health = "critical"
        elif healthy_count < total_count * 0.5:
            ecosystem_health = "unhealthy"
        
        print("=" * 60)
        print(f"📊 Ecosystem Status: {ecosystem_health.upper()}")
        print(f"📈 Servers Online: {healthy_count}/{total_count}")
        
        if ecosystem_health == "healthy":
            print("🎉 All MCP servers are running and healthy!")
        elif ecosystem_health == "degraded":
            print("⚠️ Some MCP servers are experiencing issues")
        else:
            print("🚨 Critical issues detected in MCP ecosystem")
        
        return {
            "ecosystem_status": ecosystem_health,
            "servers_online": healthy_count,
            "total_servers": total_count,
            "server_statuses": {k: {
                "name": v.name,
                "port": v.port,
                "status": v.status,
                "response_time_ms": v.response_time_ms,
                "error": v.error
            } for k, v in server_statuses.items()},
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def check_inter_server_communication(self) -> Dict[str, Any]:
        """Test communication between MCP servers"""
        print("\n🔗 Testing Inter-Server Communication")
        print("=" * 60)
        
        communication_tests = []
        
        # Test 1: Agent Orchestration → A2A Discovery
        print("📡 Test 1: Agent Orchestration → A2A Agent Discovery")
        try:
            async with self.session.get("http://127.0.0.1:8005/v1/a2a/agents") as response:
                if response.status == 200:
                    data = await response.json()
                    agent_count = data.get('total_agents', 0)
                    print(f"✅ A2A Discovery: {agent_count} agents found")
                    communication_tests.append({"test": "a2a_discovery", "status": "pass", "details": f"{agent_count} agents"})
                else:
                    print(f"❌ A2A Discovery failed: HTTP {response.status}")
                    communication_tests.append({"test": "a2a_discovery", "status": "fail", "error": f"HTTP {response.status}"})
        except Exception as e:
            print(f"❌ A2A Discovery error: {e}")
            communication_tests.append({"test": "a2a_discovery", "status": "error", "error": str(e)})
        
        # Test 2: A2A Message Sending
        print("\n📤 Test 2: A2A Message Sending")
        try:
            # Get available agents first
            async with self.session.get("http://127.0.0.1:8005/v1/a2a/agents") as response:
                if response.status == 200:
                    agents_data = await response.json()
                    agents = agents_data.get('a2a_agents', {})
                    
                    if agents:
                        agent_id = list(agents.keys())[0]
                        agent_name = agents[agent_id]['name']
                        
                        message_payload = {
                            "agent_id": agent_id,
                            "message": "Ecosystem health check test message",
                            "context_id": "health-check-001"
                        }
                        
                        async with self.session.post(
                            "http://127.0.0.1:8005/v1/a2a/message",
                            json=message_payload
                        ) as msg_response:
                            if msg_response.status == 200:
                                print(f"✅ A2A Messaging: Message sent to {agent_name}")
                                communication_tests.append({"test": "a2a_messaging", "status": "pass", "details": f"Message to {agent_name}"})
                            else:
                                print(f"❌ A2A Messaging failed: HTTP {msg_response.status}")
                                communication_tests.append({"test": "a2a_messaging", "status": "fail", "error": f"HTTP {msg_response.status}"})
                    else:
                        print("⚠️ A2A Messaging: No agents available for testing")
                        communication_tests.append({"test": "a2a_messaging", "status": "skip", "reason": "No agents available"})
        except Exception as e:
            print(f"❌ A2A Messaging error: {e}")
            communication_tests.append({"test": "a2a_messaging", "status": "error", "error": str(e)})
        
        # Test 3: Document Processing Capability Check
        print("\n📄 Test 3: Document Processing Capability")
        try:
            async with self.session.get("http://127.0.0.1:8003/") as response:
                if response.status == 200:
                    data = await response.json()
                    capabilities = data.get('capabilities', [])
                    print(f"✅ Document Processing: {len(capabilities)} capabilities available")
                    communication_tests.append({"test": "document_processing", "status": "pass", "details": f"{len(capabilities)} capabilities"})
                else:
                    print(f"❌ Document Processing failed: HTTP {response.status}")
                    communication_tests.append({"test": "document_processing", "status": "fail", "error": f"HTTP {response.status}"})
        except Exception as e:
            print(f"❌ Document Processing error: {e}")
            communication_tests.append({"test": "document_processing", "status": "error", "error": str(e)})
        
        # Test 4: Vector Search Capability Check
        print("\n🔍 Test 4: Vector Search Capability")
        try:
            async with self.session.get("http://127.0.0.1:8004/") as response:
                if response.status == 200:
                    data = await response.json()
                    capabilities = data.get('capabilities', [])
                    print(f"✅ Vector Search: {len(capabilities)} capabilities available")
                    communication_tests.append({"test": "vector_search", "status": "pass", "details": f"{len(capabilities)} capabilities"})
                else:
                    print(f"❌ Vector Search failed: HTTP {response.status}")
                    communication_tests.append({"test": "vector_search", "status": "fail", "error": f"HTTP {response.status}"})
        except Exception as e:
            print(f"❌ Vector Search error: {e}")
            communication_tests.append({"test": "vector_search", "status": "error", "error": str(e)})
        
        # Summary
        passed_tests = len([t for t in communication_tests if t['status'] == 'pass'])
        total_tests = len(communication_tests)
        
        print("\n" + "=" * 60)
        print(f"📊 Communication Tests: {passed_tests}/{total_tests} passed")
        
        return {
            "tests_passed": passed_tests,
            "total_tests": total_tests,
            "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
            "test_results": communication_tests
        }


async def main():
    """Main health monitoring execution"""
    async with EcosystemHealthMonitor() as monitor:
        # Check ecosystem health
        health_results = await monitor.check_ecosystem_health()
        
        # Test inter-server communication
        communication_results = await monitor.check_inter_server_communication()
        
        # Combine results
        final_results = {
            "ecosystem_health": health_results,
            "inter_server_communication": communication_results,
            "overall_status": "healthy" if (
                health_results["ecosystem_status"] == "healthy" and 
                communication_results["success_rate"] >= 75
            ) else "degraded"
        }
        
        # Save results
        with open('ecosystem_health_report.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n📄 Health report saved to: ecosystem_health_report.json")
        
        # Return appropriate exit code
        return 0 if final_results["overall_status"] == "healthy" else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
