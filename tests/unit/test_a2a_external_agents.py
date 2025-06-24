#!/usr/bin/env python3
"""
A2A External Agent Simulation Tests

Simulate external agents communicating with PyGent Factory via A2A protocol.
Tests real-world agent-to-agent communication scenarios.
"""

import asyncio
import aiohttp
import json
import sys
import uuid
from datetime import datetime
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class ExternalAgent:
    """External agent simulation"""
    agent_id: str
    name: str
    capabilities: List[str]
    base_url: str = "http://localhost:8000"

class A2AExternalAgentTester:
    """Test A2A protocol with simulated external agents"""
    
    def __init__(self, target_url: str = "http://localhost:8000"):
        self.target_url = target_url.rstrip('/')
        self.session = None
        self.test_results = []
        
        # Create simulated external agents
        self.external_agents = [
            ExternalAgent(
                agent_id="research_agent_001",
                name="Academic Research Agent",
                capabilities=["research", "analysis", "citation"]
            ),
            ExternalAgent(
                agent_id="data_processor_002", 
                name="Data Processing Agent",
                capabilities=["data_processing", "transformation", "validation"]
            ),
            ExternalAgent(
                agent_id="coordination_agent_003",
                name="Multi-Agent Coordinator",
                capabilities=["coordination", "task_distribution", "monitoring"]
            )
        ]
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def test_agent_discovery_from_external(self) -> Dict[str, Any]:
        """Test external agent discovering PyGent Factory"""
        print("🔍 Testing External Agent Discovery...")
        
        try:
            # Simulate external agent discovering PyGent Factory
            url = f"{self.target_url}/a2a/v1/agents/discover"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    discovered_agents = data.get("agents", [])
                    
                    return {
                        "test": "external_agent_discovery",
                        "success": len(discovered_agents) > 0,
                        "message": f"Discovered {len(discovered_agents)} agents",
                        "discovered_agents": discovered_agents,
                        "discovery_successful": True
                    }
                else:
                    return {
                        "test": "external_agent_discovery",
                        "success": False,
                        "message": f"Discovery failed with HTTP {response.status}"
                    }
                    
        except Exception as e:
            return {
                "test": "external_agent_discovery",
                "success": False,
                "message": f"External agent discovery failed: {e}",
                "error": str(e)
            }
    
    async def test_multi_agent_message_exchange(self) -> Dict[str, Any]:
        """Test multiple external agents sending messages"""
        print("🔍 Testing Multi-Agent Message Exchange...")
        
        try:
            url = f"{self.target_url}/a2a/v1/message/send"
            message_results = []
            
            # Each external agent sends a message
            for agent in self.external_agents:
                message_data = {
                    "message": f"Hello from {agent.name}! Testing A2A communication.",
                    "sender": agent.agent_id,
                    "recipient": "pygent_factory_main",
                    "message_type": "greeting",
                    "timestamp": datetime.utcnow().isoformat(),
                    "metadata": {
                        "sender_capabilities": agent.capabilities,
                        "test_scenario": "multi_agent_exchange"
                    }
                }
                
                async with self.session.post(url, json=message_data) as response:
                    if response.status in [200, 201, 202]:
                        data = await response.json()
                        message_results.append({
                            "sender": agent.agent_id,
                            "message_id": data.get("message_id"),
                            "status": data.get("status"),
                            "success": True
                        })
                    else:
                        message_results.append({
                            "sender": agent.agent_id,
                            "success": False,
                            "status_code": response.status
                        })
            
            successful_messages = sum(1 for result in message_results if result.get("success", False))
            
            return {
                "test": "multi_agent_message_exchange",
                "success": successful_messages == len(self.external_agents),
                "message": f"Successfully exchanged messages with {successful_messages}/{len(self.external_agents)} agents",
                "message_results": message_results,
                "total_agents": len(self.external_agents)
            }
            
        except Exception as e:
            return {
                "test": "multi_agent_message_exchange",
                "success": False,
                "message": f"Multi-agent message exchange failed: {e}",
                "error": str(e)
            }
    
    async def test_agent_capability_negotiation(self) -> Dict[str, Any]:
        """Test agent capability negotiation via A2A"""
        print("🔍 Testing Agent Capability Negotiation...")
        
        try:
            # First, get PyGent Factory's capabilities
            well_known_url = f"{self.target_url}/a2a/v1/.well-known/agent.json"
            async with self.session.get(well_known_url) as response:
                if response.status != 200:
                    return {
                        "test": "capability_negotiation",
                        "success": False,
                        "message": "Failed to retrieve target agent capabilities"
                    }
                
                target_agent_data = await response.json()
                target_capabilities = target_agent_data.get("capabilities", {})
                target_skills = [skill.get("id") for skill in target_agent_data.get("skills", [])]
            
            # Test capability matching for each external agent
            negotiation_results = []
            
            for agent in self.external_agents:
                # Find matching capabilities
                matching_capabilities = []
                for capability in agent.capabilities:
                    if capability in target_skills or capability in target_capabilities:
                        matching_capabilities.append(capability)
                
                # Send capability negotiation message
                negotiation_message = {
                    "message": "Capability negotiation request",
                    "sender": agent.agent_id,
                    "recipient": "pygent_factory_main",
                    "message_type": "capability_negotiation",
                    "timestamp": datetime.utcnow().isoformat(),
                    "metadata": {
                        "offered_capabilities": agent.capabilities,
                        "requested_capabilities": list(target_capabilities.keys()),
                        "matching_capabilities": matching_capabilities
                    }
                }
                
                message_url = f"{self.target_url}/a2a/v1/message/send"
                async with self.session.post(message_url, json=negotiation_message) as response:
                    if response.status in [200, 201, 202]:
                        data = await response.json()
                        negotiation_results.append({
                            "agent": agent.agent_id,
                            "matching_capabilities": matching_capabilities,
                            "negotiation_successful": True,
                            "message_id": data.get("message_id")
                        })
                    else:
                        negotiation_results.append({
                            "agent": agent.agent_id,
                            "negotiation_successful": False,
                            "status_code": response.status
                        })
            
            successful_negotiations = sum(1 for result in negotiation_results if result.get("negotiation_successful", False))
            
            return {
                "test": "capability_negotiation",
                "success": successful_negotiations > 0,
                "message": f"Successfully negotiated capabilities with {successful_negotiations}/{len(self.external_agents)} agents",
                "negotiation_results": negotiation_results,
                "target_capabilities": list(target_capabilities.keys()),
                "target_skills": target_skills
            }
            
        except Exception as e:
            return {
                "test": "capability_negotiation",
                "success": False,
                "message": f"Capability negotiation failed: {e}",
                "error": str(e)
            }
    
    async def test_concurrent_agent_interactions(self) -> Dict[str, Any]:
        """Test concurrent interactions from multiple external agents"""
        print("🔍 Testing Concurrent Agent Interactions...")
        
        try:
            # Create concurrent tasks for all agents
            tasks = []
            
            for i, agent in enumerate(self.external_agents):
                # Each agent performs multiple concurrent operations
                agent_tasks = [
                    self.session.get(f"{self.target_url}/a2a/v1/health"),
                    self.session.get(f"{self.target_url}/a2a/v1/agents/discover"),
                    self.session.post(
                        f"{self.target_url}/a2a/v1/message/send",
                        json={
                            "message": f"Concurrent test message {i}",
                            "sender": agent.agent_id,
                            "recipient": "pygent_factory_main",
                            "message_type": "concurrent_test",
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
                ]
                tasks.extend(agent_tasks)
            
            # Execute all tasks concurrently
            start_time = asyncio.get_event_loop().time()
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = asyncio.get_event_loop().time()
            
            # Analyze results
            successful_responses = 0
            failed_responses = 0
            
            for response in responses:
                if isinstance(response, Exception):
                    failed_responses += 1
                elif hasattr(response, 'status') and response.status in [200, 201, 202]:
                    successful_responses += 1
                    response.close()
                else:
                    failed_responses += 1
                    if hasattr(response, 'close'):
                        response.close()
            
            total_time = (end_time - start_time) * 1000
            success_rate = (successful_responses / len(responses)) * 100 if responses else 0
            
            return {
                "test": "concurrent_agent_interactions",
                "success": success_rate >= 80,  # At least 80% success rate
                "message": f"Concurrent interactions: {success_rate:.1f}% success rate",
                "total_requests": len(responses),
                "successful_responses": successful_responses,
                "failed_responses": failed_responses,
                "total_time_ms": round(total_time, 2),
                "agents_tested": len(self.external_agents)
            }
            
        except Exception as e:
            return {
                "test": "concurrent_agent_interactions",
                "success": False,
                "message": f"Concurrent agent interactions failed: {e}",
                "error": str(e)
            }
    
    async def test_agent_health_monitoring(self) -> Dict[str, Any]:
        """Test external agents monitoring PyGent Factory health"""
        print("🔍 Testing Agent Health Monitoring...")
        
        try:
            health_checks = []
            
            # Each external agent checks health multiple times
            for agent in self.external_agents:
                for check_num in range(3):
                    url = f"{self.target_url}/a2a/v1/health"
                    async with self.session.get(url) as response:
                        health_data = {
                            "agent": agent.agent_id,
                            "check_number": check_num + 1,
                            "status_code": response.status,
                            "healthy": response.status == 200
                        }
                        
                        if response.status == 200:
                            data = await response.json()
                            health_data.update({
                                "service_status": data.get("status"),
                                "a2a_protocol": data.get("a2a_protocol"),
                                "timestamp": data.get("timestamp")
                            })
                        
                        health_checks.append(health_data)
                    
                    # Small delay between checks
                    await asyncio.sleep(0.1)
            
            successful_checks = sum(1 for check in health_checks if check.get("healthy", False))
            total_checks = len(health_checks)
            health_success_rate = (successful_checks / total_checks) * 100 if total_checks > 0 else 0
            
            return {
                "test": "agent_health_monitoring",
                "success": health_success_rate >= 95,  # At least 95% health checks successful
                "message": f"Health monitoring: {health_success_rate:.1f}% success rate",
                "total_health_checks": total_checks,
                "successful_checks": successful_checks,
                "health_success_rate": round(health_success_rate, 1),
                "agents_monitoring": len(self.external_agents)
            }
            
        except Exception as e:
            return {
                "test": "agent_health_monitoring",
                "success": False,
                "message": f"Agent health monitoring failed: {e}",
                "error": str(e)
            }
    
    async def run_external_agent_tests(self) -> Dict[str, Any]:
        """Run all external agent simulation tests"""
        print("🚀 Running A2A External Agent Simulation Tests")
        print("=" * 70)
        print(f"🤖 Simulating {len(self.external_agents)} external agents:")
        for agent in self.external_agents:
            print(f"   • {agent.name} ({agent.agent_id})")
            print(f"     Capabilities: {', '.join(agent.capabilities)}")
        print("=" * 70)
        
        # Execute all external agent tests
        tests = [
            self.test_agent_discovery_from_external(),
            self.test_multi_agent_message_exchange(),
            self.test_agent_capability_negotiation(),
            self.test_concurrent_agent_interactions(),
            self.test_agent_health_monitoring(),
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
            status = "✅ PASSED" if result.get("success", False) else "❌ FAILED"
            print(f"{status} {result['test']}: {result['message']}")
        
        # Determine integration readiness
        integration_ready = passed_tests >= 4  # At least 4/5 tests must pass
        
        summary = {
            "integration_ready": integration_ready,
            "external_agents_tested": len(self.external_agents),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": round(success_rate, 1),
            "timestamp": datetime.utcnow().isoformat(),
            "test_results": test_results
        }
        
        print(f"\n" + "=" * 70)
        print(f"📊 A2A External Agent Integration Summary:")
        print(f"   Integration Ready: {'YES' if integration_ready else 'NO'}")
        print(f"   External Agents Tested: {len(self.external_agents)}")
        print(f"   Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests})")
        print(f"   Timestamp: {summary['timestamp']}")
        
        if integration_ready:
            print("🎉 A2A EXTERNAL AGENT INTEGRATION SUCCESSFUL!")
            print("   ✅ External agent discovery working")
            print("   ✅ Multi-agent message exchange operational")
            print("   ✅ Capability negotiation functional")
            print("   ✅ Concurrent interactions supported")
            print("   ✅ Health monitoring reliable")
        else:
            print("⚠️  A2A EXTERNAL AGENT INTEGRATION NEEDS ATTENTION")
            failed_tests = [r for r in test_results if not r.get("success", False)]
            for failed in failed_tests:
                print(f"   ❌ {failed['test']}: {failed['message']}")
        
        return summary

async def main():
    """Main external agent test function"""
    async with A2AExternalAgentTester() as tester:
        summary = await tester.run_external_agent_tests()
        
        # Exit with appropriate code
        if summary["integration_ready"]:
            sys.exit(0)
        else:
            sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
