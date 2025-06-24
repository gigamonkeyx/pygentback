#!/usr/bin/env python3
"""
Test Multi-Agent A2A Communication

Test communication between multiple A2A agents via the orchestration server.
"""

import asyncio
import aiohttp
import json
import time

async def test_multi_agent_communication():
    """Test multi-agent A2A communication"""
    
    async with aiohttp.ClientSession() as session:
        print("🚀 Multi-Agent A2A Communication Test")
        print("=" * 50)
        
        # Step 1: Trigger agent discovery
        print("\n🔄 Step 1: Triggering agent discovery...")
        try:
            async with session.post("http://127.0.0.1:8005/v1/a2a/discover") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Discovery completed: {data}")
                else:
                    print(f"❌ Discovery failed: {response.status}")
        except Exception as e:
            print(f"❌ Discovery error: {e}")
        
        # Step 2: List discovered agents
        print("\n🔍 Step 2: Listing discovered A2A agents...")
        try:
            async with session.get("http://127.0.0.1:8005/v1/a2a/agents") as response:
                if response.status == 200:
                    data = await response.json()
                    agents = data.get('a2a_agents', {})
                    total = data.get('total_agents', 0)
                    
                    print(f"✅ Found {total} A2A agents:")
                    for agent_id, agent_info in agents.items():
                        print(f"  - {agent_info['name']} ({agent_id})")
                        print(f"    URL: {agent_info['url']}")
                        print(f"    Description: {agent_info['description']}")
                    
                    if total == 0:
                        print("❌ No agents found - cannot test communication")
                        return
                    
                    # Step 3: Test communication with each agent
                    print(f"\n💬 Step 3: Testing communication with {total} agents...")
                    
                    for i, (agent_id, agent_info) in enumerate(agents.items(), 1):
                        print(f"\n📤 Test {i}: Sending message to {agent_info['name']}...")
                        
                        message_payload = {
                            "agent_id": agent_id,
                            "message": f"Hello {agent_info['name']}! This is a test message from the orchestration server. Please analyze this text: 'PyGent Factory is an amazing multi-agent orchestration platform that enables seamless communication between AI agents using the A2A protocol.'",
                            "context_id": f"test-context-{i}"
                        }
                        
                        start_time = time.time()
                        try:
                            async with session.post(
                                "http://127.0.0.1:8005/v1/a2a/message",
                                json=message_payload
                            ) as msg_response:
                                duration = time.time() - start_time
                                
                                if msg_response.status == 200:
                                    result = await msg_response.json()
                                    print(f"✅ Message sent successfully ({duration*1000:.1f}ms)")
                                    print(f"   Response: {result.get('status', 'No status')}")
                                    
                                    # Check if there's a response in the result
                                    if 'response' in result:
                                        response_data = result['response']
                                        if 'artifacts' in response_data:
                                            artifacts = response_data['artifacts']
                                            if artifacts:
                                                print(f"   Agent processed the message and returned {len(artifacts)} artifacts")
                                                for artifact in artifacts[:1]:  # Show first artifact
                                                    if 'parts' in artifact:
                                                        for part in artifact['parts'][:1]:  # Show first part
                                                            if part.get('kind') == 'text':
                                                                text = part.get('text', '')
                                                                preview = text[:200] + '...' if len(text) > 200 else text
                                                                print(f"   Preview: {preview}")
                                else:
                                    error_text = await msg_response.text()
                                    print(f"❌ Message failed ({duration*1000:.1f}ms): {error_text}")
                                    
                        except Exception as e:
                            duration = time.time() - start_time
                            print(f"❌ Message error ({duration*1000:.1f}ms): {e}")
                    
                    # Step 4: Test direct agent communication
                    print(f"\n🔗 Step 4: Testing direct agent communication...")
                    
                    if total >= 2:
                        agent_ids = list(agents.keys())
                        agent1_id = agent_ids[0]
                        agent2_id = agent_ids[1]
                        agent1_name = agents[agent1_id]['name']
                        agent2_name = agents[agent2_id]['name']
                        
                        print(f"Testing communication from {agent1_name} to {agent2_name}...")
                        
                        # This would require implementing agent-to-agent communication
                        # For now, we'll test via the orchestration server
                        cross_message = {
                            "agent_id": agent2_id,
                            "message": f"Cross-agent message: {agent1_name} says hello to {agent2_name}! Please process this collaborative task.",
                            "context_id": "cross-agent-test"
                        }
                        
                        try:
                            async with session.post(
                                "http://127.0.0.1:8005/v1/a2a/message",
                                json=cross_message
                            ) as cross_response:
                                if cross_response.status == 200:
                                    cross_result = await cross_response.json()
                                    print(f"✅ Cross-agent communication successful")
                                else:
                                    print(f"❌ Cross-agent communication failed: {cross_response.status}")
                        except Exception as e:
                            print(f"❌ Cross-agent communication error: {e}")
                    else:
                        print("⚠️ Need at least 2 agents for cross-agent communication test")
                    
                else:
                    print(f"❌ Failed to list agents: {response.status}")
        except Exception as e:
            print(f"❌ Agent listing error: {e}")
        
        print("\n" + "=" * 50)
        print("🏁 Multi-Agent A2A Communication Test Complete")


if __name__ == "__main__":
    asyncio.run(test_multi_agent_communication())
