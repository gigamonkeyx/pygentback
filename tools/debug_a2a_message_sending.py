#!/usr/bin/env python3
"""
Debug A2A Message Sending

Debug the A2A message sending process.
"""

import asyncio
import aiohttp
import json

async def debug_message_sending():
    """Debug A2A message sending"""
    
    async with aiohttp.ClientSession() as session:
        print("🔍 Step 1: Get A2A agents from orchestration server...")
        
        # Get A2A agents
        try:
            async with session.get("http://127.0.0.1:8005/v1/a2a/agents") as response:
                print(f"Status: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    print(f"Response: {json.dumps(data, indent=2)}")
                    
                    a2a_agents = data.get('a2a_agents', {})
                    if a2a_agents:
                        agent_id = list(a2a_agents.keys())[0]
                        agent_name = a2a_agents[agent_id]['name']
                        print(f"\n🎯 Selected agent: {agent_name} (ID: {agent_id})")
                        
                        print("\n🔍 Step 2: Test message sending via orchestration server...")
                        
                        # Test message sending
                        message_payload = {
                            "agent_id": agent_id,
                            "message": "Debug test message from orchestration server",
                            "context_id": "debug-context-001"
                        }
                        
                        print(f"Message payload: {json.dumps(message_payload, indent=2)}")
                        
                        async with session.post(
                            "http://127.0.0.1:8005/v1/a2a/message",
                            json=message_payload
                        ) as msg_response:
                            print(f"Message Status: {msg_response.status}")
                            if msg_response.status == 200:
                                msg_data = await msg_response.json()
                                print(f"Message Response: {json.dumps(msg_data, indent=2)}")
                            else:
                                msg_text = await msg_response.text()
                                print(f"Message Error: {msg_text}")
                        
                        print("\n🔍 Step 3: Test direct A2A server message sending...")
                        
                        # Test direct A2A server message sending
                        direct_payload = {
                            "agent_id": agent_id,
                            "message": "Direct test message to A2A server",
                            "context_id": "debug-direct-001"
                        }
                        
                        async with session.post(
                            "http://127.0.0.1:8006/mcp/a2a/send_to_agent",
                            json=direct_payload
                        ) as direct_response:
                            print(f"Direct Status: {direct_response.status}")
                            if direct_response.status == 200:
                                direct_data = await direct_response.json()
                                print(f"Direct Response: {json.dumps(direct_data, indent=2)}")
                            else:
                                direct_text = await direct_response.text()
                                print(f"Direct Error: {direct_text}")
                        
                        print("\n🔍 Step 4: Check A2A server registered agents...")
                        
                        # Check A2A server agents
                        async with session.get("http://127.0.0.1:8006/mcp/a2a/agents") as agents_response:
                            print(f"A2A Agents Status: {agents_response.status}")
                            if agents_response.status == 200:
                                agents_data = await agents_response.json()
                                print(f"A2A Agents: {json.dumps(agents_data, indent=2)}")
                            else:
                                agents_text = await agents_response.text()
                                print(f"A2A Agents Error: {agents_text}")
                    
                    else:
                        print("No A2A agents found")
                else:
                    text = await response.text()
                    print(f"Error: {text}")
        except Exception as e:
            print(f"Exception: {e}")

if __name__ == "__main__":
    asyncio.run(debug_message_sending())
