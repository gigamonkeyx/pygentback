#!/usr/bin/env python3
"""
Register Simple A2A Agent

Register the simple A2A agent with the A2A MCP Server for discovery.
"""

import asyncio
import aiohttp
import json

async def register_agent():
    """Register the simple A2A agent"""
    
    async with aiohttp.ClientSession() as session:
        print("🔗 Registering Simple A2A Agent with A2A MCP Server...")
        
        # Register the agent
        registration_payload = {
            "agent_url": "http://127.0.0.1:8007"
        }
        
        try:
            async with session.post(
                "http://127.0.0.1:8006/mcp/a2a/discover_agent",
                json=registration_payload
            ) as response:
                print(f"Registration Status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    print(f"Registration Response: {json.dumps(data, indent=2)}")
                    
                    agent_id = data.get('agent_id')
                    agent_name = data.get('agent_card', {}).get('name', 'Unknown')
                    
                    print(f"\n✅ Successfully registered agent: {agent_name} (ID: {agent_id})")
                    
                    # Verify registration by listing agents
                    print("\n🔍 Verifying registration...")
                    async with session.get("http://127.0.0.1:8006/mcp/a2a/agents") as verify_response:
                        if verify_response.status == 200:
                            verify_data = await verify_response.json()
                            total_agents = verify_data.get('total_agents', 0)
                            agents = verify_data.get('agents', {})
                            
                            print(f"Total registered agents: {total_agents}")
                            for aid, agent_info in agents.items():
                                print(f"  - {agent_info['name']} ({aid})")
                        else:
                            print(f"Verification failed: {verify_response.status}")
                    
                else:
                    error_text = await response.text()
                    print(f"Registration failed: {error_text}")
                    
        except Exception as e:
            print(f"Registration error: {e}")

if __name__ == "__main__":
    asyncio.run(register_agent())
