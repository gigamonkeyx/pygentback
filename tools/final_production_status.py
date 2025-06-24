#!/usr/bin/env python3
"""
Final Production Status Check

Quick verification that all systems are operational for go-live.
"""

import asyncio
import aiohttp
import json
from datetime import datetime

async def check_production_status():
    """Quick production status check"""
    
    print("🔍 FINAL PRODUCTION STATUS CHECK")
    print("=" * 50)
    print(f"Check Time: {datetime.utcnow().isoformat()}")
    print("=" * 50)
    
    base_url = "http://localhost:8080"
    
    try:
        async with aiohttp.ClientSession() as session:
            # Health check
            print("\n🏥 Health Check:")
            async with session.get(f"{base_url}/health") as response:
                if response.status == 200:
                    health = await response.json()
                    print(f"   ✅ Status: {health.get('status')}")
                    print(f"   ✅ Agents: {health.get('agents_registered')}")
                    print(f"   ✅ Tasks: {health.get('tasks_active', 'N/A')}")
                else:
                    print(f"   ❌ Health check failed: HTTP {response.status}")
                    return False
            
            # Agent discovery
            print("\n📡 Agent Discovery:")
            async with session.get(f"{base_url}/.well-known/agent.json") as response:
                if response.status == 200:
                    agent_card = await response.json()
                    print(f"   ✅ Agent: {agent_card.get('name')}")
                    print(f"   ✅ Skills: {len(agent_card.get('skills', []))}")
                    print(f"   ✅ Version: {agent_card.get('version')}")
                else:
                    print(f"   ❌ Agent discovery failed: HTTP {response.status}")
                    return False
            
            # Quick task test
            print("\n🧪 Quick Task Test:")
            request = {
                "jsonrpc": "2.0",
                "method": "tasks/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"type": "text", "text": "Final production test"}]
                    }
                },
                "id": "final-test"
            }
            
            async with session.post(base_url, json=request) as response:
                if response.status == 200:
                    result = await response.json()
                    if "result" in result:
                        task_id = result["result"]["id"]
                        print(f"   ✅ Task Created: {task_id[:8]}...")
                        print(f"   ✅ Status: {result['result']['status']['state']}")
                    else:
                        print(f"   ❌ Task creation failed: {result}")
                        return False
                else:
                    print(f"   ❌ Task creation failed: HTTP {response.status}")
                    return False
            
            print("\n" + "=" * 50)
            print("🎉 PRODUCTION STATUS: ALL SYSTEMS OPERATIONAL")
            print("✅ A2A Server: Running")
            print("✅ Health Check: Passing")
            print("✅ Agent Discovery: Working")
            print("✅ Task Processing: Functional")
            print("✅ System Ready: GO-LIVE CONFIRMED")
            print("=" * 50)
            
            return True
            
    except Exception as e:
        print(f"\n❌ Production status check failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(check_production_status())
    if success:
        print("\n🚀 PRODUCTION GO-LIVE: CONFIRMED")
    else:
        print("\n❌ PRODUCTION GO-LIVE: ISSUES DETECTED")
    exit(0 if success else 1)
