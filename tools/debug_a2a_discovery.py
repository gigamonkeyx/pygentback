#!/usr/bin/env python3
"""
Debug A2A Agent Discovery

Debug the A2A agent discovery process.
"""

import asyncio
import aiohttp
import json

async def debug_discovery():
    """Debug A2A agent discovery"""
    
    async with aiohttp.ClientSession() as session:
        print("🔍 Testing A2A MCP Server Agent Listing...")
        
        # Test A2A server agent listing
        try:
            async with session.get("http://127.0.0.1:8006/mcp/a2a/agents") as response:
                print(f"A2A Server Status: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    print(f"A2A Server Response: {json.dumps(data, indent=2)}")
                else:
                    text = await response.text()
                    print(f"A2A Server Error: {text}")
        except Exception as e:
            print(f"A2A Server Exception: {e}")
        
        print("\n🔍 Testing Orchestration Server A2A Discovery...")
        
        # Test orchestration server A2A discovery
        try:
            async with session.get("http://127.0.0.1:8005/v1/a2a/agents") as response:
                print(f"Orchestration Status: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    print(f"Orchestration Response: {json.dumps(data, indent=2)}")
                else:
                    text = await response.text()
                    print(f"Orchestration Error: {text}")
        except Exception as e:
            print(f"Orchestration Exception: {e}")
        
        print("\n🔄 Testing Manual Discovery Trigger...")
        
        # Test manual discovery trigger
        try:
            async with session.post("http://127.0.0.1:8005/v1/a2a/discover") as response:
                print(f"Discovery Trigger Status: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    print(f"Discovery Trigger Response: {json.dumps(data, indent=2)}")
                else:
                    text = await response.text()
                    print(f"Discovery Trigger Error: {text}")
        except Exception as e:
            print(f"Discovery Trigger Exception: {e}")
        
        print("\n🔍 Re-testing Orchestration Server A2A Discovery...")
        
        # Test orchestration server A2A discovery again
        try:
            async with session.get("http://127.0.0.1:8005/v1/a2a/agents") as response:
                print(f"Orchestration Status: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    print(f"Orchestration Response: {json.dumps(data, indent=2)}")
                else:
                    text = await response.text()
                    print(f"Orchestration Error: {text}")
        except Exception as e:
            print(f"Orchestration Exception: {e}")

if __name__ == "__main__":
    asyncio.run(debug_discovery())
