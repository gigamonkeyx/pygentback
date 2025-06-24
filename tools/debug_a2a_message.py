#!/usr/bin/env python3
"""
Debug A2A Message Sending

Test A2A message sending with detailed error reporting.
"""

import asyncio
import aiohttp
import json

async def test_message_sending():
    """Test A2A message sending with detailed error reporting"""
    
    message_payload = {
        "jsonrpc": "2.0",
        "id": "debug-001",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": "Debug test message"
                    }
                ],
                "messageId": "debug-msg-001"
            }
        }
    }
    
    print("🔍 Testing A2A Message Sending...")
    print(f"Payload: {json.dumps(message_payload, indent=2)}")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                "http://127.0.0.1:8006/a2a/message/send",
                json=message_payload
            ) as response:
                print(f"Status: {response.status}")
                print(f"Headers: {dict(response.headers)}")
                
                if response.status == 200:
                    data = await response.json()
                    print(f"Response: {json.dumps(data, indent=2)}")
                else:
                    text = await response.text()
                    print(f"Error Response: {text}")
                    
        except Exception as e:
            print(f"Exception: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_message_sending())
