#!/usr/bin/env python3
"""
Test WebSocket message sending to debug query pipeline
"""

import asyncio
import websockets
import json


async def test_websocket():
    uri = "ws://localhost:8000/api/ws"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket")
            
            # Send a test message with the problematic query
            test_message = {
                "type": "chat_message",
                "data": {
                    "message": {
                        "id": "test-msg-12345",
                        "type": "user",
                        "content": "6*9",  # This is our test case
                        "timestamp": "2024-01-01T12:00:00.000Z",
                        "agentId": "general"
                    },
                    "timestamp": "2024-01-01T12:00:00.000Z"
                }
            }
            
            print(f"Sending message: {json.dumps(test_message, indent=2)}")
            await websocket.send(json.dumps(test_message))
            
            # Wait for response
            print("Waiting for response...")
            response = await websocket.recv()
            print(f"Received response: {response}")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_websocket())
