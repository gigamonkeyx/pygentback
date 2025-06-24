#!/usr/bin/env python3
"""
Test WebSocket connectivity to production deployment at timpayne.net
"""

import asyncio
import websockets
import json
import ssl

async def test_production_websocket():
    """Test production WebSocket connection"""
    uri = "wss://ws.timpayne.net/ws"
    
    try:
        print(f"🔍 Testing production WebSocket: {uri}")
        
        # Create SSL context for secure connection
        ssl_context = ssl.create_default_context()
        
        async with websockets.connect(uri, ssl=ssl_context) as websocket:
            print("✅ Production WebSocket connection established!")
            
            # Send a test message
            test_message = {
                "type": "ping",
                "data": {"message": "Hello from test client"},
                "timestamp": "2025-01-01T00:00:00Z"
            }
            
            await websocket.send(json.dumps(test_message))
            print(f"📤 Sent test message: {test_message}")
            
            # Wait for response (with timeout)
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                print(f"📨 Received response: {response}")
            except asyncio.TimeoutError:
                print("⏰ No response received within 10 seconds (this may be normal)")
            
            print("✅ Production WebSocket test completed successfully!")
            return True
            
    except websockets.exceptions.ConnectionClosed as e:
        print(f"❌ WebSocket connection closed: {e}")
        return False
    except websockets.exceptions.InvalidURI as e:
        print(f"❌ Invalid WebSocket URI: {e}")
        return False
    except websockets.exceptions.InvalidHandshake as e:
        print(f"❌ WebSocket handshake failed: {e}")
        return False
    except Exception as e:
        print(f"❌ WebSocket connection failed: {e}")
        return False

async def test_api_endpoint():
    """Test if the API endpoint is accessible"""
    import aiohttp
    
    try:
        print("🔍 Testing API endpoint accessibility...")
        
        async with aiohttp.ClientSession() as session:
            # Test health endpoint
            async with session.get("https://api.timpayne.net/api/v1/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ API health check successful: {data}")
                    return True
                else:
                    print(f"❌ API health check failed: {response.status}")
                    return False
                    
    except Exception as e:
        print(f"❌ API endpoint test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 Production WebSocket Connectivity Test")
    print("=" * 50)
    
    # Test API endpoint first
    api_success = await test_api_endpoint()
    print()
    
    # Test WebSocket connection
    ws_success = await test_production_websocket()
    print()
    
    print("📊 Test Results:")
    print(f"   API Endpoint:      {'✅ PASS' if api_success else '❌ FAIL'}")
    print(f"   WebSocket:         {'✅ PASS' if ws_success else '❌ FAIL'}")
    
    if api_success and ws_success:
        print("\n🎉 All production connectivity tests passed!")
    else:
        print("\n⚠️  Some production connectivity tests failed.")

if __name__ == "__main__":
    asyncio.run(main())
