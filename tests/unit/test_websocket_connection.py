#!/usr/bin/env python3
"""
Test WebSocket connection to production deployment
"""

import asyncio
import websockets
import json
import ssl
import socket

async def test_websocket_connection():
    """Test WebSocket connection with multiple approaches"""
    
    print("🔍 Testing WebSocket Connection")
    print("=" * 50)
    
    # Test 1: Try with domain name (if DNS resolves)
    print("\n1. Testing with domain name: wss://ws.timpayne.net/ws")
    try:
        # First check if DNS resolves
        try:
            ip = socket.gethostbyname('ws.timpayne.net')
            print(f"   DNS resolved to: {ip}")
        except socket.gaierror:
            print("   ❌ DNS resolution failed for ws.timpayne.net")
            print("   ⏭️  Skipping domain name test")
            return False
        
        # Try WebSocket connection
        ssl_context = ssl.create_default_context()
        async with websockets.connect("wss://ws.timpayne.net/ws", ssl=ssl_context) as websocket:
            print("   ✅ WebSocket connection established!")
            
            # Send test message
            test_message = {
                "type": "ping",
                "data": {"message": "Hello from test client"},
                "timestamp": "2025-06-04T05:57:00Z"
            }
            
            await websocket.send(json.dumps(test_message))
            print(f"   📤 Sent: {test_message}")
            
            # Wait for response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                print(f"   📨 Received: {response}")
                return True
            except asyncio.TimeoutError:
                print("   ⏰ No response within 10 seconds (connection working, no echo)")
                return True
                
    except websockets.exceptions.ConnectionClosed as e:
        print(f"   ❌ WebSocket connection closed: {e}")
        return False
    except websockets.exceptions.InvalidURI as e:
        print(f"   ❌ Invalid WebSocket URI: {e}")
        return False
    except websockets.exceptions.InvalidHandshake as e:
        print(f"   ❌ WebSocket handshake failed: {e}")
        return False
    except Exception as e:
        print(f"   ❌ WebSocket connection failed: {e}")
        return False

async def test_websocket_via_api_domain():
    """Test WebSocket connection via API domain (fallback)"""
    
    print("\n2. Testing WebSocket via API domain: wss://api.timpayne.net/ws")
    try:
        ssl_context = ssl.create_default_context()
        async with websockets.connect("wss://api.timpayne.net/ws", ssl=ssl_context) as websocket:
            print("   ✅ WebSocket connection via API domain established!")
            
            # Send test message
            test_message = {
                "type": "ping",
                "data": {"message": "Hello via API domain"},
                "timestamp": "2025-06-04T05:57:00Z"
            }
            
            await websocket.send(json.dumps(test_message))
            print(f"   📤 Sent: {test_message}")
            
            # Wait for response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                print(f"   📨 Received: {response}")
                return True
            except asyncio.TimeoutError:
                print("   ⏰ No response within 10 seconds (connection working, no echo)")
                return True
                
    except Exception as e:
        print(f"   ❌ WebSocket via API domain failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 WebSocket Connectivity Test for PyGent Factory")
    print("=" * 60)
    
    # Test WebSocket connection
    ws_success = await test_websocket_connection()
    
    # If first test fails, try via API domain
    if not ws_success:
        ws_success = await test_websocket_via_api_domain()
    
    print("\n" + "=" * 60)
    print("📊 WebSocket Test Results:")
    print(f"   WebSocket Connection: {'✅ PASS' if ws_success else '❌ FAIL'}")
    
    if ws_success:
        print("\n🎉 WebSocket connectivity test passed!")
        print("   The frontend should be able to connect to WebSocket endpoints.")
    else:
        print("\n⚠️  WebSocket connectivity test failed.")
        print("   Check DNS configuration and tunnel routing.")

if __name__ == "__main__":
    asyncio.run(main())
