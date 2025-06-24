#!/usr/bin/env python3
"""
Test API connection with custom DNS resolution
"""

import socket
import requests
import time

def test_api_connection():
    """Test API connection with various approaches"""
    
    print("🔍 Testing API Connection with Multiple Approaches")
    print("=" * 60)
    
    # Approach 1: Direct IP connection
    print("\n1. Testing with direct IP (104.21.47.61)...")
    try:
        response = requests.get(
            "https://104.21.47.61/api/v1/health",
            headers={"Host": "api.timpayne.net"},
            timeout=30,
            verify=False  # Skip SSL verification for IP-based request
        )
        print(f"   ✅ Direct IP: Status {response.status_code}")
        print(f"   Response: {response.text[:100]}...")
    except Exception as e:
        print(f"   ❌ Direct IP failed: {e}")
    
    # Approach 2: Try with different DNS
    print("\n2. Testing with custom DNS resolution...")
    try:
        # Manually resolve using Google DNS
        import dns.resolver
        resolver = dns.resolver.Resolver()
        resolver.nameservers = ['8.8.8.8', '1.1.1.1']
        
        answers = resolver.resolve('api.timpayne.net', 'A')
        ip = str(answers[0])
        print(f"   DNS resolved to: {ip}")
        
        response = requests.get(
            f"https://{ip}/api/v1/health",
            headers={"Host": "api.timpayne.net"},
            timeout=30,
            verify=False
        )
        print(f"   ✅ Custom DNS: Status {response.status_code}")
        print(f"   Response: {response.text[:100]}...")
    except ImportError:
        print("   ⚠️  dnspython not available, skipping custom DNS test")
    except Exception as e:
        print(f"   ❌ Custom DNS failed: {e}")
    
    # Approach 3: Wait and retry with domain name
    print("\n3. Testing with domain name (with retries)...")
    for attempt in range(3):
        try:
            print(f"   Attempt {attempt + 1}/3...")
            response = requests.get(
                "https://api.timpayne.net/api/v1/health",
                timeout=30
            )
            print(f"   ✅ Domain name: Status {response.status_code}")
            print(f"   Response: {response.text[:100]}...")
            break
        except Exception as e:
            print(f"   ❌ Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                print("   ⏳ Waiting 10 seconds before retry...")
                time.sleep(10)
    
    # Approach 4: Test WebSocket domain
    print("\n4. Testing WebSocket domain resolution...")
    try:
        # Just test DNS resolution for WebSocket
        ip = socket.gethostbyname('ws.timpayne.net')
        print(f"   ✅ ws.timpayne.net resolves to: {ip}")
    except Exception as e:
        print(f"   ❌ ws.timpayne.net resolution failed: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 API Connection Test Complete")

if __name__ == "__main__":
    test_api_connection()
