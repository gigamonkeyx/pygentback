#!/usr/bin/env python3
"""
Quick Backend API Test

Test that the health endpoints are working correctly after fixes.
"""

import requests
import sys

BASE_URL = "http://localhost:8000"

def test_health_endpoints():
    """Test both GET and HEAD health endpoints."""
    print("🏥 Testing Health Endpoints...")
    
    # Test GET /api/v1/health
    try:
        print("1. Testing GET /api/v1/health:")
        response = requests.get(f"{BASE_URL}/api/v1/health", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ GET request successful - Status: {data.get('status', 'unknown')}")
        else:
            print(f"   ❌ GET request failed")
    except Exception as e:
        print(f"   ❌ GET request error: {e}")
    
    # Test HEAD /api/v1/health  
    try:
        print("2. Testing HEAD /api/v1/health:")
        response = requests.head(f"{BASE_URL}/api/v1/health", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   ✅ HEAD request successful")
        else:
            print(f"   ❌ HEAD request failed")
    except Exception as e:
        print(f"   ❌ HEAD request error: {e}")

def test_documentation_endpoints():
    """Test documentation endpoints that frontend uses."""
    print("\n📚 Testing Documentation Endpoints...")
    
    endpoints = [
        "/",  # Documentation index
        "/files",  # Documentation files
        "/stats"  # Documentation stats
    ]
    
    for endpoint in endpoints:
        try:
            print(f"Testing {endpoint}:")
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=5)
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ {endpoint} working")
            else:
                print(f"   ❌ {endpoint} failed")
        except Exception as e:
            print(f"   ❌ {endpoint} error: {e}")

def test_cors_headers():
    """Test CORS headers for frontend compatibility."""
    print("\n🌍 Testing CORS Headers...")
    
    headers = {
        'Origin': 'http://localhost:5173',
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/health", headers=headers, timeout=5)
        cors_header = response.headers.get('Access-Control-Allow-Origin')
        print(f"CORS Origin Header: {cors_header}")
        
        if cors_header:
            print("✅ CORS properly configured")
        else:
            print("⚠️ CORS headers may not be configured")
            
    except Exception as e:
        print(f"❌ CORS test error: {e}")

def main():
    print("🚀 Quick Backend API Test\n")
    
    test_health_endpoints()
    test_documentation_endpoints()
    test_cors_headers()
    
    print("\n🎉 Test completed!")

if __name__ == "__main__":
    main()
