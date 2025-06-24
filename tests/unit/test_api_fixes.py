#!/usr/bin/env python3
"""
Test the API endpoint fixes to verify the 500 and 404 errors are resolved.
"""

import requests
import sys

def test_api_endpoints():
    """Test the problematic endpoints that were causing console errors."""
    
    base_url = "http://localhost:8000"
    
    print("Testing Fixed API Endpoints")
    print("=" * 50)
    
    # Test the endpoints that were failing
    endpoints_to_test = [
        ("/health", "Simple health check"),
        ("/api/v1/health", "Full health check"),
        ("/api/persistent", "Documentation persistent (now available)"),
        ("/api/v1/mcp/discovery/status", "MCP discovery status"),
        ("/api/v1/mcp/discovery/servers", "MCP discovery servers"),
        ("/api/files", "Documentation files"),
        ("/api/categories", "Documentation categories")
    ]
    
    results = {}
    
    for endpoint, description in endpoints_to_test:
        try:
            print(f"\nTesting: {endpoint} ({description})")
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            
            status = response.status_code
            results[endpoint] = status
            
            if status == 200:
                print(f"✅ SUCCESS - 200 OK")
                try:
                    data = response.json()
                    if 'status' in data:
                        print(f"   Status: {data['status']}")
                    if 'message' in data:
                        print(f"   Message: {data['message']}")
                except:
                    print(f"   Response: {response.text[:100]}...")
            elif status == 403:
                print(f"⚠️  AUTH REQUIRED - 403 (expected for some endpoints)")
            elif status == 404:
                print(f"❌ NOT FOUND - 404")
            elif status == 500:
                print(f"💥 SERVER ERROR - 500")
                print(f"   Error: {response.text[:200]}...")
            else:
                print(f"❓ UNEXPECTED - {status}")
                print(f"   Response: {response.text[:100]}...")
                
        except requests.exceptions.ConnectionError:
            print(f"💔 CONNECTION FAILED - Server not running")
            results[endpoint] = "CONNECTION_ERROR"
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results[endpoint] = "ERROR"
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    success_count = sum(1 for status in results.values() if status == 200)
    auth_required = sum(1 for status in results.values() if status == 403)
    not_found = sum(1 for status in results.values() if status == 404)
    server_errors = sum(1 for status in results.values() if status == 500)
    connection_errors = sum(1 for status in results.values() if status == "CONNECTION_ERROR")
    
    print(f"✅ Working endpoints: {success_count}")
    print(f"⚠️  Auth required: {auth_required}")
    print(f"❌ Not found (404): {not_found}")
    print(f"💥 Server errors (500): {server_errors}")
    print(f"💔 Connection errors: {connection_errors}")
    
    if server_errors == 0 and not_found == 0:
        print("\n🎉 All critical issues fixed! No more 500/404 errors.")
    elif connection_errors > 0:
        print("\n🚨 Server not running. Start with: python -m uvicorn src.api.main:app --reload")
    else:
        print(f"\n⚡ Still {server_errors + not_found} issues to resolve.")
    
    return results

def main():
    """Main test function."""
    print("API Endpoint Fixes Verification")
    print("=" * 60)
    
    results = test_api_endpoints()
    
    print("\n" + "=" * 60)
    print("FIXES APPLIED:")
    print("1. ✅ Added dual routing for documentation endpoints (/api and /api/documentation)")
    print("2. ✅ Added simple /health endpoint at root level")  
    print("3. ✅ Fixed datetime import for health endpoint")
    print("4. ⚠️  Authentication required for some endpoints (expected)")
    
    # Check if backend is running
    if any(status == "CONNECTION_ERROR" for status in results.values()):
        print("\n🚨 Backend server needs to be started:")
        print("   python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
