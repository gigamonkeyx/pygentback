#!/usr/bin/env python3
"""
Test script to check MCP server endpoints and identify issues
with the MCP Marketplace page.
"""

import requests
import json
import sys

def test_mcp_endpoints():
    """Test MCP server endpoints that the UI is trying to access."""
    
    base_url = "http://localhost:8000"
    
    print("Testing MCP Server Endpoints")
    print("=" * 50)
    
    # Test endpoints that the UI tries to access
    endpoints_to_test = [
        "/api/v1/mcp/discovery/status",
        "/api/v1/mcp/discovery/servers", 
        "/api/v1/mcp/servers",
        "/health"
    ]
    
    for endpoint in endpoints_to_test:
        try:
            print(f"\nTesting: {endpoint}")
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print("✓ SUCCESS")
                    if endpoint == "/api/v1/mcp/discovery/status":
                        print(f"  Discovery enabled: {data.get('discovery_enabled', False)}")
                        print(f"  Status: {data.get('status', 'unknown')}")
                        if 'results' in data:
                            results = data['results']
                            print(f"  Servers discovered: {results.get('servers_discovered', 0)}")
                            print(f"  Servers registered: {results.get('total_servers_registered', 0)}")
                    elif endpoint == "/api/v1/mcp/discovery/servers":
                        print(f"  Total discovered: {data.get('total_discovered', 0)}")
                        categories = data.get('categories', {})
                        print(f"  Categories: {list(categories.keys())}")
                        for cat, servers in categories.items():
                            print(f"    {cat}: {len(servers)} servers")
                    elif endpoint == "/api/v1/mcp/servers":
                        servers = data.get('servers', [])
                        print(f"  Registered servers: {len(servers)}")
                except json.JSONDecodeError:
                    print("✓ SUCCESS (non-JSON response)")
                    print(f"  Response preview: {response.text[:200]}...")
            else:
                print(f"✗ FAILED - HTTP {response.status_code}")
                print(f"  Error: {response.text[:200]}...")
                
        except Exception as e:
            print(f"✗ ERROR: {e}")
    
    return True

def analyze_ui_issues():
    """Analyze potential issues with the MCP UI."""
    print("\n" + "=" * 50)
    print("MCP UI Analysis")
    print("=" * 50)
    
    print("\nPotential Issues:")
    print("1. Backend server not running on localhost:8000")
    print("2. MCP discovery not initialized or failed")
    print("3. Authentication required for MCP endpoints")
    print("4. Network/CORS issues")
    print("5. Missing MCP server configurations")
    
    print("\nUI Expectations:")
    print("- GET /api/v1/mcp/discovery/status -> discovery status")
    print("- GET /api/v1/mcp/discovery/servers -> available servers")
    print("- POST /api/v1/mcp/servers/install -> install server")
    print("- GET /api/v1/mcp/servers/install/{name}/status -> installation status")
    
    print("\nTo fix:")
    print("1. Start backend: python -m uvicorn src.api.main:app --reload")
    print("2. Check if MCP discovery is working")
    print("3. Verify API authentication")
    print("4. Test UI endpoints manually")

def main():
    """Main test function."""
    print("MCP Marketplace Debugging")
    print("=" * 60)
    
    if not test_mcp_endpoints():
        print("MCP endpoint tests failed!")
        return 1
    
    analyze_ui_issues()
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("1. If endpoints fail -> Start backend server")
    print("2. If discovery returns empty -> Check MCP server discovery system")
    print("3. If authentication errors -> Check auth requirements")
    print("4. Test UI with browser dev tools for specific errors")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
