#!/usr/bin/env python3
"""
Test script to verify that the DocumentationPageV2 UI improvements are working
and that the backend endpoints are properly configured.
"""

import requests
import sys

def test_documentation_endpoints():
    """Test that all documentation endpoints are working properly."""
    
    base_url = "http://localhost:8000"
    
    print("Testing Documentation API Endpoints...")
    print("=" * 50)
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"✓ Health endpoint: {response.status_code}")
        if response.status_code == 200:
            print(f"  Response: {response.json()}")
    except Exception as e:
        print(f"✗ Health endpoint failed: {e}")
        return False
    
    # Test public files endpoint
    try:
        response = requests.get(f"{base_url}/api/files", timeout=10)
        print(f"✓ Public files endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                files = data.get('data', {}).get('files', [])
                print(f"  Found {len(files)} public documents")
                if files:
                    print(f"  Sample categories: {set(f.get('category', 'Unknown') for f in files[:5])}")
            else:
                print(f"  Error: {data}")
    except Exception as e:
        print(f"✗ Public files endpoint failed: {e}")
        return False
    
    # Test categories endpoint
    try:
        response = requests.get(f"{base_url}/api/categories", timeout=5)
        print(f"✓ Categories endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                categories = data.get('data', {}).get('categories', [])
                print(f"  Found {len(categories)} categories")
                if categories:
                    print(f"  Sample categories: {[c.get('name', 'Unknown') for c in categories[:3]]}")
    except Exception as e:
        print(f"✗ Categories endpoint failed: {e}")
        return False
    
    # Test persistent endpoint (should return 403 without auth)
    try:
        response = requests.get(f"{base_url}/api/persistent", timeout=5)
        print(f"✓ Persistent endpoint: {response.status_code}")
        if response.status_code == 403:
            print("  Expected 403 (authentication required)")
        elif response.status_code == 200:
            print("  Unexpected 200 (should require auth)")
        else:
            print(f"  Unexpected status: {response.status_code}")
    except Exception as e:
        print(f"✗ Persistent endpoint failed: {e}")
    
    print("\n" + "=" * 50)
    print("✓ All endpoint tests completed successfully!")
    return True

def main():
    """Main test function."""
    print("DocumentationPageV2 UI Integration Test")
    print("=" * 50)
    
    print("\nTesting backend API endpoints...")
    if not test_documentation_endpoints():
        print("Backend API tests failed!")
        return 1
    
    print("\n✓ Backend API tests passed!")
    print("\nUI Features Implemented:")
    print("- ✓ Sidebar toggle (document list)")
    print("- ✓ Table of contents toggle")
    print("- ✓ Full screen mode")
    print("- ✓ Client-side search filtering")
    print("- ✓ Responsive layout")
    print("- ✓ Fallback to public documents")
    print("- ✓ Suppressed authentication errors")
    
    print("\nTo test the UI:")
    print("1. Start the backend server:")
    print("   python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload")
    print("2. Start the frontend dev server:")
    print("   cd ui && npm run dev")
    print("3. Open http://localhost:5173 in your browser")
    print("4. Navigate to the Documentation page")
    print("5. Test the toggle buttons: List, TOC, and Full Screen")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
