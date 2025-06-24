#!/usr/bin/env python3
"""
Test Public Documentation Endpoints

Tests the public documentation endpoints that don't require authentication.
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_public_endpoints():
    """Test public documentation endpoints."""
    print("🌐 Testing public documentation endpoints...\n")
    
    # Test documentation index
    print("1. Testing documentation index:")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                total_files = data.get('data', {}).get('total_files', 0)
                print(f"   ✅ Found {total_files} documentation files")
                categories = data.get('data', {}).get('categories', {})
                print(f"   📁 Categories: {list(categories.keys())}")
            else:
                print(f"   ❌ Unexpected response format: {data}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test documentation files list
    print("\n2. Testing documentation files list:")
    try:
        response = requests.get(f"{BASE_URL}/files")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                files = data.get('data', {}).get('files', [])
                print(f"   ✅ Found {len(files)} files")
                if files:
                    print(f"   📄 Sample file: {files[0].get('title')}")
            else:
                print(f"   ❌ Unexpected response format")
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test documentation search
    print("\n3. Testing documentation search:")
    try:
        response = requests.get(f"{BASE_URL}/search?query=agent")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                results = data.get('data', {}).get('results', [])
                print(f"   ✅ Found {len(results)} search results for 'agent'")
                if results:
                    print(f"   🔍 Sample result: {results[0].get('title')}")
            else:
                print(f"   ❌ Unexpected response format")
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test getting a specific file
    print("\n4. Testing specific file retrieval:")
    try:
        # Try to get a file that should exist
        response = requests.get(f"{BASE_URL}/file/MASTER_DOCUMENTATION_INDEX.md")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                content = data.get('data', {}).get('content', '')
                title = data.get('data', {}).get('title', '')
                print(f"   ✅ Retrieved file: {title}")
                print(f"   📝 Content length: {len(content)} characters")
                html_content = data.get('data', {}).get('html_content', '')
                if html_content:
                    print(f"   🌐 HTML content generated: {len(html_content)} characters")
            else:
                print(f"   ❌ Unexpected response format")
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test documentation stats
    print("\n5. Testing documentation statistics:")
    try:
        response = requests.get(f"{BASE_URL}/stats")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                stats = data.get('data', {})
                total_files = stats.get('total_files', 0)
                total_size_mb = stats.get('total_size_mb', 0)
                categories = stats.get('categories', {})
                print(f"   ✅ Total files: {total_files}")
                print(f"   📊 Total size: {total_size_mb} MB")
                print(f"   📁 Categories: {categories}")
            else:
                print(f"   ❌ Unexpected response format")
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

def test_cors_headers():
    """Test CORS headers for frontend compatibility."""
    print("\n🌍 Testing CORS headers for frontend compatibility...")
    
    try:
        # Test preflight request
        headers = {
            'Origin': 'http://localhost:5173',
            'Access-Control-Request-Method': 'GET',
            'Access-Control-Request-Headers': 'Content-Type,Authorization'
        }
        
        response = requests.options(f"{BASE_URL}/", headers=headers)
        print(f"   Preflight request status: {response.status_code}")
        
        cors_headers = {
            'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin'),
            'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods'),
            'Access-Control-Allow-Headers': response.headers.get('Access-Control-Allow-Headers'),
            'Access-Control-Allow-Credentials': response.headers.get('Access-Control-Allow-Credentials')
        }
        
        print(f"   CORS headers: {cors_headers}")
        
        if cors_headers['Access-Control-Allow-Origin']:
            print("   ✅ CORS properly configured for frontend")
        else:
            print("   ⚠️ CORS may not be configured for frontend")
            
    except Exception as e:
        print(f"   ❌ CORS test error: {e}")

def main():
    """Run all public endpoint tests."""
    print("🚀 Testing PyGent Factory Documentation API (Public Endpoints)\n")
    
    test_public_endpoints()
    test_cors_headers()
    
    print("\n🎉 Public endpoint testing completed!")
    print("✅ These endpoints should work with the frontend")
    print("🔐 Authentication required for persistent document management")

if __name__ == "__main__":
    main()
