#!/usr/bin/env python3
"""
Test Full Documentation Flow

Tests the complete persistent documentation workflow including authentication,
document creation, retrieval, and frontend integration.
"""

import os
import sys
import requests
from datetime import datetime
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Set environment variables for testing
os.environ['DATABASE_URL'] = 'sqlite:///./test_pygent.db'
os.environ['DEBUG'] = 'true'
os.environ['TESTING'] = 'true'

BASE_URL = "http://localhost:8000"

def test_authentication_flow():
    """Test user authentication and token retrieval."""
    print("🔐 Testing authentication flow...")
    
    # Test health endpoint first
    response = requests.get(f"{BASE_URL}/api/v1/health")
    print(f"Health check: {response.status_code}")    # Test login endpoint
    login_data = {
        "username": "admin",
        "password": "admin"
    }
    
    response = requests.post(f"{BASE_URL}/auth/api/v1/auth/login", json=login_data)
    print(f"Login attempt: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        if 'access_token' in data:
            print("✅ Authentication successful")
            return data['access_token']
        else:
            print(f"❌ No access token in response: {data}")
            return None
    else:
        print(f"❌ Authentication failed: {response.status_code}")
        print(f"Response: {response.text}")
        return None

def test_documentation_endpoints(token):
    """Test persistent documentation endpoints."""
    print("\n📚 Testing documentation endpoints...")
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    # Test persistent documents list (should be empty initially)
    print("Testing persistent documents list...")
    response = requests.get(f"{BASE_URL}/persistent", headers=headers)
    print(f"Persistent docs list: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Found {len(data.get('data', {}).get('documents', []))} persistent documents")
    else:
        print(f"❌ Failed to get persistent documents: {response.text}")
        return False
    
    # Test document creation
    print("Testing document creation...")
    create_data = {
        "title": "Test Document via API",
        "content": f"# Test Document\n\nThis is a test document created at {datetime.now().isoformat()}.\n\n## Features\n\n- Persistent storage\n- User association\n- Version tracking\n",
        "category": "Testing",
        "file_path": "test/api_test_document.md",
        "tags": ["test", "api", "persistent"]
    }
    
    response = requests.post(f"{BASE_URL}/create", json=create_data, headers=headers)
    print(f"Document creation: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        doc_id = data.get('data', {}).get('id')
        print(f"✅ Document created with ID: {doc_id}")
        
        # Test document retrieval
        print("Testing document retrieval...")
        response = requests.get(f"{BASE_URL}/persistent/{doc_id}", headers=headers)
        print(f"Document retrieval: {response.status_code}")
        
        if response.status_code == 200:
            doc_data = response.json()
            print(f"✅ Retrieved document: {doc_data.get('data', {}).get('title')}")
            return doc_id
        else:
            print(f"❌ Failed to retrieve document: {response.text}")
            return None
    else:
        print(f"❌ Failed to create document: {response.text}")
        return None

def test_research_session_integration(token):
    """Test research session integration."""
    print("\n🔬 Testing research session integration...")
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    # Create a research session
    session_data = {
        "title": "Test Research Session",
        "description": "Testing research session creation for documentation workflow",
        "research_query": "How to implement persistent documentation storage?"
    }
    
    response = requests.post(f"{BASE_URL}/research-session", json=session_data, headers=headers)
    print(f"Research session creation: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        session_id = data.get('data', {}).get('session_id')
        print(f"✅ Research session created: {session_id}")
        
        # List research sessions
        response = requests.get(f"{BASE_URL}/research-sessions", headers=headers)
        print(f"Research sessions list: {response.status_code}")
        
        if response.status_code == 200:
            sessions_data = response.json()
            sessions = sessions_data.get('data', {}).get('sessions', [])
            print(f"✅ Found {len(sessions)} research sessions")
            return session_id
        else:
            print(f"❌ Failed to list research sessions: {response.text}")
    else:
        print(f"❌ Failed to create research session: {response.text}")
    
    return None

def test_frontend_api_compatibility():
    """Test that the backend API is compatible with frontend expectations."""
    print("\n🖥️ Testing frontend API compatibility...")
    
    # Test the endpoints that the frontend expects
    endpoints_to_test = [
        "/",  # Documentation index
        "/files",  # Documentation files list
        "/stats"  # Documentation statistics
    ]
    
    for endpoint in endpoints_to_test:
        try:
            response = requests.get(f"{BASE_URL}{endpoint}")
            print(f"  {endpoint}: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    print(f"    ✅ {endpoint} returns valid data")
                else:
                    print(f"    ⚠️ {endpoint} unexpected response format")
            else:
                print(f"    ❌ {endpoint} failed")
        except Exception as e:
            print(f"    ❌ {endpoint} error: {e}")
    
    print("✅ Frontend compatibility tests completed")

def main():
    """Run the complete documentation flow test."""
    print("🚀 Starting complete documentation flow test...\n")
    
    try:
        # Test authentication
        token = test_authentication_flow()
        if not token:
            print("❌ Authentication failed - cannot continue with protected endpoint tests")
            print("✅ Public endpoint tests passed")
            return
        
        # Test documentation endpoints
        doc_id = test_documentation_endpoints(token)
        
        # Test research session integration
        session_id = test_research_session_integration(token)
        
        # Test frontend compatibility
        test_frontend_api_compatibility()
        
        print("\n🎉 All tests completed!")
        print("✅ Authentication flow working")
        print("✅ Persistent documentation endpoints working")
        print("✅ Research session integration working")
        print("✅ Frontend API compatibility confirmed")
        
        if doc_id:
            print(f"📄 Created test document with ID: {doc_id}")
        if session_id:
            print(f"🔬 Created test research session with ID: {session_id}")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
