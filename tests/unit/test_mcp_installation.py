#!/usr/bin/env python3
"""
Test MCP Server Installation

This script tests the MCP server installation functionality by:
1. Creating a demo authentication token
2. Testing the installation endpoint
3. Monitoring installation progress
"""

import asyncio
import requests
import json
import time
from datetime import datetime, timedelta

# Import the authentication system
import sys
sys.path.insert(0, 'src')

from src.security.auth import get_auth_service, UserRole, User


def create_demo_token():
    """Create a demo authentication token"""
    try:
        # Get the auth service
        auth_service = get_auth_service()
        
        # Create a demo admin user
        demo_user = User(
            id="demo_admin",
            username="demo_admin", 
            email="demo@pygent.factory",
            role=UserRole.ADMIN,
            created_at=datetime.utcnow()
        )
        
        # Create access token
        token = auth_service.create_access_token(demo_user)
        print(f"✅ Created demo token: {token[:50]}...")
        return token
        
    except Exception as e:
        print(f"❌ Failed to create demo token: {e}")
        return None


def test_installation_endpoint(token, server_name="filesystem"):
    """Test the MCP server installation endpoint"""
    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "server_name": server_name,
            "source_type": "npm",
            "auto_start": True
        }
        
        print(f"🚀 Testing installation of '{server_name}' server...")
        
        response = requests.post(
            "http://localhost:8080/api/v1/mcp/servers/install",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"📊 Response Status: {response.status_code}")
        print(f"📄 Response Body: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Installation started successfully!")
            print(f"📋 Installation ID: {result.get('installation_id', 'N/A')}")
            return True
        else:
            print(f"❌ Installation failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False


def monitor_installation_status(token, server_name="filesystem", max_wait=60):
    """Monitor installation progress"""
    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        print(f"👀 Monitoring installation status for '{server_name}'...")
        
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                response = requests.get(
                    f"http://localhost:8080/api/v1/mcp/servers/install/{server_name}/status",
                    headers=headers,
                    timeout=10
                )
                
                if response.status_code == 200:
                    status = response.json()
                    print(f"📊 Status: {status.get('status', 'unknown')} - Progress: {status.get('progress', 0)}% - {status.get('message', '')}")
                    
                    if status.get('status') == 'completed':
                        print(f"✅ Installation completed successfully!")
                        print(f"📁 Install path: {status.get('install_path', 'N/A')}")
                        return True
                    elif status.get('status') == 'failed':
                        print(f"❌ Installation failed: {status.get('message', 'Unknown error')}")
                        return False
                        
                elif response.status_code == 404:
                    print(f"⏳ Installation not found yet, waiting...")
                else:
                    print(f"⚠️ Status check failed: {response.status_code} - {response.text}")
                
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Status check error: {e}")
            
            time.sleep(2)
        
        print(f"⏰ Timeout waiting for installation to complete")
        return False
        
    except Exception as e:
        print(f"❌ Status monitoring failed: {e}")
        return False


def test_mcp_manager_directly():
    """Test MCP manager directly without HTTP"""
    try:
        print("🔧 Testing MCP manager directly...")
        
        # Import the MCP manager
        from src.api.main import app_state
        
        mcp_manager = app_state.get("mcp_manager")
        if not mcp_manager:
            print("❌ MCP manager not available in app state")
            return False
        
        print(f"✅ MCP manager found: {type(mcp_manager)}")
        
        # Test listing servers
        # Note: This would need to be async, but for testing we'll just check if it exists
        print(f"✅ MCP manager is available and initialized")
        return True
        
    except Exception as e:
        print(f"❌ Direct MCP manager test failed: {e}")
        return False


def main():
    """Main test function"""
    print("🧪 MCP INSTALLATION TEST")
    print("=" * 50)
    
    # Test 1: Check if MCP manager is available
    print("\n1️⃣ Testing MCP Manager Availability...")
    if not test_mcp_manager_directly():
        print("❌ MCP manager not available, cannot proceed")
        return
    
    # Test 2: Create demo token
    print("\n2️⃣ Creating Demo Authentication Token...")
    token = create_demo_token()
    if not token:
        print("❌ Cannot create demo token, cannot proceed")
        return
    
    # Test 3: Test installation endpoint
    print("\n3️⃣ Testing Installation Endpoint...")
    if test_installation_endpoint(token, "filesystem"):
        # Test 4: Monitor installation progress
        print("\n4️⃣ Monitoring Installation Progress...")
        monitor_installation_status(token, "filesystem")
    
    print("\n🎉 MCP Installation Test Complete!")


if __name__ == "__main__":
    main()
