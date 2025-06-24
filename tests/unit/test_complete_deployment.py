#!/usr/bin/env python3
"""
Complete PyGent Factory Deployment Test
Tests frontend, API, and WebSocket connectivity
"""

import asyncio
import websockets
import requests
import json
import time
from datetime import datetime

class DeploymentTester:
    def __init__(self):
        self.frontend_url = "https://timpayne.net/pygent"
        self.api_url = "https://api.timpayne.net"
        self.ws_url = "wss://ws.timpayne.net"
        self.results = {}

    def test_frontend_loading(self):
        """Test if the PyGent Factory UI loads correctly"""
        print("🌐 Testing Frontend Loading...")
        try:
            response = requests.get(self.frontend_url, timeout=30)
            
            if response.status_code == 200:
                content = response.text
                
                # Check for PyGent Factory specific content
                if "PyGent Factory" in content:
                    print("   ✅ Frontend loads with PyGent Factory content")
                    self.results['frontend'] = 'SUCCESS'
                    return True
                elif "React" in content or "vite" in content:
                    print("   ⚠️  Frontend loads but may be default React app")
                    self.results['frontend'] = 'PARTIAL'
                    return True
                else:
                    print("   ❌ Frontend loads but content doesn't match PyGent Factory")
                    self.results['frontend'] = 'WRONG_CONTENT'
                    return False
            else:
                print(f"   ❌ Frontend returned status {response.status_code}")
                self.results['frontend'] = f'HTTP_{response.status_code}'
                return False
                
        except Exception as e:
            print(f"   ❌ Frontend test failed: {e}")
            self.results['frontend'] = f'ERROR: {str(e)}'
            return False

    def test_api_connectivity(self):
        """Test API endpoints"""
        print("🔌 Testing API Connectivity...")
        try:
            # Test health endpoint
            response = requests.get(f"{self.api_url}/api/v1/health", timeout=30)
            
            if response.status_code == 200:
                health_data = response.json()
                print("   ✅ API health endpoint working")
                print(f"   📊 Backend status: {health_data.get('status', 'unknown')}")
                self.results['api'] = 'SUCCESS'
                return True
            else:
                print(f"   ❌ API returned status {response.status_code}")
                self.results['api'] = f'HTTP_{response.status_code}'
                return False
                
        except Exception as e:
            print(f"   ❌ API test failed: {e}")
            self.results['api'] = f'ERROR: {str(e)}'
            return False

    async def test_websocket_connectivity(self):
        """Test WebSocket connectivity"""
        print("🔗 Testing WebSocket Connectivity...")
        try:
            async with websockets.connect(f"{self.ws_url}/ws") as websocket:
                print("   ✅ WebSocket connection established")
                
                # Send test message
                test_message = {
                    "type": "ping",
                    "data": {"message": "Deployment test"},
                    "timestamp": datetime.now().isoformat()
                }
                
                await websocket.send(json.dumps(test_message))
                print("   📤 Test message sent")
                
                # Wait for response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    print(f"   📨 Response received: {response[:100]}...")
                    self.results['websocket'] = 'SUCCESS'
                    return True
                except asyncio.TimeoutError:
                    print("   ⏰ No response (connection working, no echo)")
                    self.results['websocket'] = 'NO_ECHO'
                    return True
                    
        except Exception as e:
            print(f"   ❌ WebSocket test failed: {e}")
            self.results['websocket'] = f'ERROR: {str(e)}'
            return False

    def test_environment_variables(self):
        """Test if environment variables are configured correctly"""
        print("⚙️  Testing Environment Configuration...")
        
        # This would require checking the built frontend files
        # For now, we'll check if the API calls are going to the right endpoints
        try:
            # Check if frontend is trying to connect to production URLs
            response = requests.get(self.frontend_url, timeout=30)
            if response.status_code == 200:
                content = response.text
                
                # Look for production API URLs in the built files
                if "api.timpayne.net" in content:
                    print("   ✅ Production API URL found in frontend")
                    self.results['env_vars'] = 'SUCCESS'
                    return True
                else:
                    print("   ⚠️  Production API URL not found in frontend")
                    self.results['env_vars'] = 'PARTIAL'
                    return False
            else:
                print("   ❌ Cannot check environment variables (frontend not loading)")
                self.results['env_vars'] = 'CANNOT_CHECK'
                return False
                
        except Exception as e:
            print(f"   ❌ Environment test failed: {e}")
            self.results['env_vars'] = f'ERROR: {str(e)}'
            return False

    async def run_complete_test(self):
        """Run all deployment tests"""
        print("🚀 PyGent Factory Complete Deployment Test")
        print("=" * 60)
        print(f"🕐 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Run tests
        frontend_ok = self.test_frontend_loading()
        print()
        
        api_ok = self.test_api_connectivity()
        print()
        
        websocket_ok = await self.test_websocket_connectivity()
        print()
        
        env_ok = self.test_environment_variables()
        print()
        
        # Summary
        print("=" * 60)
        print("📊 DEPLOYMENT TEST SUMMARY")
        print("=" * 60)
        
        total_tests = 4
        passed_tests = sum([frontend_ok, api_ok, websocket_ok, env_ok])
        
        print(f"Frontend Loading:     {'✅ PASS' if frontend_ok else '❌ FAIL'}")
        print(f"API Connectivity:     {'✅ PASS' if api_ok else '❌ FAIL'}")
        print(f"WebSocket Connection: {'✅ PASS' if websocket_ok else '❌ FAIL'}")
        print(f"Environment Config:   {'✅ PASS' if env_ok else '❌ FAIL'}")
        print()
        print(f"Overall Score: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 DEPLOYMENT FULLY SUCCESSFUL!")
            print("   PyGent Factory is ready for production use!")
        elif passed_tests >= 3:
            print("⚠️  DEPLOYMENT MOSTLY SUCCESSFUL")
            print("   Minor issues detected, but core functionality working")
        elif passed_tests >= 2:
            print("🔧 DEPLOYMENT PARTIALLY SUCCESSFUL")
            print("   Significant issues detected, troubleshooting needed")
        else:
            print("❌ DEPLOYMENT FAILED")
            print("   Major issues detected, configuration review required")
        
        print()
        print("📋 Detailed Results:")
        for component, result in self.results.items():
            print(f"   {component}: {result}")
        
        return passed_tests == total_tests

async def main():
    """Main test function"""
    tester = DeploymentTester()
    success = await tester.run_complete_test()
    return success

if __name__ == "__main__":
    asyncio.run(main())
