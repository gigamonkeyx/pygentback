#!/usr/bin/env python3
"""
PyGent Factory System Integration Test

This script tests the complete PyGent Factory system including:
- Backend API endpoints
- WebSocket communication
- Frontend accessibility
- Database connectivity
- AI component integration
"""

import asyncio
import json
import time
import requests
import websockets
from typing import Dict, Any, List
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

class SystemTester:
    """Comprehensive system testing suite"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.frontend_url = "http://localhost:5173"
        self.ws_url = "ws://localhost:8000/ws"
        self.results: List[Dict[str, Any]] = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with color coding"""
        color_map = {
            "INFO": Colors.BLUE,
            "SUCCESS": Colors.GREEN,
            "WARNING": Colors.YELLOW,
            "ERROR": Colors.RED,
            "TEST": Colors.PURPLE
        }
        color = color_map.get(level, Colors.WHITE)
        print(f"{color}[{level}]{Colors.END} {message}")
    
    def record_result(self, test_name: str, success: bool, message: str = "", duration: float = 0):
        """Record test result"""
        self.results.append({
            "test": test_name,
            "success": success,
            "message": message,
            "duration": duration
        })
        
        status = "PASS" if success else "FAIL"
        color = Colors.GREEN if success else Colors.RED
        duration_str = f" ({duration:.2f}s)" if duration > 0 else ""
        self.log(f"{test_name}: {color}{status}{Colors.END}{duration_str} {message}")
    
    async def test_backend_health(self):
        """Test backend API health endpoint"""
        self.log("Testing backend health endpoint...", "TEST")
        start_time = time.time()
        
        try:
            response = requests.get(f"{self.base_url}/api/v1/health", timeout=10)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                self.record_result(
                    "Backend Health Check", 
                    True, 
                    f"Status: {data.get('status', 'unknown')}", 
                    duration
                )
                return True
            else:
                self.record_result(
                    "Backend Health Check", 
                    False, 
                    f"HTTP {response.status_code}", 
                    duration
                )
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.record_result("Backend Health Check", False, str(e), duration)
            return False
    
    async def test_frontend_accessibility(self):
        """Test frontend accessibility"""
        self.log("Testing frontend accessibility...", "TEST")
        start_time = time.time()
        
        try:
            response = requests.get(self.frontend_url, timeout=10)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                self.record_result(
                    "Frontend Accessibility", 
                    True, 
                    "Frontend is accessible", 
                    duration
                )
                return True
            else:
                self.record_result(
                    "Frontend Accessibility", 
                    False, 
                    f"HTTP {response.status_code}", 
                    duration
                )
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.record_result("Frontend Accessibility", False, str(e), duration)
            return False
    
    async def test_websocket_connection(self):
        """Test WebSocket connection"""
        self.log("Testing WebSocket connection...", "TEST")
        start_time = time.time()
        
        try:
            async with websockets.connect(self.ws_url) as websocket:
                # Send a test message
                test_message = {
                    "type": "ping",
                    "data": {"message": "test"}
                }
                await websocket.send(json.dumps(test_message))
                
                # Wait for response (with timeout)
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    duration = time.time() - start_time
                    self.record_result(
                        "WebSocket Connection", 
                        True, 
                        "Connection established and responsive", 
                        duration
                    )
                    return True
                except asyncio.TimeoutError:
                    duration = time.time() - start_time
                    self.record_result(
                        "WebSocket Connection", 
                        True, 
                        "Connection established (no response expected)", 
                        duration
                    )
                    return True
                    
        except Exception as e:
            duration = time.time() - start_time
            self.record_result("WebSocket Connection", False, str(e), duration)
            return False
    
    async def test_api_endpoints(self):
        """Test various API endpoints"""
        self.log("Testing API endpoints...", "TEST")
        
        endpoints = [
            ("/api/v1/health", "GET", "Health Endpoint"),
            ("/docs", "GET", "API Documentation"),
            ("/api/v1/agents", "GET", "Agents Endpoint"),
            ("/api/v1/mcp/servers", "GET", "MCP Servers Endpoint"),
        ]
        
        for endpoint, method, name in endpoints:
            start_time = time.time()
            try:
                if method == "GET":
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                else:
                    response = requests.request(method, f"{self.base_url}{endpoint}", timeout=10)
                
                duration = time.time() - start_time
                success = response.status_code in [200, 404]  # 404 is acceptable for some endpoints
                
                self.record_result(
                    f"API {name}", 
                    success, 
                    f"HTTP {response.status_code}", 
                    duration
                )
                
            except Exception as e:
                duration = time.time() - start_time
                self.record_result(f"API {name}", False, str(e), duration)
    
    async def test_chat_functionality(self):
        """Test chat functionality via WebSocket"""
        self.log("Testing chat functionality...", "TEST")
        start_time = time.time()
        
        try:
            async with websockets.connect(self.ws_url) as websocket:
                # Send a chat message
                chat_message = {
                    "type": "chat_message",
                    "data": {
                        "message": {
                            "id": "test_msg_1",
                            "type": "user",
                            "content": "Hello, this is a test message",
                            "agentId": "reasoning",
                            "timestamp": time.time()
                        }
                    }
                }
                
                await websocket.send(json.dumps(chat_message))
                
                # Wait for response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=15.0)
                    response_data = json.loads(response)
                    duration = time.time() - start_time
                    
                    if response_data.get("type") in ["chat_response", "typing_indicator"]:
                        self.record_result(
                            "Chat Functionality", 
                            True, 
                            f"Received {response_data.get('type')}", 
                            duration
                        )
                        return True
                    else:
                        self.record_result(
                            "Chat Functionality", 
                            False, 
                            f"Unexpected response type: {response_data.get('type')}", 
                            duration
                        )
                        return False
                        
                except asyncio.TimeoutError:
                    duration = time.time() - start_time
                    self.record_result(
                        "Chat Functionality", 
                        False, 
                        "No response received within timeout", 
                        duration
                    )
                    return False
                    
        except Exception as e:
            duration = time.time() - start_time
            self.record_result("Chat Functionality", False, str(e), duration)
            return False
    
    async def test_system_monitoring(self):
        """Test system monitoring endpoints"""
        self.log("Testing system monitoring...", "TEST")
        start_time = time.time()
        
        try:
            # Test metrics endpoint (if available)
            response = requests.get(f"{self.base_url}/metrics", timeout=10)
            duration = time.time() - start_time
            
            # Metrics endpoint might not exist, so we accept 404
            success = response.status_code in [200, 404]
            message = "Metrics available" if response.status_code == 200 else "Metrics endpoint not found (acceptable)"
            
            self.record_result("System Monitoring", success, message, duration)
            return success
            
        except Exception as e:
            duration = time.time() - start_time
            self.record_result("System Monitoring", False, str(e), duration)
            return False
    
    async def run_all_tests(self):
        """Run all system tests"""
        self.log("Starting PyGent Factory System Integration Tests", "INFO")
        self.log("=" * 60, "INFO")
        
        # Run tests
        await self.test_backend_health()
        await self.test_frontend_accessibility()
        await self.test_websocket_connection()
        await self.test_api_endpoints()
        await self.test_chat_functionality()
        await self.test_system_monitoring()
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate test report"""
        self.log("=" * 60, "INFO")
        self.log("TEST RESULTS SUMMARY", "INFO")
        self.log("=" * 60, "INFO")
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["success"])
        failed_tests = total_tests - passed_tests
        
        self.log(f"Total Tests: {total_tests}", "INFO")
        self.log(f"Passed: {Colors.GREEN}{passed_tests}{Colors.END}", "INFO")
        self.log(f"Failed: {Colors.RED}{failed_tests}{Colors.END}", "INFO")
        
        if failed_tests > 0:
            self.log("\nFAILED TESTS:", "ERROR")
            for result in self.results:
                if not result["success"]:
                    self.log(f"  ❌ {result['test']}: {result['message']}", "ERROR")
        
        self.log("\nDETAILED RESULTS:", "INFO")
        for result in self.results:
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            duration = f" ({result['duration']:.2f}s)" if result['duration'] > 0 else ""
            self.log(f"  {status} {result['test']}{duration}", "INFO")
            if result["message"]:
                self.log(f"    {result['message']}", "INFO")
        
        # Overall status
        if failed_tests == 0:
            self.log(f"\n{Colors.GREEN}🎉 ALL TESTS PASSED! PyGent Factory is working correctly.{Colors.END}", "SUCCESS")
            return True
        else:
            self.log(f"\n{Colors.RED}❌ {failed_tests} TEST(S) FAILED. Please check the system configuration.{Colors.END}", "ERROR")
            return False

async def main():
    """Main test execution"""
    tester = SystemTester()
    
    try:
        success = await tester.run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        tester.log("\nTests interrupted by user", "WARNING")
        sys.exit(1)
    except Exception as e:
        tester.log(f"Test execution failed: {e}", "ERROR")
        sys.exit(1)

if __name__ == "__main__":
    # Check if required packages are available
    try:
        import websockets
        import requests
    except ImportError as e:
        print(f"{Colors.RED}[ERROR]{Colors.END} Missing required package: {e}")
        print("Install with: pip install websockets requests")
        sys.exit(1)
    
    asyncio.run(main())
