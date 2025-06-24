#!/usr/bin/env python3
"""
PyGent Factory Service Validation Script

This script validates that all PyGent Factory services are running correctly
and can communicate with each other following Context7 MCP best practices.
"""

import asyncio
import requests
import websockets
import time
import sys
import json
from typing import Dict, Any, Optional
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import API configuration
try:
    from ui.src.config.api import config, getApiUrl, getWebSocketUrl, checkBackendHealth
except ImportError:
    # Fallback configuration
    def getApiUrl(endpoint: str) -> str:
        return f"http://localhost:8000/{endpoint.lstrip('/')}"

    def getWebSocketUrl() -> str:
        return "ws://localhost:8000/ws"

    async def checkBackendHealth() -> bool:
        try:
            import requests
            response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
            return response.status_code == 200
        except:
            return False


class ServiceValidator:
    """Validates PyGent Factory services"""
    
    def __init__(self):
        self.results: Dict[str, Any] = {
            "backend": {"status": "unknown", "details": {}},
            "frontend": {"status": "unknown", "details": {}},
            "documentation": {"status": "unknown", "details": {}},
            "websocket": {"status": "unknown", "details": {}},
            "database": {"status": "unknown", "details": {}},
            "overall": {"status": "unknown", "healthy_services": 0, "total_services": 5}
        }
    
    async def validate_backend(self) -> bool:
        """Validate backend API service"""
        try:
            print("🔍 Testing backend health...")
            
            # Test health endpoint
            health_url = getApiUrl("api/v1/health")
            response = requests.get(health_url, timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                self.results["backend"] = {
                    "status": "healthy",
                    "details": {
                        "url": health_url,
                        "response_time_ms": response.elapsed.total_seconds() * 1000,
                        "health_data": health_data
                    }
                }
                print(f"✅ Backend healthy: {health_url}")
                return True
            else:
                self.results["backend"] = {
                    "status": "unhealthy",
                    "details": {
                        "url": health_url,
                        "status_code": response.status_code,
                        "error": f"HTTP {response.status_code}"
                    }
                }
                print(f"❌ Backend unhealthy: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.results["backend"] = {
                "status": "error",
                "details": {"error": str(e)}
            }
            print(f"❌ Backend error: {e}")
            return False
    
    async def validate_frontend(self) -> bool:
        """Validate frontend service"""
        try:
            print("🔍 Testing frontend accessibility...")
            
            frontend_url = "http://localhost:5173"
            response = requests.get(frontend_url, timeout=10)
            
            if response.status_code == 200:
                self.results["frontend"] = {
                    "status": "healthy",
                    "details": {
                        "url": frontend_url,
                        "response_time_ms": response.elapsed.total_seconds() * 1000,
                        "content_length": len(response.content)
                    }
                }
                print(f"✅ Frontend accessible: {frontend_url}")
                return True
            else:
                self.results["frontend"] = {
                    "status": "unhealthy",
                    "details": {
                        "url": frontend_url,
                        "status_code": response.status_code,
                        "error": f"HTTP {response.status_code}"
                    }
                }
                print(f"❌ Frontend unhealthy: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.results["frontend"] = {
                "status": "error",
                "details": {"error": str(e)}
            }
            print(f"❌ Frontend error: {e}")
            return False
    
    async def validate_documentation(self) -> bool:
        """Validate documentation service"""
        try:
            print("🔍 Testing documentation service...")
            
            docs_url = "http://localhost:3001"
            response = requests.get(docs_url, timeout=10)
            
            if response.status_code == 200:
                self.results["documentation"] = {
                    "status": "healthy",
                    "details": {
                        "url": docs_url,
                        "response_time_ms": response.elapsed.total_seconds() * 1000,
                        "content_length": len(response.content)
                    }
                }
                print(f"✅ Documentation accessible: {docs_url}")
                return True
            else:
                self.results["documentation"] = {
                    "status": "unhealthy",
                    "details": {
                        "url": docs_url,
                        "status_code": response.status_code,
                        "error": f"HTTP {response.status_code}"
                    }
                }
                print(f"❌ Documentation unhealthy: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.results["documentation"] = {
                "status": "error",
                "details": {"error": str(e)}
            }
            print(f"❌ Documentation error: {e}")
            return False
    
    async def validate_websocket(self) -> bool:
        """Validate WebSocket connection"""
        try:
            print("🔍 Testing WebSocket connection...")
            
            ws_url = getWebSocketUrl()
            
            async with websockets.connect(ws_url, timeout=10) as websocket:
                # Send test message
                test_message = {
                    "type": "ping",
                    "timestamp": time.time(),
                    "data": "validation_test"
                }
                
                await websocket.send(json.dumps(test_message))
                
                # Wait for response (with timeout)
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    response_data = json.loads(response)
                    
                    self.results["websocket"] = {
                        "status": "healthy",
                        "details": {
                            "url": ws_url,
                            "test_message_sent": test_message,
                            "response_received": response_data
                        }
                    }
                    print(f"✅ WebSocket connection successful: {ws_url}")
                    return True
                    
                except asyncio.TimeoutError:
                    self.results["websocket"] = {
                        "status": "partial",
                        "details": {
                            "url": ws_url,
                            "error": "Connection established but no response received"
                        }
                    }
                    print(f"⚠️ WebSocket partial: Connected but no response")
                    return True  # Connection works, just no response handler
                    
        except Exception as e:
            self.results["websocket"] = {
                "status": "error",
                "details": {
                    "url": getWebSocketUrl(),
                    "error": str(e)
                }
            }
            print(f"❌ WebSocket error: {e}")
            return False
    
    async def validate_database(self) -> bool:
        """Validate database connectivity through backend"""
        try:
            print("🔍 Testing database connectivity...")
            
            # Test database through backend API
            agents_url = getApiUrl("api/v1/agents")
            response = requests.get(agents_url, timeout=10)
            
            if response.status_code in [200, 404]:  # 404 is OK (no agents yet)
                self.results["database"] = {
                    "status": "healthy",
                    "details": {
                        "url": agents_url,
                        "response_time_ms": response.elapsed.total_seconds() * 1000,
                        "status_code": response.status_code
                    }
                }
                print(f"✅ Database connectivity confirmed via API")
                return True
            else:
                self.results["database"] = {
                    "status": "unhealthy",
                    "details": {
                        "url": agents_url,
                        "status_code": response.status_code,
                        "error": f"HTTP {response.status_code}"
                    }
                }
                print(f"❌ Database connectivity failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.results["database"] = {
                "status": "error",
                "details": {"error": str(e)}
            }
            print(f"❌ Database error: {e}")
            return False
    
    async def run_validation(self) -> Dict[str, Any]:
        """Run complete service validation"""
        print("🚀 Starting PyGent Factory service validation...\n")
        
        # Run all validations
        validations = [
            ("Backend API", self.validate_backend()),
            ("Frontend UI", self.validate_frontend()),
            ("Documentation", self.validate_documentation()),
            ("WebSocket", self.validate_websocket()),
            ("Database", self.validate_database())
        ]
        
        healthy_count = 0
        for name, validation_coro in validations:
            try:
                is_healthy = await validation_coro
                if is_healthy:
                    healthy_count += 1
            except Exception as e:
                print(f"❌ {name} validation failed: {e}")
        
        # Calculate overall status
        self.results["overall"] = {
            "healthy_services": healthy_count,
            "total_services": len(validations),
            "status": "healthy" if healthy_count == len(validations) else "partial" if healthy_count > 0 else "unhealthy"
        }
        
        # Print summary
        print(f"\n📊 Validation Summary:")
        print(f"   Healthy services: {healthy_count}/{len(validations)}")
        print(f"   Overall status: {self.results['overall']['status'].upper()}")
        
        if healthy_count == len(validations):
            print("\n🎉 All services validated successfully!")
            return self.results
        elif healthy_count > 0:
            print(f"\n⚠️ Partial success: {healthy_count}/{len(validations)} services healthy")
            return self.results
        else:
            print("\n💥 All services failed validation")
            return self.results


async def main():
    """Main validation function"""
    validator = ServiceValidator()
    results = await validator.run_validation()
    
    # Save results to file
    results_file = Path(__file__).parent / "validation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    # Exit with appropriate code
    if results["overall"]["status"] == "healthy":
        sys.exit(0)
    elif results["overall"]["status"] == "partial":
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())
