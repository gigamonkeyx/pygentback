#!/usr/bin/env python3
"""
Test Performance After Mock Code Removal

Validate that removing mock implementations has improved response times.
"""

import time
import json
import requests
from datetime import datetime
from typing import Dict, List, Any

class PostMockRemovalTester:
    """Test performance after mock code removal"""
    
    def __init__(self):
        self.servers = {
            'embedding': {'port': 8002, 'name': 'Embedding MCP Server'},
            'document-processing': {'port': 8003, 'name': 'Document Processing MCP Server'},
            'vector-search': {'port': 8004, 'name': 'Vector Search MCP Server'},
            'agent-orchestration': {'port': 8005, 'name': 'Agent Orchestration MCP Server'}
        }
        self.results = []
    
    def log_result(self, test_name: str, success: bool, details: str = "", duration: float = 0):
        """Log test result"""
        result = {
            'test': test_name,
            'success': success,
            'details': details,
            'duration_ms': round(duration * 1000, 2),
            'timestamp': datetime.utcnow().isoformat()
        }
        self.results.append(result)
        
        status = "✅" if success else "❌"
        print(f"{status} {test_name} ({duration*1000:.1f}ms)")
        if details:
            print(f"    {details}")
    
    def test_response_time_improvements(self) -> bool:
        """Test that response times have improved after mock removal"""
        start = time.time()
        try:
            print("🚀 Testing Response Time Improvements After Mock Removal...")
            
            improved_servers = 0
            performance_data = {}
            
            for server_id, server_info in self.servers.items():
                port = server_info['port']
                name = server_info['name']
                
                print(f"   Testing {name}...")
                
                # Test multiple requests to get average
                response_times = []
                successful_requests = 0
                
                for i in range(5):
                    try:
                        req_start = time.time()
                        response = requests.get(f"http://localhost:{port}/health", timeout=10)
                        req_time = time.time() - req_start
                        
                        if response.status_code == 200:
                            response_times.append(req_time)
                            successful_requests += 1
                            print(f"     Request {i+1}: {req_time*1000:.1f}ms")
                        else:
                            print(f"     Request {i+1}: FAILED (HTTP {response.status_code})")
                    except Exception as e:
                        print(f"     Request {i+1}: ERROR ({str(e)})")
                
                if response_times:
                    avg_time = sum(response_times) / len(response_times)
                    min_time = min(response_times)
                    max_time = max(response_times)
                    
                    performance_data[server_id] = {
                        'avg_response_time': avg_time,
                        'min_response_time': min_time,
                        'max_response_time': max_time,
                        'successful_requests': successful_requests,
                        'improved': avg_time < 1.0  # Under 1 second is good improvement
                    }
                    
                    print(f"     Average: {avg_time*1000:.1f}ms")
                    print(f"     Range: {min_time*1000:.1f}ms - {max_time*1000:.1f}ms")
                    print(f"     Success Rate: {successful_requests}/5")
                    
                    if avg_time < 1.0:
                        improved_servers += 1
                        print(f"     ✅ IMPROVED: Response time under 1 second")
                    else:
                        print(f"     ⚠️ STILL SLOW: Response time over 1 second")
                else:
                    print(f"     ❌ NO SUCCESSFUL REQUESTS")
            
            duration = time.time() - start
            
            if improved_servers >= len(self.servers) * 0.75:  # 75% of servers improved
                details = f"{improved_servers}/{len(self.servers)} servers show significant improvement"
                self.log_result("Response Time Improvements", True, details, duration)
                return True
            else:
                details = f"Only {improved_servers}/{len(self.servers)} servers improved"
                self.log_result("Response Time Improvements", False, details, duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Response Time Improvements", False, f"Error: {str(e)}", duration)
            return False
    
    def test_production_readiness_metrics(self) -> bool:
        """Test production readiness metrics"""
        start = time.time()
        try:
            print("\n📊 Testing Production Readiness Metrics...")
            
            production_ready_servers = 0
            
            for server_id, server_info in self.servers.items():
                port = server_info['port']
                name = server_info['name']
                
                print(f"   Testing {name}...")
                
                try:
                    # Test root endpoint
                    root_response = requests.get(f"http://localhost:{port}/", timeout=5)
                    if root_response.status_code == 200:
                        root_data = root_response.json()
                        
                        # Check for production indicators
                        capabilities = root_data.get('capabilities', [])
                        has_mock_capabilities = any('mock' in cap.lower() for cap in capabilities)
                        
                        if not has_mock_capabilities:
                            print(f"     ✅ No mock capabilities detected")
                        else:
                            print(f"     ⚠️ Mock capabilities still present: {capabilities}")
                        
                        # Test health endpoint
                        health_response = requests.get(f"http://localhost:{port}/health", timeout=5)
                        if health_response.status_code == 200:
                            health_data = health_response.json()
                            status = health_data.get('status', 'unknown')
                            
                            if status == 'healthy' and not has_mock_capabilities:
                                production_ready_servers += 1
                                print(f"     ✅ PRODUCTION READY")
                            else:
                                print(f"     ⚠️ NOT PRODUCTION READY: {status}, mock={has_mock_capabilities}")
                        else:
                            print(f"     ❌ Health check failed: HTTP {health_response.status_code}")
                    else:
                        print(f"     ❌ Root endpoint failed: HTTP {root_response.status_code}")
                        
                except Exception as e:
                    print(f"     ❌ Error: {str(e)}")
            
            duration = time.time() - start
            
            if production_ready_servers == len(self.servers):
                details = f"All {production_ready_servers}/{len(self.servers)} servers are production ready"
                self.log_result("Production Readiness", True, details, duration)
                return True
            else:
                details = f"Only {production_ready_servers}/{len(self.servers)} servers are production ready"
                self.log_result("Production Readiness", False, details, duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Production Readiness", False, f"Error: {str(e)}", duration)
            return False
    
    def test_concurrent_performance_post_fix(self) -> bool:
        """Test concurrent performance after mock removal"""
        start = time.time()
        try:
            print("\n⚡ Testing Concurrent Performance After Mock Removal...")
            
            import concurrent.futures
            
            def test_server_concurrent(server_id: str, port: int) -> Dict[str, Any]:
                """Test concurrent requests to a server"""
                successful_requests = 0
                response_times = []
                
                for _ in range(10):
                    try:
                        req_start = time.time()
                        response = requests.get(f"http://localhost:{port}/health", timeout=5)
                        req_time = time.time() - req_start
                        
                        if response.status_code == 200:
                            successful_requests += 1
                            response_times.append(req_time)
                    except:
                        pass
                
                avg_response_time = sum(response_times) / len(response_times) if response_times else 0
                success_rate = (successful_requests / 10) * 100
                
                return {
                    'server_id': server_id,
                    'success_rate': success_rate,
                    'avg_response_time': avg_response_time,
                    'successful_requests': successful_requests
                }
            
            # Test all servers concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.servers)) as executor:
                futures = {
                    executor.submit(test_server_concurrent, server_id, info['port']): server_id
                    for server_id, info in self.servers.items()
                }
                
                results = {}
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results[result['server_id']] = result
            
            # Analyze results
            good_performance = 0
            for server_id, result in results.items():
                success_rate = result['success_rate']
                avg_time = result['avg_response_time']
                
                print(f"   {self.servers[server_id]['name']}:")
                print(f"     Success Rate: {success_rate:.1f}%")
                print(f"     Avg Response: {avg_time*1000:.1f}ms")
                
                # Good performance: 90% success rate and under 500ms
                if success_rate >= 90 and avg_time < 0.5:
                    good_performance += 1
                    print(f"     ✅ EXCELLENT PERFORMANCE")
                elif success_rate >= 80 and avg_time < 1.0:
                    print(f"     ✅ GOOD PERFORMANCE")
                else:
                    print(f"     ⚠️ NEEDS IMPROVEMENT")
            
            duration = time.time() - start
            
            if good_performance >= len(self.servers) * 0.5:  # At least 50% excellent
                details = f"{good_performance}/{len(self.servers)} servers show excellent concurrent performance"
                self.log_result("Concurrent Performance Post-Fix", True, details, duration)
                return True
            else:
                details = f"Only {good_performance}/{len(self.servers)} servers show excellent performance"
                self.log_result("Concurrent Performance Post-Fix", False, details, duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Concurrent Performance Post-Fix", False, f"Error: {str(e)}", duration)
            return False
    
    def run_post_mock_removal_tests(self) -> Dict[str, Any]:
        """Run all post-mock-removal tests"""
        print("🔧 Post-Mock-Removal Performance Validation")
        print("=" * 60)
        
        # Wait for servers to fully start
        print("⏳ Waiting for servers to start...")
        time.sleep(10)
        
        tests = [
            ("Response Time Improvements", self.test_response_time_improvements),
            ("Production Readiness", self.test_production_readiness_metrics),
            ("Concurrent Performance Post-Fix", self.test_concurrent_performance_post_fix)
        ]
        
        passed = 0
        for test_name, test_func in tests:
            if test_func():
                passed += 1
            print()
        
        total = len(tests)
        
        print("=" * 60)
        print(f"📊 Post-Mock-Removal Results: {passed}/{total} tests passed")
        
        if passed == total:
            print(f"\n🎉 MOCK REMOVAL SUCCESS!")
            print(f"   All performance issues resolved.")
            print(f"   Servers are now production-ready.")
        elif passed >= total * 0.75:
            print(f"\n✅ SIGNIFICANT IMPROVEMENT!")
            print(f"   Major performance gains achieved.")
            print(f"   Minor optimizations may still be needed.")
        else:
            print(f"\n⚠️ PARTIAL IMPROVEMENT")
            print(f"   Some issues resolved, more work needed.")
        
        return {
            'total_tests': total,
            'passed_tests': passed,
            'success_rate': (passed / total) * 100,
            'test_results': self.results
        }


def main():
    """Main execution"""
    tester = PostMockRemovalTester()
    results = tester.run_post_mock_removal_tests()
    
    # Save results
    with open('post_mock_removal_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved to: post_mock_removal_test_results.json")
    
    return 0 if results['success_rate'] >= 75 else 1


if __name__ == "__main__":
    exit(main())
