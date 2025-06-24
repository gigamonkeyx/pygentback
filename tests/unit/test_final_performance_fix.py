#!/usr/bin/env python3
"""
Test Final Performance Fix

Test performance after implementing fast embedding provider.
"""

import time
import json
import requests
from datetime import datetime
from typing import Dict, List, Any

class FinalPerformanceTester:
    """Test final performance after fast embedding implementation"""
    
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
    
    def test_fast_response_times(self) -> bool:
        """Test that response times are now fast (<500ms)"""
        start = time.time()
        try:
            print("🚀 Testing Fast Response Times After SentenceTransformers Removal...")
            
            fast_servers = 0
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
                        'fast': avg_time < 0.5  # Under 500ms is fast
                    }
                    
                    print(f"     Average: {avg_time*1000:.1f}ms")
                    print(f"     Range: {min_time*1000:.1f}ms - {max_time*1000:.1f}ms")
                    print(f"     Success Rate: {successful_requests}/5")
                    
                    if avg_time < 0.5:
                        fast_servers += 1
                        print(f"     ✅ FAST: Response time under 500ms")
                    elif avg_time < 1.0:
                        print(f"     ⚠️ ACCEPTABLE: Response time under 1 second")
                    else:
                        print(f"     ❌ SLOW: Response time over 1 second")
                else:
                    print(f"     ❌ NO SUCCESSFUL REQUESTS")
            
            duration = time.time() - start
            
            if fast_servers >= len(self.servers) * 0.75:  # 75% of servers fast
                details = f"{fast_servers}/{len(self.servers)} servers are fast (<500ms)"
                self.log_result("Fast Response Times", True, details, duration)
                return True
            else:
                details = f"Only {fast_servers}/{len(self.servers)} servers are fast"
                self.log_result("Fast Response Times", False, details, duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Fast Response Times", False, f"Error: {str(e)}", duration)
            return False
    
    def test_embedding_service_performance(self) -> bool:
        """Test embedding service specific performance"""
        start = time.time()
        try:
            print("\n🧠 Testing Embedding Service Performance...")
            
            # Test embedding generation endpoint
            test_data = {
                "input": "This is a test embedding request",
                "model": "fast-deterministic-v1"
            }
            
            embedding_times = []
            successful_embeddings = 0
            
            for i in range(3):
                try:
                    req_start = time.time()
                    response = requests.post(
                        "http://localhost:8002/v1/embeddings",
                        json=test_data,
                        timeout=10
                    )
                    req_time = time.time() - req_start
                    
                    if response.status_code == 200:
                        embedding_times.append(req_time)
                        successful_embeddings += 1
                        
                        # Check response structure
                        data = response.json()
                        if 'data' in data and len(data['data']) > 0:
                            embedding = data['data'][0].get('embedding', [])
                            print(f"     Embedding {i+1}: {req_time*1000:.1f}ms (dim: {len(embedding)})")
                        else:
                            print(f"     Embedding {i+1}: {req_time*1000:.1f}ms (invalid response)")
                    else:
                        print(f"     Embedding {i+1}: FAILED (HTTP {response.status_code})")
                        
                except Exception as e:
                    print(f"     Embedding {i+1}: ERROR ({str(e)})")
            
            duration = time.time() - start
            
            if embedding_times:
                avg_time = sum(embedding_times) / len(embedding_times)
                
                if avg_time < 0.1 and successful_embeddings == 3:  # Under 100ms and all successful
                    details = f"Excellent embedding performance: {avg_time*1000:.1f}ms avg, {successful_embeddings}/3 successful"
                    self.log_result("Embedding Service Performance", True, details, duration)
                    return True
                else:
                    details = f"Embedding performance: {avg_time*1000:.1f}ms avg, {successful_embeddings}/3 successful"
                    self.log_result("Embedding Service Performance", False, details, duration)
                    return False
            else:
                details = "No successful embedding requests"
                self.log_result("Embedding Service Performance", False, details, duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Embedding Service Performance", False, f"Error: {str(e)}", duration)
            return False
    
    def test_production_readiness_final(self) -> bool:
        """Final production readiness test"""
        start = time.time()
        try:
            print("\n🎯 Final Production Readiness Test...")
            
            production_ready_count = 0
            
            for server_id, server_info in self.servers.items():
                port = server_info['port']
                name = server_info['name']
                
                print(f"   Testing {name}...")
                
                try:
                    # Test root endpoint
                    root_response = requests.get(f"http://localhost:{port}/", timeout=5)
                    if root_response.status_code == 200:
                        root_data = root_response.json()
                        
                        # Test health endpoint with timing
                        health_start = time.time()
                        health_response = requests.get(f"http://localhost:{port}/health", timeout=5)
                        health_time = time.time() - health_start
                        
                        if health_response.status_code == 200:
                            health_data = health_response.json()
                            status = health_data.get('status', 'unknown')
                            
                            # Check production readiness criteria
                            is_healthy = status == 'healthy'
                            is_fast = health_time < 0.5  # Under 500ms
                            has_no_mock = not any('mock' in cap.lower() for cap in root_data.get('capabilities', []))
                            
                            if is_healthy and is_fast and has_no_mock:
                                production_ready_count += 1
                                print(f"     ✅ PRODUCTION READY: {status}, {health_time*1000:.1f}ms, no mock code")
                            else:
                                print(f"     ⚠️ NOT READY: healthy={is_healthy}, fast={is_fast}, no_mock={has_no_mock}")
                        else:
                            print(f"     ❌ Health check failed: HTTP {health_response.status_code}")
                    else:
                        print(f"     ❌ Root endpoint failed: HTTP {root_response.status_code}")
                        
                except Exception as e:
                    print(f"     ❌ Error: {str(e)}")
            
            duration = time.time() - start
            
            if production_ready_count == len(self.servers):
                details = f"All {production_ready_count}/{len(self.servers)} servers are production ready"
                self.log_result("Final Production Readiness", True, details, duration)
                return True
            else:
                details = f"Only {production_ready_count}/{len(self.servers)} servers are production ready"
                self.log_result("Final Production Readiness", False, details, duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Final Production Readiness", False, f"Error: {str(e)}", duration)
            return False
    
    def run_final_performance_tests(self) -> Dict[str, Any]:
        """Run final performance tests"""
        print("🎯 Final Performance Test Suite")
        print("=" * 60)
        
        # Wait for servers to fully start
        print("⏳ Waiting for servers to start...")
        time.sleep(5)
        
        tests = [
            ("Fast Response Times", self.test_fast_response_times),
            ("Embedding Service Performance", self.test_embedding_service_performance),
            ("Final Production Readiness", self.test_production_readiness_final)
        ]
        
        passed = 0
        for test_name, test_func in tests:
            if test_func():
                passed += 1
            print()
        
        total = len(tests)
        
        print("=" * 60)
        print(f"📊 Final Performance Results: {passed}/{total} tests passed")
        
        if passed == total:
            print(f"\n🎉 PERFORMANCE ISSUE RESOLVED!")
            print(f"   All servers are now fast and production-ready.")
            print(f"   The SentenceTransformers bottleneck has been eliminated.")
        elif passed >= total * 0.75:
            print(f"\n✅ MAJOR IMPROVEMENT!")
            print(f"   Significant performance gains achieved.")
            print(f"   Ready for production deployment.")
        else:
            print(f"\n⚠️ PARTIAL IMPROVEMENT")
            print(f"   Some performance issues may remain.")
        
        return {
            'total_tests': total,
            'passed_tests': passed,
            'success_rate': (passed / total) * 100,
            'test_results': self.results
        }


def main():
    """Main execution"""
    tester = FinalPerformanceTester()
    results = tester.run_final_performance_tests()
    
    # Save results
    with open('final_performance_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved to: final_performance_test_results.json")
    
    return 0 if results['success_rate'] >= 75 else 1


if __name__ == "__main__":
    exit(main())
