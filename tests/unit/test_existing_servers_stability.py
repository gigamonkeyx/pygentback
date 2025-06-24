#!/usr/bin/env python3
"""
Test stability of existing running MCP servers

Focus on testing the four custom MCP servers that are already running.
"""

import time
import json
import requests
import subprocess
from datetime import datetime
from typing import Dict, List, Any

class ExistingServersStabilityTester:
    """Test stability of existing MCP servers"""
    
    def __init__(self):
        self.results = []
        self.servers = {
            'embedding': {'port': 8002, 'name': 'Embedding MCP Server'},
            'document-processing': {'port': 8003, 'name': 'Document Processing MCP Server'},
            'vector-search': {'port': 8004, 'name': 'Vector Search MCP Server'},
            'agent-orchestration': {'port': 8005, 'name': 'Agent Orchestration MCP Server'}
        }
        self.start_time = time.time()
    
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
    
    def test_all_servers_running(self) -> bool:
        """Test that all servers are running and responding"""
        start = time.time()
        try:
            running_servers = 0
            server_details = []
            
            for server_id, server_info in self.servers.items():
                port = server_info['port']
                name = server_info['name']
                
                try:
                    # Test root endpoint
                    root_response = requests.get(f"http://localhost:{port}/", timeout=5)
                    if root_response.status_code == 200:
                        root_data = root_response.json()
                        service_name = root_data.get('service', 'Unknown')
                        
                        # Test health endpoint
                        health_response = requests.get(f"http://localhost:{port}/health", timeout=5)
                        if health_response.status_code == 200:
                            health_data = health_response.json()
                            status = health_data.get('status', 'unknown')
                            
                            if status == 'healthy':
                                running_servers += 1
                                server_details.append(f"{server_id} (port {port}): {status}")
                            else:
                                server_details.append(f"{server_id} (port {port}): {status}")
                        else:
                            server_details.append(f"{server_id} (port {port}): health check failed")
                    else:
                        server_details.append(f"{server_id} (port {port}): not responding")
                        
                except requests.RequestException as e:
                    server_details.append(f"{server_id} (port {port}): {str(e)}")
            
            duration = time.time() - start
            total_servers = len(self.servers)
            
            if running_servers == total_servers:
                details = f"All {running_servers}/{total_servers} servers healthy: " + ", ".join(server_details)
                self.log_result("All Servers Running", True, details, duration)
                return True
            else:
                details = f"Only {running_servers}/{total_servers} servers healthy: " + ", ".join(server_details)
                self.log_result("All Servers Running", False, details, duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("All Servers Running", False, f"Error: {str(e)}", duration)
            return False
    
    def test_server_performance_metrics(self) -> bool:
        """Test server performance metrics"""
        start = time.time()
        try:
            performance_data = {}
            
            for server_id, server_info in self.servers.items():
                port = server_info['port']
                
                try:
                    health_response = requests.get(f"http://localhost:{port}/health", timeout=5)
                    if health_response.status_code == 200:
                        health_data = health_response.json()
                        performance = health_data.get('performance', {})
                        
                        # Extract key metrics
                        uptime = performance.get('uptime_seconds', 0)
                        error_rate = performance.get('error_rate', 0)
                        
                        performance_data[server_id] = {
                            'uptime': uptime,
                            'error_rate': error_rate,
                            'healthy': uptime > 0 and error_rate < 10  # Less than 10% error rate
                        }
                    else:
                        performance_data[server_id] = {'healthy': False, 'error': f"HTTP {health_response.status_code}"}
                        
                except requests.RequestException as e:
                    performance_data[server_id] = {'healthy': False, 'error': str(e)}
            
            duration = time.time() - start
            
            healthy_servers = sum(1 for data in performance_data.values() if data.get('healthy', False))
            total_servers = len(performance_data)
            
            if healthy_servers == total_servers:
                avg_uptime = sum(data.get('uptime', 0) for data in performance_data.values()) / total_servers
                details = f"All servers performing well, Average uptime: {avg_uptime:.1f}s"
                self.log_result("Server Performance Metrics", True, details, duration)
                return True
            else:
                unhealthy = [server_id for server_id, data in performance_data.items() if not data.get('healthy', False)]
                details = f"Performance issues in {len(unhealthy)} servers: {unhealthy}"
                self.log_result("Server Performance Metrics", False, details, duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Server Performance Metrics", False, f"Error: {str(e)}", duration)
            return False
    
    def test_concurrent_requests(self) -> bool:
        """Test concurrent requests to all servers"""
        start = time.time()
        try:
            import concurrent.futures
            
            def test_server_concurrent(server_id: str, port: int) -> Dict[str, Any]:
                """Test concurrent requests to a single server"""
                successful_requests = 0
                total_requests = 10
                response_times = []
                
                for _ in range(total_requests):
                    try:
                        req_start = time.time()
                        response = requests.get(f"http://localhost:{port}/health", timeout=3)
                        req_time = time.time() - req_start
                        
                        if response.status_code == 200:
                            successful_requests += 1
                            response_times.append(req_time)
                    except:
                        pass
                
                avg_response_time = sum(response_times) / len(response_times) if response_times else 0
                success_rate = (successful_requests / total_requests) * 100
                
                return {
                    'server_id': server_id,
                    'success_rate': success_rate,
                    'avg_response_time': avg_response_time,
                    'successful_requests': successful_requests,
                    'total_requests': total_requests
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
            
            duration = time.time() - start
            
            # Analyze results
            good_performance = 0
            performance_details = []
            
            for server_id, result in results.items():
                success_rate = result['success_rate']
                avg_time = result['avg_response_time']
                
                if success_rate >= 80 and avg_time < 1.0:  # 80% success rate, under 1 second
                    good_performance += 1
                    performance_details.append(f"{server_id}: {success_rate:.0f}% success, {avg_time*1000:.0f}ms avg")
                else:
                    performance_details.append(f"{server_id}: {success_rate:.0f}% success, {avg_time*1000:.0f}ms avg (POOR)")
            
            total_servers = len(results)
            
            if good_performance == total_servers:
                details = f"All servers handled concurrent load well: " + ", ".join(performance_details)
                self.log_result("Concurrent Requests", True, details, duration)
                return True
            else:
                details = f"Only {good_performance}/{total_servers} servers performed well: " + ", ".join(performance_details)
                self.log_result("Concurrent Requests", False, details, duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Concurrent Requests", False, f"Error: {str(e)}", duration)
            return False
    
    def test_service_endpoints(self) -> bool:
        """Test specific service endpoints"""
        start = time.time()
        try:
            endpoint_tests = []
            
            # Test embedding server
            try:
                response = requests.post(
                    "http://localhost:8002/v1/embeddings",
                    json={"input": "test", "model": "sentence-transformers"},
                    timeout=10
                )
                endpoint_tests.append(("Embedding API", response.status_code == 200))
            except:
                endpoint_tests.append(("Embedding API", False))
            
            # Test document processing server
            try:
                response = requests.post(
                    "http://localhost:8003/v1/documents/analyze",
                    json={"text": "This is a test document", "analysis_type": "basic"},
                    timeout=10
                )
                endpoint_tests.append(("Document Analysis API", response.status_code == 200))
            except:
                endpoint_tests.append(("Document Analysis API", False))
            
            # Test vector search server
            try:
                response = requests.get("http://localhost:8004/v1/collections", timeout=10)
                endpoint_tests.append(("Vector Collections API", response.status_code == 200))
            except:
                endpoint_tests.append(("Vector Collections API", False))
            
            # Test agent orchestration server
            try:
                response = requests.get("http://localhost:8005/v1/agents", timeout=10)
                endpoint_tests.append(("Agent Management API", response.status_code == 200))
            except:
                endpoint_tests.append(("Agent Management API", False))
            
            duration = time.time() - start
            
            successful_endpoints = sum(1 for _, success in endpoint_tests if success)
            total_endpoints = len(endpoint_tests)
            
            if successful_endpoints == total_endpoints:
                details = f"All {successful_endpoints}/{total_endpoints} service endpoints working"
                self.log_result("Service Endpoints", True, details, duration)
                return True
            else:
                failed_endpoints = [name for name, success in endpoint_tests if not success]
                details = f"Only {successful_endpoints}/{total_endpoints} endpoints working. Failed: {failed_endpoints}"
                self.log_result("Service Endpoints", False, details, duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Service Endpoints", False, f"Error: {str(e)}", duration)
            return False
    
    def test_stability_over_time(self) -> bool:
        """Test stability over a 30-second period"""
        start = time.time()
        try:
            print("    🕐 Running 30-second stability test...")
            
            stability_checks = []
            test_duration = 30  # seconds
            check_interval = 5   # seconds
            
            end_time = time.time() + test_duration
            
            while time.time() < end_time:
                check_start = time.time()
                
                # Quick health check of all servers
                all_healthy = True
                for server_id, server_info in self.servers.items():
                    port = server_info['port']
                    try:
                        response = requests.get(f"http://localhost:{port}/health", timeout=2)
                        if response.status_code != 200:
                            all_healthy = False
                            break
                    except:
                        all_healthy = False
                        break
                
                stability_checks.append(all_healthy)
                
                # Wait for next check
                elapsed = time.time() - check_start
                sleep_time = max(0, check_interval - elapsed)
                time.sleep(sleep_time)
            
            duration = time.time() - start
            
            stable_checks = sum(1 for check in stability_checks if check)
            total_checks = len(stability_checks)
            stability_rate = (stable_checks / total_checks) * 100 if total_checks > 0 else 0
            
            if stability_rate >= 90:  # 90% stability required
                details = f"Stability rate: {stability_rate:.1f}% ({stable_checks}/{total_checks} checks passed)"
                self.log_result("Stability Over Time", True, details, duration)
                return True
            else:
                details = f"Poor stability rate: {stability_rate:.1f}% ({stable_checks}/{total_checks} checks passed)"
                self.log_result("Stability Over Time", False, details, duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Stability Over Time", False, f"Error: {str(e)}", duration)
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all stability tests"""
        print("🔧 Existing MCP Servers Stability Test Suite")
        print("=" * 60)
        
        tests = [
            ("All Servers Running", self.test_all_servers_running),
            ("Server Performance Metrics", self.test_server_performance_metrics),
            ("Concurrent Requests", self.test_concurrent_requests),
            ("Service Endpoints", self.test_service_endpoints),
            ("Stability Over Time", self.test_stability_over_time)
        ]
        
        passed = 0
        for test_name, test_func in tests:
            if test_func():
                passed += 1
            print()  # Add spacing
        
        total = len(tests)
        
        print("=" * 60)
        print(f"📊 Stability Test Results: {passed}/{total} tests passed")
        print(f"🕐 Total test duration: {time.time() - self.start_time:.1f} seconds")
        
        return {
            'total_tests': total,
            'passed_tests': passed,
            'success_rate': (passed / total) * 100,
            'test_results': self.results,
            'total_duration': time.time() - self.start_time
        }


def main():
    """Main test execution"""
    tester = ExistingServersStabilityTester()
    results = tester.run_all_tests()
    
    # Save results
    with open('existing_servers_stability_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved to: existing_servers_stability_results.json")
    
    # Return appropriate exit code
    return 0 if results['success_rate'] >= 80 else 1


if __name__ == "__main__":
    exit(main())
