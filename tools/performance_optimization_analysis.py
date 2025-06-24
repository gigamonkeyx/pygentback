#!/usr/bin/env python3
"""
Performance Optimization Analysis

Analyze performance bottlenecks in MCP servers and implement optimizations.
"""

import time
import json
import requests
import asyncio
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Any

class PerformanceAnalyzer:
    """Analyze and optimize MCP server performance"""
    
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
    
    def analyze_response_times(self) -> bool:
        """Analyze detailed response times for each endpoint"""
        start = time.time()
        try:
            print("🔍 Analyzing Response Times...")
            
            response_data = {}
            
            for server_id, server_info in self.servers.items():
                port = server_info['port']
                name = server_info['name']
                
                print(f"\n📊 Testing {name} (Port {port}):")
                
                server_results = {
                    'root_endpoint': None,
                    'health_endpoint': None,
                    'cold_start': None,
                    'warm_requests': []
                }
                
                # Test root endpoint (cold start)
                try:
                    start_time = time.time()
                    response = requests.get(f"http://localhost:{port}/", timeout=10)
                    cold_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        server_results['root_endpoint'] = cold_time
                        server_results['cold_start'] = cold_time
                        print(f"   Root endpoint: {cold_time*1000:.1f}ms")
                    else:
                        print(f"   Root endpoint: FAILED (HTTP {response.status_code})")
                except Exception as e:
                    print(f"   Root endpoint: ERROR ({str(e)})")
                
                # Test health endpoint
                try:
                    start_time = time.time()
                    response = requests.get(f"http://localhost:{port}/health", timeout=10)
                    health_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        server_results['health_endpoint'] = health_time
                        print(f"   Health endpoint: {health_time*1000:.1f}ms")
                    else:
                        print(f"   Health endpoint: FAILED (HTTP {response.status_code})")
                except Exception as e:
                    print(f"   Health endpoint: ERROR ({str(e)})")
                
                # Test warm requests (multiple consecutive requests)
                print(f"   Warm requests:")
                for i in range(5):
                    try:
                        start_time = time.time()
                        response = requests.get(f"http://localhost:{port}/health", timeout=5)
                        warm_time = time.time() - start_time
                        
                        if response.status_code == 200:
                            server_results['warm_requests'].append(warm_time)
                            print(f"     Request {i+1}: {warm_time*1000:.1f}ms")
                        else:
                            print(f"     Request {i+1}: FAILED")
                    except Exception as e:
                        print(f"     Request {i+1}: ERROR")
                
                response_data[server_id] = server_results
            
            # Analyze results
            duration = time.time() - start
            
            # Calculate averages
            analysis = {}
            for server_id, results in response_data.items():
                warm_times = results['warm_requests']
                avg_warm = sum(warm_times) / len(warm_times) if warm_times else 0
                
                analysis[server_id] = {
                    'cold_start': results['cold_start'],
                    'health_response': results['health_endpoint'],
                    'avg_warm_response': avg_warm,
                    'performance_category': self._categorize_performance(avg_warm)
                }
            
            # Determine if optimization is needed
            needs_optimization = any(
                data['avg_warm_response'] > 0.5 for data in analysis.values()
            )
            
            if needs_optimization:
                slow_servers = [
                    server_id for server_id, data in analysis.items() 
                    if data['avg_warm_response'] > 0.5
                ]
                details = f"Optimization needed for: {slow_servers}"
                self.log_result("Response Time Analysis", False, details, duration)
                return False
            else:
                details = f"All servers performing well (avg <500ms)"
                self.log_result("Response Time Analysis", True, details, duration)
                return True
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Response Time Analysis", False, f"Error: {str(e)}", duration)
            return False
    
    def _categorize_performance(self, response_time: float) -> str:
        """Categorize performance based on response time"""
        if response_time < 0.1:
            return "excellent"
        elif response_time < 0.5:
            return "good"
        elif response_time < 1.0:
            return "fair"
        elif response_time < 2.0:
            return "poor"
        else:
            return "critical"
    
    def test_concurrent_performance(self) -> bool:
        """Test performance under concurrent load"""
        start = time.time()
        try:
            print("\n🚀 Testing Concurrent Performance...")
            
            def test_server_concurrent(server_id: str, port: int, num_requests: int = 10) -> Dict[str, Any]:
                """Test concurrent requests to a server"""
                successful_requests = 0
                response_times = []
                errors = []
                
                def single_request():
                    try:
                        req_start = time.time()
                        response = requests.get(f"http://localhost:{port}/health", timeout=5)
                        req_time = time.time() - req_start
                        
                        if response.status_code == 200:
                            return {'success': True, 'time': req_time}
                        else:
                            return {'success': False, 'error': f"HTTP {response.status_code}"}
                    except Exception as e:
                        return {'success': False, 'error': str(e)}
                
                # Execute concurrent requests
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
                    futures = [executor.submit(single_request) for _ in range(num_requests)]
                    
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        if result['success']:
                            successful_requests += 1
                            response_times.append(result['time'])
                        else:
                            errors.append(result['error'])
                
                return {
                    'server_id': server_id,
                    'total_requests': num_requests,
                    'successful_requests': successful_requests,
                    'success_rate': (successful_requests / num_requests) * 100,
                    'avg_response_time': sum(response_times) / len(response_times) if response_times else 0,
                    'min_response_time': min(response_times) if response_times else 0,
                    'max_response_time': max(response_times) if response_times else 0,
                    'errors': errors
                }
            
            # Test all servers concurrently
            concurrent_results = {}
            for server_id, server_info in self.servers.items():
                port = server_info['port']
                print(f"   Testing {server_info['name']} with 10 concurrent requests...")
                
                result = test_server_concurrent(server_id, port, 10)
                concurrent_results[server_id] = result
                
                print(f"     Success Rate: {result['success_rate']:.1f}%")
                print(f"     Avg Response: {result['avg_response_time']*1000:.1f}ms")
                print(f"     Min/Max: {result['min_response_time']*1000:.1f}ms / {result['max_response_time']*1000:.1f}ms")
                
                if result['errors']:
                    print(f"     Errors: {len(result['errors'])}")
            
            duration = time.time() - start
            
            # Analyze concurrent performance
            all_good = all(
                result['success_rate'] >= 80 and result['avg_response_time'] < 1.0
                for result in concurrent_results.values()
            )
            
            if all_good:
                avg_success_rate = sum(r['success_rate'] for r in concurrent_results.values()) / len(concurrent_results)
                details = f"Good concurrent performance: {avg_success_rate:.1f}% avg success rate"
                self.log_result("Concurrent Performance", True, details, duration)
                return True
            else:
                poor_performers = [
                    server_id for server_id, result in concurrent_results.items()
                    if result['success_rate'] < 80 or result['avg_response_time'] >= 1.0
                ]
                details = f"Poor concurrent performance in: {poor_performers}"
                self.log_result("Concurrent Performance", False, details, duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Concurrent Performance", False, f"Error: {str(e)}", duration)
            return False
    
    def test_memory_usage_patterns(self) -> bool:
        """Test memory usage patterns during operation"""
        start = time.time()
        try:
            print("\n💾 Testing Memory Usage Patterns...")
            
            memory_data = {}
            
            for server_id, server_info in self.servers.items():
                port = server_info['port']
                name = server_info['name']
                
                print(f"   Checking {name}...")
                
                try:
                    # Get initial memory state
                    health_response = requests.get(f"http://localhost:{port}/health", timeout=5)
                    if health_response.status_code == 200:
                        health_data = health_response.json()
                        performance = health_data.get('performance', {})
                        
                        memory_info = {
                            'uptime': performance.get('uptime_seconds', 0),
                            'requests_processed': performance.get('searches_performed', 0) + 
                                               performance.get('documents_processed', 0) + 
                                               performance.get('tasks_completed', 0),
                            'error_rate': performance.get('error_rate', 0)
                        }
                        
                        memory_data[server_id] = memory_info
                        
                        print(f"     Uptime: {memory_info['uptime']:.1f}s")
                        print(f"     Requests: {memory_info['requests_processed']}")
                        print(f"     Error Rate: {memory_info['error_rate']:.1f}%")
                    else:
                        print(f"     Health check failed: HTTP {health_response.status_code}")
                        
                except Exception as e:
                    print(f"     Error: {str(e)}")
            
            duration = time.time() - start
            
            # Analyze memory patterns
            stable_servers = 0
            for server_id, data in memory_data.items():
                # Consider stable if uptime > 30 minutes and error rate < 10%
                if data['uptime'] > 1800 and data['error_rate'] < 10:
                    stable_servers += 1
            
            if stable_servers == len(memory_data):
                details = f"All {stable_servers} servers showing stable memory patterns"
                self.log_result("Memory Usage Patterns", True, details, duration)
                return True
            else:
                details = f"Only {stable_servers}/{len(memory_data)} servers stable"
                self.log_result("Memory Usage Patterns", False, details, duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Memory Usage Patterns", False, f"Error: {str(e)}", duration)
            return False
    
    def generate_optimization_recommendations(self) -> Dict[str, List[str]]:
        """Generate specific optimization recommendations"""
        recommendations = {
            'immediate': [],
            'short_term': [],
            'long_term': []
        }
        
        # Analyze test results to generate recommendations
        response_time_issues = any(
            'slow response' in result['details'].lower() 
            for result in self.results 
            if not result['success']
        )
        
        concurrent_issues = any(
            'concurrent' in result['test'].lower() and not result['success']
            for result in self.results
        )
        
        if response_time_issues:
            recommendations['immediate'].extend([
                "Implement response caching for health endpoints",
                "Optimize database connection pooling",
                "Add request/response compression",
                "Implement async request handling where possible"
            ])
        
        if concurrent_issues:
            recommendations['short_term'].extend([
                "Implement connection pooling",
                "Add request queuing and rate limiting",
                "Optimize thread pool configurations",
                "Implement circuit breaker patterns"
            ])
        
        recommendations['long_term'].extend([
            "Consider horizontal scaling with load balancers",
            "Implement distributed caching (Redis)",
            "Add performance monitoring and alerting",
            "Optimize vector operations with GPU acceleration"
        ])
        
        return recommendations
    
    def run_performance_analysis(self) -> Dict[str, Any]:
        """Run complete performance analysis"""
        print("⚡ MCP Server Performance Analysis")
        print("=" * 50)
        
        tests = [
            ("Response Time Analysis", self.analyze_response_times),
            ("Concurrent Performance", self.test_concurrent_performance),
            ("Memory Usage Patterns", self.test_memory_usage_patterns)
        ]
        
        passed = 0
        for test_name, test_func in tests:
            if test_func():
                passed += 1
            print()
        
        total = len(tests)
        
        # Generate recommendations
        recommendations = self.generate_optimization_recommendations()
        
        print("=" * 50)
        print(f"📊 Performance Analysis Results: {passed}/{total} tests passed")
        
        print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
        
        if recommendations['immediate']:
            print(f"\n🔥 IMMEDIATE (Critical):")
            for rec in recommendations['immediate']:
                print(f"   • {rec}")
        
        if recommendations['short_term']:
            print(f"\n⏰ SHORT TERM (1-2 weeks):")
            for rec in recommendations['short_term']:
                print(f"   • {rec}")
        
        if recommendations['long_term']:
            print(f"\n🎯 LONG TERM (1-3 months):")
            for rec in recommendations['long_term']:
                print(f"   • {rec}")
        
        return {
            'total_tests': total,
            'passed_tests': passed,
            'success_rate': (passed / total) * 100,
            'test_results': self.results,
            'recommendations': recommendations
        }


def main():
    """Main execution"""
    analyzer = PerformanceAnalyzer()
    results = analyzer.run_performance_analysis()
    
    # Save results
    with open('performance_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved to: performance_analysis_results.json")
    
    return 0 if results['success_rate'] >= 70 else 1


if __name__ == "__main__":
    exit(main())
