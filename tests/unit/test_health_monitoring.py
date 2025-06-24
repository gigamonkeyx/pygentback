#!/usr/bin/env python3
"""
Health Monitoring Test Suite

Tests health endpoints, monitoring capabilities, and service status reporting
of the embedding server.
"""

import time
import json
import requests
from typing import Dict, List, Any
from datetime import datetime

class HealthMonitoringTester:
    """Test health monitoring and status endpoints"""
    
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url
        self.results = []
    
    def log_result(self, test_name: str, success: bool, details: str = "", data: Dict = None):
        """Log test result"""
        result = {
            'test': test_name,
            'success': success,
            'details': details,
            'data': data or {},
            'timestamp': datetime.utcnow().isoformat()
        }
        self.results.append(result)
        
        status = "✅" if success else "❌"
        print(f"{status} {test_name}")
        if details:
            print(f"    {details}")
        if data:
            for key, value in data.items():
                print(f"    {key}: {value}")
    
    def test_health_endpoint_structure(self) -> bool:
        """Test health endpoint response structure"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                
                # Required fields
                required_fields = ['status', 'timestamp', 'service', 'providers', 'performance']
                missing_fields = [field for field in required_fields if field not in health_data]
                
                if missing_fields:
                    self.log_result("Health Structure", False, f"Missing fields: {missing_fields}")
                    return False
                
                # Validate field types and values
                validations = [
                    (health_data['status'] in ['healthy', 'unhealthy', 'degraded'], "Invalid status value"),
                    (isinstance(health_data['timestamp'], str), "Timestamp not string"),
                    (isinstance(health_data['service'], dict), "Service not dict"),
                    (isinstance(health_data['providers'], dict), "Providers not dict"),
                    (isinstance(health_data['performance'], dict), "Performance not dict")
                ]
                
                failed_validations = [msg for valid, msg in validations if not valid]
                
                if failed_validations:
                    self.log_result("Health Structure", False, f"Validation failures: {failed_validations}")
                    return False
                
                self.log_result("Health Structure", True, "All required fields present and valid", {
                    'status': health_data['status'],
                    'service_name': health_data['service'].get('name', 'unknown'),
                    'provider_count': health_data['providers'].get('provider_count', 0)
                })
                return True
            else:
                self.log_result("Health Structure", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_result("Health Structure", False, f"Error: {str(e)}")
            return False
    
    def test_provider_status_reporting(self) -> bool:
        """Test provider status reporting in health endpoint"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                providers = health_data.get('providers', {})
                
                # Check provider information
                required_provider_fields = ['provider_count', 'current_provider', 'available_providers']
                missing_fields = [field for field in required_provider_fields if field not in providers]
                
                if missing_fields:
                    self.log_result("Provider Status", False, f"Missing provider fields: {missing_fields}")
                    return False
                
                provider_count = providers.get('provider_count', 0)
                current_provider = providers.get('current_provider', '')
                available_providers = providers.get('available_providers', [])
                
                # Validate provider data
                if provider_count > 0 and current_provider and available_providers:
                    self.log_result("Provider Status", True, "Provider information complete", {
                        'provider_count': provider_count,
                        'current_provider': current_provider,
                        'available_providers': ', '.join(available_providers)
                    })
                    return True
                else:
                    self.log_result("Provider Status", False, "Incomplete provider information")
                    return False
            else:
                self.log_result("Provider Status", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_result("Provider Status", False, f"Error: {str(e)}")
            return False
    
    def test_performance_metrics(self) -> bool:
        """Test performance metrics in health endpoint"""
        try:
            # First, generate some load to populate metrics
            for i in range(5):
                requests.post(
                    f"{self.base_url}/v1/embeddings",
                    json={"input": f"Metrics test {i}", "model": "text-embedding-ada-002"},
                    timeout=30
                )
            
            # Now check health metrics
            response = requests.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                performance = health_data.get('performance', {})
                
                # Check for performance metrics
                expected_metrics = ['request_count', 'avg_response_time', 'cache_hit_rate']
                available_metrics = [metric for metric in expected_metrics if metric in performance]
                
                if available_metrics:
                    metrics_data = {metric: performance[metric] for metric in available_metrics}
                    self.log_result("Performance Metrics", True, f"Available metrics: {', '.join(available_metrics)}", metrics_data)
                    return True
                else:
                    self.log_result("Performance Metrics", False, "No performance metrics found")
                    return False
            else:
                self.log_result("Performance Metrics", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_result("Performance Metrics", False, f"Error: {str(e)}")
            return False
    
    def test_health_endpoint_consistency(self) -> bool:
        """Test health endpoint consistency over multiple calls"""
        try:
            health_responses = []
            
            # Make multiple health requests
            for i in range(5):
                response = requests.get(f"{self.base_url}/health", timeout=10)
                if response.status_code == 200:
                    health_responses.append(response.json())
                time.sleep(1)  # Small delay between requests
            
            if len(health_responses) == 5:
                # Check consistency of key fields
                statuses = [h['status'] for h in health_responses]
                provider_counts = [h['providers']['provider_count'] for h in health_responses]
                service_names = [h['service']['name'] for h in health_responses]
                
                # These should be consistent
                status_consistent = len(set(statuses)) == 1
                provider_count_consistent = len(set(provider_counts)) == 1
                service_name_consistent = len(set(service_names)) == 1
                
                if status_consistent and provider_count_consistent and service_name_consistent:
                    self.log_result("Health Consistency", True, "Health data consistent across calls", {
                        'status': statuses[0],
                        'provider_count': provider_counts[0],
                        'service_name': service_names[0]
                    })
                    return True
                else:
                    inconsistencies = []
                    if not status_consistent:
                        inconsistencies.append(f"status: {statuses}")
                    if not provider_count_consistent:
                        inconsistencies.append(f"provider_count: {provider_counts}")
                    if not service_name_consistent:
                        inconsistencies.append(f"service_name: {service_names}")
                    
                    self.log_result("Health Consistency", False, f"Inconsistencies: {inconsistencies}")
                    return False
            else:
                self.log_result("Health Consistency", False, f"Only got {len(health_responses)}/5 responses")
                return False
                
        except Exception as e:
            self.log_result("Health Consistency", False, f"Error: {str(e)}")
            return False
    
    def test_root_endpoint_info(self) -> bool:
        """Test root endpoint service information"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            
            if response.status_code == 200:
                root_data = response.json()
                
                # Check for service information
                required_fields = ['service', 'version', 'endpoints']
                missing_fields = [field for field in required_fields if field not in root_data]
                
                if missing_fields:
                    self.log_result("Root Endpoint", False, f"Missing fields: {missing_fields}")
                    return False
                
                # Validate endpoints information
                endpoints = root_data.get('endpoints', {})
                expected_endpoints = ['embeddings', 'health']
                available_endpoints = [ep for ep in expected_endpoints if ep in endpoints]
                
                if len(available_endpoints) >= 2:
                    self.log_result("Root Endpoint", True, "Service info complete", {
                        'service': root_data['service'],
                        'version': root_data['version'],
                        'endpoints': ', '.join(available_endpoints)
                    })
                    return True
                else:
                    self.log_result("Root Endpoint", False, f"Missing endpoints: {set(expected_endpoints) - set(available_endpoints)}")
                    return False
            else:
                self.log_result("Root Endpoint", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_result("Root Endpoint", False, f"Error: {str(e)}")
            return False
    
    def test_health_under_load(self) -> bool:
        """Test health endpoint responsiveness under load"""
        try:
            import threading
            import concurrent.futures
            
            def generate_load():
                """Generate embedding requests to create load"""
                for i in range(10):
                    try:
                        requests.post(
                            f"{self.base_url}/v1/embeddings",
                            json={"input": f"Load test {i}", "model": "text-embedding-ada-002"},
                            timeout=30
                        )
                    except:
                        pass  # Ignore errors, just generate load
            
            def check_health():
                """Check health endpoint during load"""
                health_times = []
                for i in range(10):
                    start = time.time()
                    try:
                        response = requests.get(f"{self.base_url}/health", timeout=5)
                        if response.status_code == 200:
                            health_times.append(time.time() - start)
                    except:
                        pass
                    time.sleep(0.5)
                return health_times
            
            # Start load generation in background
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                # Start load generators
                load_futures = [executor.submit(generate_load) for _ in range(2)]
                
                # Check health during load
                health_future = executor.submit(check_health)
                
                # Wait for health checks to complete
                health_times = health_future.result()
                
                # Wait for load generation to complete
                for future in load_futures:
                    future.result()
            
            if health_times:
                avg_health_time = sum(health_times) / len(health_times)
                max_health_time = max(health_times)
                
                # Health endpoint should remain responsive (< 1 second)
                if avg_health_time < 1.0 and max_health_time < 2.0:
                    self.log_result("Health Under Load", True, "Health endpoint remained responsive", {
                        'avg_response_time_ms': round(avg_health_time * 1000, 2),
                        'max_response_time_ms': round(max_health_time * 1000, 2),
                        'health_checks': len(health_times)
                    })
                    return True
                else:
                    self.log_result("Health Under Load", False, f"Health endpoint slow: avg={avg_health_time:.3f}s, max={max_health_time:.3f}s")
                    return False
            else:
                self.log_result("Health Under Load", False, "No successful health checks during load")
                return False
                
        except Exception as e:
            self.log_result("Health Under Load", False, f"Error: {str(e)}")
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all health monitoring tests"""
        print("🏥 Health Monitoring Test Suite")
        print("=" * 35)
        
        tests = [
            ("Health Structure", self.test_health_endpoint_structure),
            ("Provider Status", self.test_provider_status_reporting),
            ("Performance Metrics", self.test_performance_metrics),
            ("Health Consistency", self.test_health_endpoint_consistency),
            ("Root Endpoint", self.test_root_endpoint_info),
            ("Health Under Load", self.test_health_under_load)
        ]
        
        passed = 0
        for test_name, test_func in tests:
            if test_func():
                passed += 1
            print()  # Add spacing
        
        total = len(tests)
        
        print("=" * 35)
        print(f"📊 Health Monitoring Results: {passed}/{total} tests passed")
        
        return {
            'total_tests': total,
            'passed_tests': passed,
            'success_rate': (passed / total) * 100,
            'test_results': self.results
        }


def main():
    """Main test execution"""
    tester = HealthMonitoringTester()
    results = tester.run_all_tests()
    
    # Save results
    with open('health_monitoring_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved to: health_monitoring_results.json")
    return 0 if results['success_rate'] >= 80 else 1


if __name__ == "__main__":
    exit(main())
