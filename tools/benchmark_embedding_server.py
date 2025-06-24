#!/usr/bin/env python3
"""
Embedding Server Performance Benchmark Suite

Comprehensive performance testing including throughput, latency, 
memory usage, and stress testing.
"""

import time
import json
import requests
import statistics
import psutil
import threading
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

class EmbeddingServerBenchmark:
    """Comprehensive performance benchmark suite"""
    
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url
        self.results = []
        self.process = psutil.Process()
    
    def log_result(self, test_name: str, success: bool, metrics: Dict[str, Any] = None, details: str = ""):
        """Log benchmark result with metrics"""
        result = {
            'test': test_name,
            'success': success,
            'metrics': metrics or {},
            'details': details,
            'timestamp': time.time()
        }
        self.results.append(result)
        
        status = "✅" if success else "❌"
        print(f"{status} {test_name}")
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.2f}")
                else:
                    print(f"    {key}: {value}")
        if details:
            print(f"    {details}")
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def benchmark_latency(self) -> bool:
        """Benchmark request latency with statistical analysis"""
        print("🚀 Running latency benchmark...")
        
        test_text = "Latency benchmark test text for performance measurement"
        latencies = []
        
        try:
            # Warm up
            for _ in range(3):
                requests.post(
                    f"{self.base_url}/v1/embeddings",
                    json={"input": test_text, "model": "text-embedding-ada-002"},
                    timeout=30
                )
            
            # Actual benchmark
            for i in range(20):
                start = time.time()
                response = requests.post(
                    f"{self.base_url}/v1/embeddings",
                    json={"input": test_text, "model": "text-embedding-ada-002"},
                    timeout=30
                )
                latency = time.time() - start
                
                if response.status_code == 200:
                    latencies.append(latency * 1000)  # Convert to ms
                else:
                    return False
            
            if latencies:
                metrics = {
                    'mean_latency_ms': statistics.mean(latencies),
                    'median_latency_ms': statistics.median(latencies),
                    'min_latency_ms': min(latencies),
                    'max_latency_ms': max(latencies),
                    'std_dev_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0,
                    'p95_latency_ms': sorted(latencies)[int(0.95 * len(latencies))],
                    'p99_latency_ms': sorted(latencies)[int(0.99 * len(latencies))],
                    'sample_count': len(latencies)
                }
                
                self.log_result("Latency Benchmark", True, metrics)
                return True
            else:
                self.log_result("Latency Benchmark", False, details="No successful requests")
                return False
                
        except Exception as e:
            self.log_result("Latency Benchmark", False, details=f"Error: {str(e)}")
            return False
    
    def benchmark_throughput(self) -> bool:
        """Benchmark throughput with concurrent requests"""
        print("📈 Running throughput benchmark...")
        
        def send_request(request_id: int) -> Tuple[bool, float]:
            start = time.time()
            try:
                response = requests.post(
                    f"{self.base_url}/v1/embeddings",
                    json={"input": f"Throughput test request {request_id}", "model": "text-embedding-ada-002"},
                    timeout=60
                )
                duration = time.time() - start
                return response.status_code == 200, duration
            except:
                return False, time.time() - start
        
        try:
            concurrent_levels = [1, 2, 4, 8]
            throughput_results = {}
            
            for concurrency in concurrent_levels:
                print(f"  Testing concurrency level: {concurrency}")
                
                start_time = time.time()
                successful_requests = 0
                total_requests = concurrency * 5  # 5 requests per thread
                
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = [executor.submit(send_request, i) for i in range(total_requests)]
                    
                    for future in as_completed(futures):
                        success, _ = future.result()
                        if success:
                            successful_requests += 1
                
                total_time = time.time() - start_time
                throughput = successful_requests / total_time if total_time > 0 else 0
                
                throughput_results[f'concurrency_{concurrency}'] = {
                    'requests_per_second': throughput,
                    'successful_requests': successful_requests,
                    'total_requests': total_requests,
                    'success_rate': (successful_requests / total_requests) * 100,
                    'total_time_seconds': total_time
                }
            
            # Find optimal concurrency
            best_throughput = max(throughput_results.values(), key=lambda x: x['requests_per_second'])
            
            metrics = {
                'optimal_concurrency': max(throughput_results.keys(), key=lambda k: throughput_results[k]['requests_per_second']),
                'max_throughput_rps': best_throughput['requests_per_second'],
                'throughput_by_concurrency': throughput_results
            }
            
            self.log_result("Throughput Benchmark", True, metrics)
            return True
            
        except Exception as e:
            self.log_result("Throughput Benchmark", False, details=f"Error: {str(e)}")
            return False
    
    def benchmark_batch_efficiency(self) -> bool:
        """Benchmark batch processing efficiency"""
        print("📦 Running batch efficiency benchmark...")
        
        try:
            batch_sizes = [1, 5, 10, 20, 50]
            efficiency_results = {}
            
            for batch_size in batch_sizes:
                print(f"  Testing batch size: {batch_size}")
                
                texts = [f"Batch efficiency test item {i}" for i in range(batch_size)]
                
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/v1/embeddings",
                    json={"input": texts, "model": "text-embedding-ada-002"},
                    timeout=120
                )
                total_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('data') and len(data['data']) == batch_size:
                        throughput = batch_size / total_time
                        latency_per_item = (total_time / batch_size) * 1000  # ms per item
                        
                        efficiency_results[f'batch_{batch_size}'] = {
                            'items_per_second': throughput,
                            'latency_per_item_ms': latency_per_item,
                            'total_time_seconds': total_time,
                            'batch_size': batch_size
                        }
                    else:
                        efficiency_results[f'batch_{batch_size}'] = {'error': 'Invalid response'}
                else:
                    efficiency_results[f'batch_{batch_size}'] = {'error': f'HTTP {response.status_code}'}
            
            # Calculate efficiency metrics
            valid_results = {k: v for k, v in efficiency_results.items() if 'error' not in v}
            
            if valid_results:
                best_efficiency = max(valid_results.values(), key=lambda x: x['items_per_second'])
                
                metrics = {
                    'optimal_batch_size': max(valid_results.keys(), key=lambda k: valid_results[k]['items_per_second']),
                    'max_batch_throughput': best_efficiency['items_per_second'],
                    'efficiency_by_batch_size': efficiency_results
                }
                
                self.log_result("Batch Efficiency", True, metrics)
                return True
            else:
                self.log_result("Batch Efficiency", False, details="No valid batch results")
                return False
                
        except Exception as e:
            self.log_result("Batch Efficiency", False, details=f"Error: {str(e)}")
            return False
    
    def benchmark_memory_usage(self) -> bool:
        """Benchmark memory usage patterns"""
        print("💾 Running memory usage benchmark...")
        
        try:
            initial_memory = self.get_memory_usage()
            memory_samples = [initial_memory]
            
            # Send requests while monitoring memory
            for i in range(50):
                # Send a request
                response = requests.post(
                    f"{self.base_url}/v1/embeddings",
                    json={"input": f"Memory test request {i} with substantial content to test memory usage patterns", "model": "text-embedding-ada-002"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    current_memory = self.get_memory_usage()
                    memory_samples.append(current_memory)
                
                # Sample every 10 requests
                if i % 10 == 0:
                    time.sleep(0.1)  # Brief pause for memory measurement
            
            # Force garbage collection and final measurement
            gc.collect()
            time.sleep(1)
            final_memory = self.get_memory_usage()
            
            metrics = {
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'peak_memory_mb': max(memory_samples),
                'memory_growth_mb': final_memory - initial_memory,
                'avg_memory_mb': statistics.mean(memory_samples),
                'memory_samples': len(memory_samples)
            }
            
            self.log_result("Memory Usage", True, metrics)
            return True
            
        except Exception as e:
            self.log_result("Memory Usage", False, details=f"Error: {str(e)}")
            return False
    
    def stress_test(self) -> bool:
        """Stress test with high load"""
        print("🔥 Running stress test...")
        
        try:
            stress_duration = 30  # seconds
            max_concurrent = 10
            
            def stress_worker(worker_id: int) -> Dict[str, Any]:
                start_time = time.time()
                requests_sent = 0
                successful_requests = 0
                errors = []
                
                while time.time() - start_time < stress_duration:
                    try:
                        response = requests.post(
                            f"{self.base_url}/v1/embeddings",
                            json={"input": f"Stress test from worker {worker_id} request {requests_sent}", "model": "text-embedding-ada-002"},
                            timeout=10
                        )
                        
                        requests_sent += 1
                        if response.status_code == 200:
                            successful_requests += 1
                        else:
                            errors.append(f"HTTP {response.status_code}")
                            
                    except Exception as e:
                        requests_sent += 1
                        errors.append(str(e))
                
                return {
                    'worker_id': worker_id,
                    'requests_sent': requests_sent,
                    'successful_requests': successful_requests,
                    'error_count': len(errors),
                    'success_rate': (successful_requests / requests_sent) * 100 if requests_sent > 0 else 0
                }
            
            print(f"  Running {max_concurrent} workers for {stress_duration} seconds...")
            
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                futures = [executor.submit(stress_worker, i) for i in range(max_concurrent)]
                worker_results = [future.result() for future in as_completed(futures)]
            
            total_time = time.time() - start_time
            
            # Aggregate results
            total_requests = sum(r['requests_sent'] for r in worker_results)
            total_successful = sum(r['successful_requests'] for r in worker_results)
            total_errors = sum(r['error_count'] for r in worker_results)
            
            metrics = {
                'duration_seconds': total_time,
                'total_requests': total_requests,
                'successful_requests': total_successful,
                'failed_requests': total_errors,
                'overall_success_rate': (total_successful / total_requests) * 100 if total_requests > 0 else 0,
                'requests_per_second': total_requests / total_time if total_time > 0 else 0,
                'successful_rps': total_successful / total_time if total_time > 0 else 0,
                'concurrent_workers': max_concurrent,
                'worker_results': worker_results
            }
            
            success = metrics['overall_success_rate'] > 80  # 80% success rate threshold
            self.log_result("Stress Test", success, metrics)
            return success
            
        except Exception as e:
            self.log_result("Stress Test", False, details=f"Error: {str(e)}")
            return False
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmarks"""
        print("⚡ Embedding Server Performance Benchmark Suite")
        print("=" * 50)
        
        benchmarks = [
            ("Latency", self.benchmark_latency),
            ("Throughput", self.benchmark_throughput),
            ("Batch Efficiency", self.benchmark_batch_efficiency),
            ("Memory Usage", self.benchmark_memory_usage),
            ("Stress Test", self.stress_test)
        ]
        
        passed = 0
        for benchmark_name, benchmark_func in benchmarks:
            if benchmark_func():
                passed += 1
            print()  # Add spacing between benchmarks
        
        total = len(benchmarks)
        
        print("=" * 50)
        print(f"📊 Benchmark Results: {passed}/{total} benchmarks passed")
        
        return {
            'total_benchmarks': total,
            'passed_benchmarks': passed,
            'success_rate': (passed / total) * 100,
            'benchmark_results': self.results
        }


def main():
    """Main benchmark execution"""
    benchmark = EmbeddingServerBenchmark()
    results = benchmark.run_all_benchmarks()
    
    # Save detailed results
    with open('embedding_server_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: embedding_server_benchmark.json")
    return 0 if results['success_rate'] >= 80 else 1


if __name__ == "__main__":
    exit(main())
