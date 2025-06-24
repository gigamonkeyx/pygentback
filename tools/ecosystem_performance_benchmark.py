#!/usr/bin/env python3
"""
PyGent Factory MCP Ecosystem Performance Benchmark

Comprehensive performance testing and benchmarking for the complete MCP ecosystem.
"""

import asyncio
import aiohttp
import json
import time
import statistics
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class PerformanceResult:
    """Performance test result"""
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    p95_response_time_ms: float
    requests_per_second: float
    success_rate: float
    errors: List[str]

class EcosystemPerformanceBenchmark:
    """Performance benchmark for PyGent Factory MCP ecosystem"""
    
    def __init__(self):
        self.session = None
        self.results = []
    
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=50)
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def benchmark_endpoint(self, name: str, url: str, method: str = "GET", 
                                payload: Optional[Dict] = None, concurrent_requests: int = 10, 
                                total_requests: int = 100) -> PerformanceResult:
        """Benchmark a specific endpoint"""
        print(f"🚀 Benchmarking {name}...")
        print(f"   URL: {url}")
        print(f"   Method: {method}")
        print(f"   Concurrent: {concurrent_requests}")
        print(f"   Total Requests: {total_requests}")
        
        response_times = []
        errors = []
        successful_requests = 0
        failed_requests = 0
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def make_request():
            async with semaphore:
                start_time = time.time()
                try:
                    if method.upper() == "GET":
                        async with self.session.get(url) as response:
                            await response.text()  # Consume response
                            response_time = (time.time() - start_time) * 1000
                            
                            if response.status == 200:
                                return response_time, None
                            else:
                                return None, f"HTTP {response.status}"
                    
                    elif method.upper() == "POST":
                        async with self.session.post(url, json=payload) as response:
                            await response.text()  # Consume response
                            response_time = (time.time() - start_time) * 1000
                            
                            if response.status == 200:
                                return response_time, None
                            else:
                                return None, f"HTTP {response.status}"
                    
                except Exception as e:
                    return None, str(e)
        
        # Execute all requests
        start_benchmark = time.time()
        tasks = [make_request() for _ in range(total_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_benchmark_time = time.time() - start_benchmark
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                failed_requests += 1
                errors.append(str(result))
            else:
                response_time, error = result
                if error:
                    failed_requests += 1
                    errors.append(error)
                else:
                    successful_requests += 1
                    response_times.append(response_time)
        
        # Calculate statistics
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
        else:
            avg_response_time = min_response_time = max_response_time = p95_response_time = 0
        
        requests_per_second = total_requests / total_benchmark_time if total_benchmark_time > 0 else 0
        success_rate = (successful_requests / total_requests) * 100 if total_requests > 0 else 0
        
        result = PerformanceResult(
            test_name=name,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time_ms=round(avg_response_time, 2),
            min_response_time_ms=round(min_response_time, 2),
            max_response_time_ms=round(max_response_time, 2),
            p95_response_time_ms=round(p95_response_time, 2),
            requests_per_second=round(requests_per_second, 2),
            success_rate=round(success_rate, 2),
            errors=list(set(errors))  # Unique errors only
        )
        
        # Display results
        print(f"   ✅ Success Rate: {result.success_rate}%")
        print(f"   ⚡ Avg Response Time: {result.avg_response_time_ms}ms")
        print(f"   📊 P95 Response Time: {result.p95_response_time_ms}ms")
        print(f"   🔥 Requests/Second: {result.requests_per_second}")
        if result.errors:
            print(f"   ❌ Errors: {len(result.errors)} unique error types")
        print()
        
        return result
    
    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmark"""
        print("🏁 PyGent Factory MCP Ecosystem Performance Benchmark")
        print("=" * 70)
        
        benchmark_results = []
        
        # Benchmark 1: Health Check Performance
        health_result = await self.benchmark_endpoint(
            name="Health Check Performance",
            url="http://127.0.0.1:8005/health",
            concurrent_requests=20,
            total_requests=200
        )
        benchmark_results.append(health_result)
        
        # Benchmark 2: A2A Agent Discovery Performance
        discovery_result = await self.benchmark_endpoint(
            name="A2A Agent Discovery",
            url="http://127.0.0.1:8005/v1/a2a/agents",
            concurrent_requests=15,
            total_requests=150
        )
        benchmark_results.append(discovery_result)
        
        # Benchmark 3: Document Processing Server Info
        doc_result = await self.benchmark_endpoint(
            name="Document Processing Info",
            url="http://127.0.0.1:8003/",
            concurrent_requests=10,
            total_requests=100
        )
        benchmark_results.append(doc_result)
        
        # Benchmark 4: Vector Search Server Info
        vector_result = await self.benchmark_endpoint(
            name="Vector Search Info",
            url="http://127.0.0.1:8004/",
            concurrent_requests=10,
            total_requests=100
        )
        benchmark_results.append(vector_result)
        
        # Benchmark 5: A2A MCP Server Info
        a2a_result = await self.benchmark_endpoint(
            name="A2A MCP Server Info",
            url="http://127.0.0.1:8006/",
            concurrent_requests=10,
            total_requests=100
        )
        benchmark_results.append(a2a_result)
        
        # Benchmark 6: Simple A2A Agent Info
        agent_result = await self.benchmark_endpoint(
            name="Simple A2A Agent Info",
            url="http://127.0.0.1:8007/",
            concurrent_requests=10,
            total_requests=100
        )
        benchmark_results.append(agent_result)
        
        # Benchmark 7: A2A Message Sending (Light Load)
        message_payload = {
            "agent_id": "pygent_factory_2b874e6f",  # Assuming this agent exists
            "message": "Performance test message",
            "context_id": "perf-test-001"
        }
        
        message_result = await self.benchmark_endpoint(
            name="A2A Message Sending",
            url="http://127.0.0.1:8005/v1/a2a/message",
            method="POST",
            payload=message_payload,
            concurrent_requests=5,
            total_requests=50
        )
        benchmark_results.append(message_result)
        
        # Calculate overall statistics
        total_requests = sum(r.total_requests for r in benchmark_results)
        total_successful = sum(r.successful_requests for r in benchmark_results)
        overall_success_rate = (total_successful / total_requests) * 100 if total_requests > 0 else 0
        
        avg_response_times = [r.avg_response_time_ms for r in benchmark_results if r.successful_requests > 0]
        overall_avg_response_time = statistics.mean(avg_response_times) if avg_response_times else 0
        
        total_rps = sum(r.requests_per_second for r in benchmark_results)
        
        # Display summary
        print("=" * 70)
        print("📊 PERFORMANCE BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"🎯 Overall Success Rate: {overall_success_rate:.2f}%")
        print(f"⚡ Average Response Time: {overall_avg_response_time:.2f}ms")
        print(f"🔥 Total Throughput: {total_rps:.2f} requests/second")
        print(f"📈 Total Requests Processed: {total_successful}/{total_requests}")
        
        # Performance rating
        if overall_success_rate >= 99 and overall_avg_response_time <= 50:
            rating = "EXCELLENT"
            emoji = "🏆"
        elif overall_success_rate >= 95 and overall_avg_response_time <= 100:
            rating = "GOOD"
            emoji = "✅"
        elif overall_success_rate >= 90 and overall_avg_response_time <= 200:
            rating = "ACCEPTABLE"
            emoji = "⚠️"
        else:
            rating = "NEEDS IMPROVEMENT"
            emoji = "❌"
        
        print(f"\n{emoji} Performance Rating: {rating}")
        
        return {
            "overall_success_rate": overall_success_rate,
            "overall_avg_response_time_ms": round(overall_avg_response_time, 2),
            "total_throughput_rps": round(total_rps, 2),
            "total_requests": total_requests,
            "successful_requests": total_successful,
            "performance_rating": rating,
            "benchmark_results": [
                {
                    "test_name": r.test_name,
                    "success_rate": r.success_rate,
                    "avg_response_time_ms": r.avg_response_time_ms,
                    "p95_response_time_ms": r.p95_response_time_ms,
                    "requests_per_second": r.requests_per_second,
                    "total_requests": r.total_requests,
                    "errors": r.errors
                }
                for r in benchmark_results
            ],
            "timestamp": datetime.utcnow().isoformat()
        }


async def main():
    """Main benchmark execution"""
    async with EcosystemPerformanceBenchmark() as benchmark:
        results = await benchmark.run_comprehensive_benchmark()
        
        # Save results
        with open('ecosystem_performance_report.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Performance report saved to: ecosystem_performance_report.json")
        
        # Return appropriate exit code based on performance
        return 0 if results["performance_rating"] in ["EXCELLENT", "GOOD"] else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
