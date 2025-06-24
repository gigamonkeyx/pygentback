#!/usr/bin/env python3
"""
A2A System Load Testing Framework

Enterprise-scale load testing for A2A multi-agent system.
"""

import asyncio
import aiohttp
import time
import json
import statistics
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)


@dataclass
class LoadTestConfig:
    """Load test configuration"""
    base_url: str = "http://localhost:8080"
    concurrent_users: int = 10
    requests_per_user: int = 100
    ramp_up_time: int = 30  # seconds
    test_duration: int = 300  # seconds
    request_timeout: int = 30
    think_time: float = 1.0  # seconds between requests
    
    # Test scenarios
    scenarios: Dict[str, float] = field(default_factory=lambda: {
        "document_search": 0.4,
        "analysis_task": 0.3,
        "synthesis_task": 0.2,
        "health_check": 0.1
    })


@dataclass
class RequestResult:
    """Individual request result"""
    scenario: str
    method: str
    status_code: int
    response_time: float
    success: bool
    error: Optional[str]
    timestamp: datetime
    task_id: Optional[str] = None


@dataclass
class LoadTestResults:
    """Load test results summary"""
    config: LoadTestConfig
    start_time: datetime
    end_time: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    min_response_time: float
    max_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    error_rate: float
    errors_by_type: Dict[str, int]
    scenario_results: Dict[str, Dict[str, Any]]


class A2ALoadTester:
    """A2A System Load Tester"""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results: List[RequestResult] = []
        self.active_sessions = 0
        self.test_start_time = None
        self.test_end_time = None
        
    async def run_load_test(self) -> LoadTestResults:
        """Run comprehensive load test"""
        
        logger.info(f"Starting load test with {self.config.concurrent_users} users")
        logger.info(f"Test duration: {self.config.test_duration}s, Ramp-up: {self.config.ramp_up_time}s")
        
        self.test_start_time = datetime.utcnow()
        
        # Create user tasks with staggered start
        user_tasks = []
        ramp_up_delay = self.config.ramp_up_time / self.config.concurrent_users
        
        for user_id in range(self.config.concurrent_users):
            start_delay = user_id * ramp_up_delay
            task = asyncio.create_task(
                self._simulate_user(user_id, start_delay)
            )
            user_tasks.append(task)
        
        # Wait for test duration
        await asyncio.sleep(self.config.test_duration + self.config.ramp_up_time)
        
        # Cancel remaining tasks
        for task in user_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for cleanup
        await asyncio.gather(*user_tasks, return_exceptions=True)
        
        self.test_end_time = datetime.utcnow()
        
        # Generate results
        return self._generate_results()
    
    async def _simulate_user(self, user_id: int, start_delay: float):
        """Simulate individual user behavior"""
        
        # Wait for ramp-up
        await asyncio.sleep(start_delay)
        
        logger.debug(f"User {user_id} starting simulation")
        
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        ) as session:
            
            self.active_sessions += 1
            
            try:
                request_count = 0
                user_start = time.time()
                
                while (time.time() - user_start) < self.config.test_duration:
                    if request_count >= self.config.requests_per_user:
                        break
                    
                    # Select scenario based on weights
                    scenario = self._select_scenario()
                    
                    # Execute scenario
                    await self._execute_scenario(session, user_id, scenario)
                    
                    request_count += 1
                    
                    # Think time between requests
                    if self.config.think_time > 0:
                        await asyncio.sleep(self.config.think_time)
                
                logger.debug(f"User {user_id} completed {request_count} requests")
                
            except asyncio.CancelledError:
                logger.debug(f"User {user_id} cancelled")
            except Exception as e:
                logger.error(f"User {user_id} error: {e}")
            finally:
                self.active_sessions -= 1
    
    def _select_scenario(self) -> str:
        """Select scenario based on configured weights"""
        import random
        
        rand = random.random()
        cumulative = 0
        
        for scenario, weight in self.config.scenarios.items():
            cumulative += weight
            if rand <= cumulative:
                return scenario
        
        # Fallback to first scenario
        return list(self.config.scenarios.keys())[0]
    
    async def _execute_scenario(self, session: aiohttp.ClientSession, user_id: int, scenario: str):
        """Execute specific test scenario"""
        
        start_time = time.time()
        timestamp = datetime.utcnow()
        
        try:
            if scenario == "document_search":
                result = await self._scenario_document_search(session, user_id)
            elif scenario == "analysis_task":
                result = await self._scenario_analysis_task(session, user_id)
            elif scenario == "synthesis_task":
                result = await self._scenario_synthesis_task(session, user_id)
            elif scenario == "health_check":
                result = await self._scenario_health_check(session, user_id)
            else:
                raise ValueError(f"Unknown scenario: {scenario}")
            
            response_time = time.time() - start_time
            
            self.results.append(RequestResult(
                scenario=scenario,
                method=result.get("method", "unknown"),
                status_code=result.get("status_code", 0),
                response_time=response_time,
                success=result.get("success", False),
                error=result.get("error"),
                timestamp=timestamp,
                task_id=result.get("task_id")
            ))
            
        except Exception as e:
            response_time = time.time() - start_time
            
            self.results.append(RequestResult(
                scenario=scenario,
                method="unknown",
                status_code=0,
                response_time=response_time,
                success=False,
                error=str(e),
                timestamp=timestamp
            ))
    
    async def _scenario_document_search(self, session: aiohttp.ClientSession, user_id: int) -> Dict[str, Any]:
        """Document search scenario"""
        
        queries = [
            "machine learning algorithms",
            "neural network architectures",
            "artificial intelligence research",
            "deep learning applications",
            "natural language processing"
        ]
        
        import random
        query = random.choice(queries)
        
        request = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": f"Search for documents about {query}"}]
                }
            },
            "id": f"load_test_{user_id}_{int(time.time())}"
        }
        
        async with session.post(self.config.base_url, json=request) as response:
            response_data = await response.json()
            
            success = response.status == 200 and "result" in response_data
            task_id = response_data.get("result", {}).get("id") if success else None
            
            return {
                "method": "tasks/send",
                "status_code": response.status,
                "success": success,
                "task_id": task_id,
                "error": response_data.get("error", {}).get("message") if not success else None
            }
    
    async def _scenario_analysis_task(self, session: aiohttp.ClientSession, user_id: int) -> Dict[str, Any]:
        """Analysis task scenario"""
        
        request = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": "Analyze statistical trends in AI research"}]
                }
            },
            "id": f"load_test_analysis_{user_id}_{int(time.time())}"
        }
        
        async with session.post(self.config.base_url, json=request) as response:
            response_data = await response.json()
            
            success = response.status == 200 and "result" in response_data
            task_id = response_data.get("result", {}).get("id") if success else None
            
            return {
                "method": "tasks/send",
                "status_code": response.status,
                "success": success,
                "task_id": task_id,
                "error": response_data.get("error", {}).get("message") if not success else None
            }
    
    async def _scenario_synthesis_task(self, session: aiohttp.ClientSession, user_id: int) -> Dict[str, Any]:
        """Synthesis task scenario"""
        
        request = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": "Synthesize findings from multiple research sources"}]
                }
            },
            "id": f"load_test_synthesis_{user_id}_{int(time.time())}"
        }
        
        async with session.post(self.config.base_url, json=request) as response:
            response_data = await response.json()
            
            success = response.status == 200 and "result" in response_data
            task_id = response_data.get("result", {}).get("id") if success else None
            
            return {
                "method": "tasks/send",
                "status_code": response.status,
                "success": success,
                "task_id": task_id,
                "error": response_data.get("error", {}).get("message") if not success else None
            }
    
    async def _scenario_health_check(self, session: aiohttp.ClientSession, user_id: int) -> Dict[str, Any]:
        """Health check scenario"""
        
        async with session.get(f"{self.config.base_url}/health") as response:
            response_data = await response.json() if response.status == 200 else {}
            
            success = response.status == 200 and response_data.get("status") == "healthy"
            
            return {
                "method": "GET /health",
                "status_code": response.status,
                "success": success,
                "error": f"Unhealthy status: {response_data.get('status')}" if not success else None
            }
    
    def _generate_results(self) -> LoadTestResults:
        """Generate comprehensive test results"""
        
        if not self.results:
            raise ValueError("No test results available")
        
        # Basic statistics
        total_requests = len(self.results)
        successful_requests = sum(1 for r in self.results if r.success)
        failed_requests = total_requests - successful_requests
        
        response_times = [r.response_time for r in self.results]
        average_response_time = statistics.mean(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)
        
        # Percentiles
        sorted_times = sorted(response_times)
        p95_response_time = sorted_times[int(len(sorted_times) * 0.95)]
        p99_response_time = sorted_times[int(len(sorted_times) * 0.99)]
        
        # Throughput
        test_duration = (self.test_end_time - self.test_start_time).total_seconds()
        requests_per_second = total_requests / test_duration if test_duration > 0 else 0
        
        # Error analysis
        error_rate = failed_requests / total_requests if total_requests > 0 else 0
        errors_by_type = {}
        
        for result in self.results:
            if not result.success and result.error:
                error_type = result.error[:50]  # Truncate long errors
                errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1
        
        # Scenario analysis
        scenario_results = {}
        for scenario in self.config.scenarios.keys():
            scenario_requests = [r for r in self.results if r.scenario == scenario]
            
            if scenario_requests:
                scenario_successful = sum(1 for r in scenario_requests if r.success)
                scenario_times = [r.response_time for r in scenario_requests]
                
                scenario_results[scenario] = {
                    "total_requests": len(scenario_requests),
                    "successful_requests": scenario_successful,
                    "success_rate": scenario_successful / len(scenario_requests),
                    "average_response_time": statistics.mean(scenario_times),
                    "min_response_time": min(scenario_times),
                    "max_response_time": max(scenario_times)
                }
        
        return LoadTestResults(
            config=self.config,
            start_time=self.test_start_time,
            end_time=self.test_end_time,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_response_time=average_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            requests_per_second=requests_per_second,
            error_rate=error_rate,
            errors_by_type=errors_by_type,
            scenario_results=scenario_results
        )
    
    def print_results(self, results: LoadTestResults):
        """Print formatted test results"""
        
        print("\n" + "=" * 80)
        print("A2A LOAD TEST RESULTS")
        print("=" * 80)
        
        print(f"Test Duration: {(results.end_time - results.start_time).total_seconds():.1f}s")
        print(f"Concurrent Users: {results.config.concurrent_users}")
        print(f"Total Requests: {results.total_requests}")
        print(f"Successful Requests: {results.successful_requests}")
        print(f"Failed Requests: {results.failed_requests}")
        print(f"Success Rate: {(1 - results.error_rate) * 100:.2f}%")
        print(f"Requests/Second: {results.requests_per_second:.2f}")
        
        print("\nRESPONSE TIMES:")
        print(f"  Average: {results.average_response_time:.3f}s")
        print(f"  Min: {results.min_response_time:.3f}s")
        print(f"  Max: {results.max_response_time:.3f}s")
        print(f"  95th Percentile: {results.p95_response_time:.3f}s")
        print(f"  99th Percentile: {results.p99_response_time:.3f}s")
        
        if results.errors_by_type:
            print("\nERRORS BY TYPE:")
            for error_type, count in sorted(results.errors_by_type.items(), key=lambda x: x[1], reverse=True):
                print(f"  {error_type}: {count}")
        
        print("\nSCENARIO RESULTS:")
        for scenario, stats in results.scenario_results.items():
            print(f"  {scenario}:")
            print(f"    Requests: {stats['total_requests']}")
            print(f"    Success Rate: {stats['success_rate'] * 100:.2f}%")
            print(f"    Avg Response Time: {stats['average_response_time']:.3f}s")
        
        print("=" * 80)


async def run_load_test():
    """Run load test with default configuration"""
    
    config = LoadTestConfig(
        concurrent_users=20,
        requests_per_user=50,
        test_duration=120,
        ramp_up_time=20
    )
    
    tester = A2ALoadTester(config)
    results = await tester.run_load_test()
    tester.print_results(results)
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_load_test())
