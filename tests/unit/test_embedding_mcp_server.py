#!/usr/bin/env python3
"""
Test script for PyGent Factory Embedding MCP Server

This script validates the existing embedding MCP server implementation,
tests all endpoints, and verifies OpenAI compatibility.
"""

import asyncio
import json
import time
import requests
from typing import Dict, List, Any
import openai
from datetime import datetime

class EmbeddingMCPServerTester:
    """Comprehensive test suite for the Embedding MCP Server"""
    
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url
        self.test_results = []
        
        # Configure OpenAI client for compatibility testing
        self.openai_client = openai.OpenAI(
            api_key="not-needed",
            base_url=base_url
        )
    
    def log_test(self, test_name: str, success: bool, details: str = "", duration: float = 0):
        """Log test results"""
        result = {
            'test_name': test_name,
            'success': success,
            'details': details,
            'duration_ms': round(duration * 1000, 2),
            'timestamp': datetime.utcnow().isoformat()
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name} ({duration*1000:.1f}ms)")
        if details:
            print(f"    {details}")
    
    def test_server_health(self) -> bool:
        """Test server health endpoint"""
        start_time = time.time()
        
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                health_data = response.json()
                
                # Validate health response structure
                required_fields = ['status', 'timestamp', 'service', 'providers', 'performance']
                missing_fields = [field for field in required_fields if field not in health_data]
                
                if missing_fields:
                    self.log_test("Health Endpoint", False, f"Missing fields: {missing_fields}", duration)
                    return False
                
                # Check if service is healthy
                if health_data['status'] == 'healthy':
                    provider_info = health_data.get('providers', {})
                    performance_info = health_data.get('performance', {})
                    
                    details = f"Status: {health_data['status']}, Providers: {provider_info.get('provider_count', 0)}"
                    self.log_test("Health Endpoint", True, details, duration)
                    return True
                else:
                    self.log_test("Health Endpoint", False, f"Service unhealthy: {health_data['status']}", duration)
                    return False
            else:
                self.log_test("Health Endpoint", False, f"HTTP {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Health Endpoint", False, f"Exception: {str(e)}", duration)
            return False
    
    def test_root_endpoint(self) -> bool:
        """Test root endpoint"""
        start_time = time.time()
        
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                root_data = response.json()
                
                # Validate root response
                if 'service' in root_data and 'endpoints' in root_data:
                    details = f"Service: {root_data['service']}, Version: {root_data.get('version', 'unknown')}"
                    self.log_test("Root Endpoint", True, details, duration)
                    return True
                else:
                    self.log_test("Root Endpoint", False, "Invalid response structure", duration)
                    return False
            else:
                self.log_test("Root Endpoint", False, f"HTTP {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Root Endpoint", False, f"Exception: {str(e)}", duration)
            return False
    
    def test_embeddings_endpoint_direct(self) -> bool:
        """Test embeddings endpoint with direct HTTP request"""
        start_time = time.time()
        
        try:
            payload = {
                "input": "Hello, world! This is a test embedding.",
                "model": "text-embedding-ada-002"
            }
            
            response = requests.post(
                f"{self.base_url}/v1/embeddings",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                embedding_data = response.json()
                
                # Validate response structure
                required_fields = ['object', 'data', 'model', 'usage']
                missing_fields = [field for field in required_fields if field not in embedding_data]
                
                if missing_fields:
                    self.log_test("Embeddings Direct", False, f"Missing fields: {missing_fields}", duration)
                    return False
                
                # Validate data structure
                if embedding_data['data'] and len(embedding_data['data']) > 0:
                    first_embedding = embedding_data['data'][0]
                    if 'embedding' in first_embedding and isinstance(first_embedding['embedding'], list):
                        embedding_dim = len(first_embedding['embedding'])
                        details = f"Embedding dimension: {embedding_dim}, Model: {embedding_data['model']}"
                        self.log_test("Embeddings Direct", True, details, duration)
                        return True
                    else:
                        self.log_test("Embeddings Direct", False, "Invalid embedding data structure", duration)
                        return False
                else:
                    self.log_test("Embeddings Direct", False, "No embedding data returned", duration)
                    return False
            else:
                self.log_test("Embeddings Direct", False, f"HTTP {response.status_code}: {response.text}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Embeddings Direct", False, f"Exception: {str(e)}", duration)
            return False
    
    def test_openai_compatibility(self) -> bool:
        """Test OpenAI SDK compatibility"""
        start_time = time.time()
        
        try:
            response = self.openai_client.embeddings.create(
                input="This is a test of OpenAI SDK compatibility with PyGent Factory Embedding Server.",
                model="text-embedding-ada-002"
            )
            
            duration = time.time() - start_time
            
            if response and response.data and len(response.data) > 0:
                embedding = response.data[0].embedding
                if isinstance(embedding, list) and len(embedding) > 0:
                    details = f"SDK compatible, embedding dimension: {len(embedding)}"
                    self.log_test("OpenAI SDK Compatibility", True, details, duration)
                    return True
                else:
                    self.log_test("OpenAI SDK Compatibility", False, "Invalid embedding format", duration)
                    return False
            else:
                self.log_test("OpenAI SDK Compatibility", False, "No response data", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("OpenAI SDK Compatibility", False, f"Exception: {str(e)}", duration)
            return False
    
    def test_batch_processing(self) -> bool:
        """Test batch embedding processing"""
        start_time = time.time()
        
        try:
            texts = [
                "First test sentence for batch processing.",
                "Second test sentence for batch processing.",
                "Third test sentence for batch processing.",
                "Fourth test sentence for batch processing.",
                "Fifth test sentence for batch processing."
            ]
            
            response = self.openai_client.embeddings.create(
                input=texts,
                model="text-embedding-ada-002"
            )
            
            duration = time.time() - start_time
            
            if response and response.data and len(response.data) == len(texts):
                # Verify all embeddings are valid
                valid_embeddings = 0
                for i, embedding_data in enumerate(response.data):
                    if embedding_data.embedding and len(embedding_data.embedding) > 0:
                        valid_embeddings += 1
                
                if valid_embeddings == len(texts):
                    details = f"Processed {len(texts)} texts successfully"
                    self.log_test("Batch Processing", True, details, duration)
                    return True
                else:
                    self.log_test("Batch Processing", False, f"Only {valid_embeddings}/{len(texts)} valid embeddings", duration)
                    return False
            else:
                expected = len(texts)
                actual = len(response.data) if response and response.data else 0
                self.log_test("Batch Processing", False, f"Expected {expected} embeddings, got {actual}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Batch Processing", False, f"Exception: {str(e)}", duration)
            return False
    
    def test_performance_metrics(self) -> bool:
        """Test performance and collect metrics"""
        start_time = time.time()
        
        try:
            # Perform multiple requests to test performance
            test_text = "Performance testing text for embedding generation."
            request_times = []
            
            for i in range(5):
                req_start = time.time()
                response = self.openai_client.embeddings.create(
                    input=test_text,
                    model="text-embedding-ada-002"
                )
                req_duration = time.time() - req_start
                request_times.append(req_duration)
                
                if not response or not response.data:
                    self.log_test("Performance Metrics", False, f"Request {i+1} failed", time.time() - start_time)
                    return False
            
            duration = time.time() - start_time
            
            # Calculate performance metrics
            avg_time = sum(request_times) / len(request_times)
            min_time = min(request_times)
            max_time = max(request_times)
            
            details = f"Avg: {avg_time*1000:.1f}ms, Min: {min_time*1000:.1f}ms, Max: {max_time*1000:.1f}ms"
            self.log_test("Performance Metrics", True, details, duration)
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Performance Metrics", False, f"Exception: {str(e)}", duration)
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results"""
        print("🧪 Starting PyGent Factory Embedding MCP Server Tests")
        print("=" * 60)
        
        # Test sequence
        tests = [
            ("Server Health", self.test_server_health),
            ("Root Endpoint", self.test_root_endpoint),
            ("Embeddings Direct", self.test_embeddings_endpoint_direct),
            ("OpenAI Compatibility", self.test_openai_compatibility),
            ("Batch Processing", self.test_batch_processing),
            ("Performance Metrics", self.test_performance_metrics)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            if test_func():
                passed += 1
        
        print("=" * 60)
        print(f"📊 Test Results: {passed}/{total} tests passed")
        
        # Generate summary
        summary = {
            'total_tests': total,
            'passed_tests': passed,
            'failed_tests': total - passed,
            'success_rate': (passed / total) * 100,
            'test_details': self.test_results,
            'overall_status': 'PASS' if passed == total else 'FAIL',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return summary


def main():
    """Main test execution"""
    tester = EmbeddingMCPServerTester()
    
    print("🚀 PyGent Factory Embedding MCP Server Test Suite")
    print("Testing server at: http://localhost:8002")
    print()
    
    # Run tests
    results = tester.run_all_tests()
    
    # Save results
    with open('embedding_server_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: embedding_server_test_results.json")
    
    # Exit with appropriate code
    exit_code = 0 if results['overall_status'] == 'PASS' else 1
    return exit_code


if __name__ == "__main__":
    exit(main())
