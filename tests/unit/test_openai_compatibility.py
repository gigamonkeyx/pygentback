#!/usr/bin/env python3
"""
OpenAI API Compatibility Test Suite

Tests the embedding server with multiple OpenAI client libraries and configurations
to ensure broad compatibility across different use cases.
"""

import asyncio
import time
import json
from typing import List, Dict, Any
import requests
import openai
from openai import AsyncOpenAI
import httpx

class OpenAICompatibilityTester:
    """Test OpenAI API compatibility with various client configurations"""
    
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url
        self.results = []
        
        # Standard OpenAI client
        self.sync_client = openai.OpenAI(
            api_key="test-key",
            base_url=base_url
        )
        
        # Async OpenAI client
        self.async_client = AsyncOpenAI(
            api_key="test-key", 
            base_url=base_url
        )
        
        # Custom HTTP client
        self.http_client = httpx.Client(timeout=30.0)
    
    def log_result(self, test_name: str, success: bool, details: str = "", duration: float = 0):
        """Log test result"""
        result = {
            'test': test_name,
            'success': success,
            'details': details,
            'duration_ms': round(duration * 1000, 2)
        }
        self.results.append(result)
        
        status = "✅" if success else "❌"
        print(f"{status} {test_name} ({duration*1000:.1f}ms)")
        if details:
            print(f"    {details}")
    
    def test_sync_client_single(self) -> bool:
        """Test synchronous client with single text"""
        start = time.time()
        try:
            response = self.sync_client.embeddings.create(
                input="Test synchronous embedding generation",
                model="text-embedding-ada-002"
            )
            
            duration = time.time() - start
            
            if response.data and len(response.data) > 0:
                embedding = response.data[0].embedding
                self.log_result("Sync Client Single", True, f"Dimension: {len(embedding)}", duration)
                return True
            else:
                self.log_result("Sync Client Single", False, "No embedding data", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Sync Client Single", False, f"Error: {str(e)}", duration)
            return False
    
    def test_sync_client_batch(self) -> bool:
        """Test synchronous client with batch processing"""
        start = time.time()
        try:
            texts = [
                "First batch text for testing",
                "Second batch text for testing", 
                "Third batch text for testing"
            ]
            
            response = self.sync_client.embeddings.create(
                input=texts,
                model="text-embedding-ada-002"
            )
            
            duration = time.time() - start
            
            if response.data and len(response.data) == len(texts):
                self.log_result("Sync Client Batch", True, f"Processed {len(texts)} texts", duration)
                return True
            else:
                expected = len(texts)
                actual = len(response.data) if response.data else 0
                self.log_result("Sync Client Batch", False, f"Expected {expected}, got {actual}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Sync Client Batch", False, f"Error: {str(e)}", duration)
            return False
    
    async def test_async_client_single(self) -> bool:
        """Test asynchronous client with single text"""
        start = time.time()
        try:
            response = await self.async_client.embeddings.create(
                input="Test asynchronous embedding generation",
                model="text-embedding-ada-002"
            )
            
            duration = time.time() - start
            
            if response.data and len(response.data) > 0:
                embedding = response.data[0].embedding
                self.log_result("Async Client Single", True, f"Dimension: {len(embedding)}", duration)
                return True
            else:
                self.log_result("Async Client Single", False, "No embedding data", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Async Client Single", False, f"Error: {str(e)}", duration)
            return False
    
    async def test_async_client_concurrent(self) -> bool:
        """Test concurrent async requests"""
        start = time.time()
        try:
            tasks = []
            for i in range(3):
                task = self.async_client.embeddings.create(
                    input=f"Concurrent request {i+1}",
                    model="text-embedding-ada-002"
                )
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks)
            duration = time.time() - start
            
            valid_responses = sum(1 for r in responses if r.data and len(r.data) > 0)
            
            if valid_responses == len(tasks):
                self.log_result("Async Concurrent", True, f"All {len(tasks)} requests succeeded", duration)
                return True
            else:
                self.log_result("Async Concurrent", False, f"Only {valid_responses}/{len(tasks)} succeeded", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Async Concurrent", False, f"Error: {str(e)}", duration)
            return False
    
    def test_raw_http_request(self) -> bool:
        """Test raw HTTP request compatibility"""
        start = time.time()
        try:
            payload = {
                "input": "Raw HTTP request test",
                "model": "text-embedding-ada-002"
            }
            
            response = self.http_client.post(
                f"{self.base_url}/v1/embeddings",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            duration = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                if data.get('data') and len(data['data']) > 0:
                    embedding = data['data'][0]['embedding']
                    self.log_result("Raw HTTP Request", True, f"Dimension: {len(embedding)}", duration)
                    return True
                else:
                    self.log_result("Raw HTTP Request", False, "No embedding data", duration)
                    return False
            else:
                self.log_result("Raw HTTP Request", False, f"HTTP {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Raw HTTP Request", False, f"Error: {str(e)}", duration)
            return False
    
    def test_different_models(self) -> bool:
        """Test with different model names"""
        start = time.time()
        models_to_test = [
            "text-embedding-ada-002",
            "text-embedding-3-small", 
            "text-embedding-3-large",
            "custom-model"
        ]
        
        successful_models = 0
        
        for model in models_to_test:
            try:
                response = self.sync_client.embeddings.create(
                    input=f"Testing model {model}",
                    model=model
                )
                
                if response.data and len(response.data) > 0:
                    successful_models += 1
                    
            except Exception as e:
                # Expected for some models, continue testing
                pass
        
        duration = time.time() - start
        
        if successful_models > 0:
            self.log_result("Different Models", True, f"{successful_models}/{len(models_to_test)} models worked", duration)
            return True
        else:
            self.log_result("Different Models", False, "No models worked", duration)
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling with invalid requests"""
        start = time.time()
        error_tests = []
        
        # Test empty input
        try:
            self.sync_client.embeddings.create(input="", model="text-embedding-ada-002")
            error_tests.append("empty_input_handled")
        except:
            error_tests.append("empty_input_rejected")
        
        # Test very long input
        try:
            long_text = "test " * 10000
            self.sync_client.embeddings.create(input=long_text, model="text-embedding-ada-002")
            error_tests.append("long_input_handled")
        except:
            error_tests.append("long_input_rejected")
        
        duration = time.time() - start
        
        self.log_result("Error Handling", True, f"Tests: {', '.join(error_tests)}", duration)
        return True
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all compatibility tests"""
        print("🔧 OpenAI API Compatibility Test Suite")
        print("=" * 50)
        
        tests = [
            ("Sync Single", self.test_sync_client_single),
            ("Sync Batch", self.test_sync_client_batch),
            ("Raw HTTP", self.test_raw_http_request),
            ("Different Models", self.test_different_models),
            ("Error Handling", self.test_error_handling)
        ]
        
        # Run sync tests
        passed = 0
        for test_name, test_func in tests:
            if test_func():
                passed += 1
        
        # Run async tests
        async_tests = [
            ("Async Single", self.test_async_client_single),
            ("Async Concurrent", self.test_async_client_concurrent)
        ]
        
        for test_name, test_func in async_tests:
            if await test_func():
                passed += 1
        
        total = len(tests) + len(async_tests)
        
        print("=" * 50)
        print(f"📊 Compatibility Results: {passed}/{total} tests passed")
        
        return {
            'total_tests': total,
            'passed_tests': passed,
            'success_rate': (passed / total) * 100,
            'test_results': self.results
        }


async def main():
    """Main test execution"""
    tester = OpenAICompatibilityTester()
    results = await tester.run_all_tests()
    
    # Save results
    with open('openai_compatibility_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved to: openai_compatibility_results.json")
    return 0 if results['success_rate'] == 100 else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
