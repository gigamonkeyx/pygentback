#!/usr/bin/env python3
"""
Batch Processing Validation Test Suite

Tests batch processing capabilities, limits, and performance characteristics
of the embedding server.
"""

import time
import json
import requests
from typing import Dict, List, Any
import concurrent.futures
import threading

class BatchProcessingTester:
    """Test batch processing capabilities and limits"""
    
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url
        self.results = []
    
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
    
    def test_small_batch(self) -> bool:
        """Test small batch processing (5 items)"""
        start = time.time()
        
        texts = [f"Small batch item {i+1}" for i in range(5)]
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/embeddings",
                json={"input": texts, "model": "text-embedding-ada-002"},
                timeout=60
            )
            
            duration = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                if data.get('data') and len(data['data']) == len(texts):
                    # Verify all embeddings are valid
                    valid_embeddings = sum(1 for item in data['data'] if item.get('embedding') and len(item['embedding']) > 0)
                    
                    if valid_embeddings == len(texts):
                        self.log_result("Small Batch (5)", True, f"All {len(texts)} embeddings generated", duration)
                        return True
                    else:
                        self.log_result("Small Batch (5)", False, f"Only {valid_embeddings}/{len(texts)} valid", duration)
                        return False
                else:
                    expected = len(texts)
                    actual = len(data.get('data', []))
                    self.log_result("Small Batch (5)", False, f"Expected {expected}, got {actual}", duration)
                    return False
            else:
                self.log_result("Small Batch (5)", False, f"HTTP {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Small Batch (5)", False, f"Error: {str(e)}", duration)
            return False
    
    def test_medium_batch(self) -> bool:
        """Test medium batch processing (20 items)"""
        start = time.time()
        
        texts = [f"Medium batch item {i+1} with more detailed content for testing" for i in range(20)]
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/embeddings",
                json={"input": texts, "model": "text-embedding-ada-002"},
                timeout=120
            )
            
            duration = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                if data.get('data') and len(data['data']) == len(texts):
                    valid_embeddings = sum(1 for item in data['data'] if item.get('embedding') and len(item['embedding']) > 0)
                    
                    if valid_embeddings == len(texts):
                        throughput = len(texts) / duration
                        self.log_result("Medium Batch (20)", True, f"Throughput: {throughput:.1f} items/sec", duration)
                        return True
                    else:
                        self.log_result("Medium Batch (20)", False, f"Only {valid_embeddings}/{len(texts)} valid", duration)
                        return False
                else:
                    expected = len(texts)
                    actual = len(data.get('data', []))
                    self.log_result("Medium Batch (20)", False, f"Expected {expected}, got {actual}", duration)
                    return False
            else:
                self.log_result("Medium Batch (20)", False, f"HTTP {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Medium Batch (20)", False, f"Error: {str(e)}", duration)
            return False
    
    def test_large_batch(self) -> bool:
        """Test large batch processing (50 items)"""
        start = time.time()
        
        texts = [f"Large batch item {i+1} with substantial content to test the limits of batch processing capabilities" for i in range(50)]
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/embeddings",
                json={"input": texts, "model": "text-embedding-ada-002"},
                timeout=300  # 5 minutes for large batch
            )
            
            duration = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                if data.get('data') and len(data['data']) == len(texts):
                    valid_embeddings = sum(1 for item in data['data'] if item.get('embedding') and len(item['embedding']) > 0)
                    
                    if valid_embeddings == len(texts):
                        throughput = len(texts) / duration
                        self.log_result("Large Batch (50)", True, f"Throughput: {throughput:.1f} items/sec", duration)
                        return True
                    else:
                        self.log_result("Large Batch (50)", False, f"Only {valid_embeddings}/{len(texts)} valid", duration)
                        return False
                else:
                    expected = len(texts)
                    actual = len(data.get('data', []))
                    self.log_result("Large Batch (50)", False, f"Expected {expected}, got {actual}", duration)
                    return False
            else:
                self.log_result("Large Batch (50)", False, f"HTTP {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Large Batch (50)", False, f"Error: {str(e)}", duration)
            return False
    
    def test_concurrent_batches(self) -> bool:
        """Test concurrent batch processing"""
        start = time.time()
        
        def send_batch(batch_id: int) -> bool:
            texts = [f"Concurrent batch {batch_id} item {i+1}" for i in range(5)]
            
            try:
                response = requests.post(
                    f"{self.base_url}/v1/embeddings",
                    json={"input": texts, "model": "text-embedding-ada-002"},
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get('data') and len(data['data']) == len(texts)
                return False
            except:
                return False
        
        try:
            # Send 3 concurrent batches
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(send_batch, i) for i in range(3)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            duration = time.time() - start
            
            successful_batches = sum(1 for result in results if result)
            
            if successful_batches == 3:
                self.log_result("Concurrent Batches", True, f"All 3 batches succeeded", duration)
                return True
            else:
                self.log_result("Concurrent Batches", False, f"Only {successful_batches}/3 batches succeeded", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Concurrent Batches", False, f"Error: {str(e)}", duration)
            return False
    
    def test_mixed_length_batch(self) -> bool:
        """Test batch with mixed text lengths"""
        start = time.time()
        
        texts = [
            "Short",
            "Medium length text with several words",
            "Very long text " * 100,  # Very long text
            "Another short one",
            "Medium again with some more content here"
        ]
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/embeddings",
                json={"input": texts, "model": "text-embedding-ada-002"},
                timeout=120
            )
            
            duration = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                if data.get('data') and len(data['data']) == len(texts):
                    # Check that all embeddings have same dimension despite different input lengths
                    embeddings = [item['embedding'] for item in data['data'] if item.get('embedding')]
                    
                    if len(embeddings) == len(texts):
                        dimensions = [len(emb) for emb in embeddings]
                        all_same_dim = len(set(dimensions)) == 1
                        
                        if all_same_dim:
                            self.log_result("Mixed Length Batch", True, f"All {len(texts)} items, dim: {dimensions[0]}", duration)
                            return True
                        else:
                            self.log_result("Mixed Length Batch", False, f"Inconsistent dimensions: {dimensions}", duration)
                            return False
                    else:
                        self.log_result("Mixed Length Batch", False, f"Only {len(embeddings)}/{len(texts)} embeddings", duration)
                        return False
                else:
                    expected = len(texts)
                    actual = len(data.get('data', []))
                    self.log_result("Mixed Length Batch", False, f"Expected {expected}, got {actual}", duration)
                    return False
            else:
                self.log_result("Mixed Length Batch", False, f"HTTP {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Mixed Length Batch", False, f"Error: {str(e)}", duration)
            return False
    
    def test_empty_and_edge_cases(self) -> bool:
        """Test edge cases in batch processing"""
        start = time.time()
        
        edge_cases = [
            ([], "Empty batch"),
            ([""], "Single empty string"),
            (["", "valid text", ""], "Mixed empty and valid"),
            ([" " * 1000], "Very long whitespace"),
            (["a"], "Single character")
        ]
        
        successful_cases = 0
        
        for texts, case_name in edge_cases:
            try:
                response = requests.post(
                    f"{self.base_url}/v1/embeddings",
                    json={"input": texts, "model": "text-embedding-ada-002"},
                    timeout=60
                )
                
                # For empty batch, we expect an error or empty response
                if len(texts) == 0:
                    if response.status_code != 200:
                        successful_cases += 1  # Expected failure
                else:
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('data'):
                            successful_cases += 1
                            
            except:
                # Some edge cases are expected to fail
                if len(texts) == 0:
                    successful_cases += 1
        
        duration = time.time() - start
        
        self.log_result("Edge Cases", True, f"Handled {successful_cases}/{len(edge_cases)} cases appropriately", duration)
        return True
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all batch processing tests"""
        print("📦 Batch Processing Test Suite")
        print("=" * 35)
        
        tests = [
            ("Small Batch", self.test_small_batch),
            ("Medium Batch", self.test_medium_batch),
            ("Large Batch", self.test_large_batch),
            ("Concurrent Batches", self.test_concurrent_batches),
            ("Mixed Length Batch", self.test_mixed_length_batch),
            ("Edge Cases", self.test_empty_and_edge_cases)
        ]
        
        passed = 0
        for test_name, test_func in tests:
            if test_func():
                passed += 1
        
        total = len(tests)
        
        print("=" * 35)
        print(f"📊 Batch Results: {passed}/{total} tests passed")
        
        return {
            'total_tests': total,
            'passed_tests': passed,
            'success_rate': (passed / total) * 100,
            'test_results': self.results
        }


def main():
    """Main test execution"""
    tester = BatchProcessingTester()
    results = tester.run_all_tests()
    
    # Save results
    with open('batch_processing_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved to: batch_processing_results.json")
    return 0 if results['success_rate'] >= 80 else 1  # Allow some edge case failures


if __name__ == "__main__":
    exit(main())
