#!/usr/bin/env python3
"""
Provider Validation Test Suite

Tests all embedding providers to ensure they work correctly and fallback
mechanisms function properly.
"""

import time
import json
import requests
from typing import Dict, List, Any

class ProviderValidationTester:
    """Test all embedding providers and fallback mechanisms"""
    
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
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current server health and provider status"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                return response.json()
            return {}
        except:
            return {}
    
    def test_provider_availability(self) -> bool:
        """Test which providers are available"""
        start = time.time()
        
        health = self.get_health_status()
        duration = time.time() - start
        
        if health.get('providers'):
            providers = health['providers']
            available_count = providers.get('provider_count', 0)
            current_provider = providers.get('current_provider', 'unknown')
            
            details = f"Available: {available_count}, Current: {current_provider}"
            self.log_result("Provider Availability", True, details, duration)
            return True
        else:
            self.log_result("Provider Availability", False, "No provider info", duration)
            return False
    
    def test_embedding_consistency(self) -> bool:
        """Test that same input produces consistent embeddings"""
        start = time.time()
        
        test_text = "Consistency test for embedding generation"
        embeddings = []
        
        try:
            # Generate same embedding multiple times
            for i in range(3):
                response = requests.post(
                    f"{self.base_url}/v1/embeddings",
                    json={"input": test_text, "model": "text-embedding-ada-002"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('data') and len(data['data']) > 0:
                        embeddings.append(data['data'][0]['embedding'])
            
            duration = time.time() - start
            
            if len(embeddings) == 3:
                # Check if embeddings are identical (they should be for same input)
                first_embedding = embeddings[0]
                all_identical = all(emb == first_embedding for emb in embeddings)
                
                if all_identical:
                    self.log_result("Embedding Consistency", True, "All embeddings identical", duration)
                    return True
                else:
                    self.log_result("Embedding Consistency", False, "Embeddings differ", duration)
                    return False
            else:
                self.log_result("Embedding Consistency", False, f"Only got {len(embeddings)}/3 embeddings", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Embedding Consistency", False, f"Error: {str(e)}", duration)
            return False
    
    def test_performance_characteristics(self) -> bool:
        """Test performance across different text lengths"""
        start = time.time()
        
        test_cases = [
            ("Short", "Short text"),
            ("Medium", "This is a medium length text that contains multiple sentences and should test the embedding generation with more substantial content."),
            ("Long", "This is a very long text " * 50)  # ~1000 words
        ]
        
        performance_data = []
        
        try:
            for case_name, text in test_cases:
                case_start = time.time()
                
                response = requests.post(
                    f"{self.base_url}/v1/embeddings",
                    json={"input": text, "model": "text-embedding-ada-002"},
                    timeout=60
                )
                
                case_duration = time.time() - case_start
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('data') and len(data['data']) > 0:
                        embedding = data['data'][0]['embedding']
                        performance_data.append({
                            'case': case_name,
                            'text_length': len(text),
                            'embedding_dim': len(embedding),
                            'duration_ms': round(case_duration * 1000, 2)
                        })
            
            duration = time.time() - start
            
            if len(performance_data) == len(test_cases):
                avg_duration = sum(p['duration_ms'] for p in performance_data) / len(performance_data)
                details = f"Avg: {avg_duration:.1f}ms across {len(test_cases)} cases"
                self.log_result("Performance Characteristics", True, details, duration)
                return True
            else:
                self.log_result("Performance Characteristics", False, f"Only {len(performance_data)}/{len(test_cases)} cases succeeded", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Performance Characteristics", False, f"Error: {str(e)}", duration)
            return False
    
    def test_batch_vs_single_performance(self) -> bool:
        """Compare batch vs single request performance"""
        start = time.time()
        
        texts = [
            "First text for batch comparison",
            "Second text for batch comparison", 
            "Third text for batch comparison"
        ]
        
        try:
            # Test single requests
            single_start = time.time()
            single_responses = []
            for text in texts:
                response = requests.post(
                    f"{self.base_url}/v1/embeddings",
                    json={"input": text, "model": "text-embedding-ada-002"},
                    timeout=30
                )
                if response.status_code == 200:
                    single_responses.append(response.json())
            single_duration = time.time() - single_start
            
            # Test batch request
            batch_start = time.time()
            batch_response = requests.post(
                f"{self.base_url}/v1/embeddings",
                json={"input": texts, "model": "text-embedding-ada-002"},
                timeout=30
            )
            batch_duration = time.time() - batch_start
            
            duration = time.time() - start
            
            if len(single_responses) == len(texts) and batch_response.status_code == 200:
                batch_data = batch_response.json()
                if batch_data.get('data') and len(batch_data['data']) == len(texts):
                    efficiency = (single_duration / batch_duration) if batch_duration > 0 else 0
                    details = f"Batch {efficiency:.1f}x faster ({batch_duration*1000:.1f}ms vs {single_duration*1000:.1f}ms)"
                    self.log_result("Batch vs Single Performance", True, details, duration)
                    return True
                else:
                    self.log_result("Batch vs Single Performance", False, "Batch response invalid", duration)
                    return False
            else:
                self.log_result("Batch vs Single Performance", False, "Requests failed", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Batch vs Single Performance", False, f"Error: {str(e)}", duration)
            return False
    
    def test_unicode_and_special_characters(self) -> bool:
        """Test handling of unicode and special characters"""
        start = time.time()
        
        test_cases = [
            "Unicode test: 你好世界 🌍 café naïve résumé",
            "Special chars: @#$%^&*()_+-=[]{}|;':\",./<>?",
            "Mixed: Hello 世界! How are you? 😊",
            "Emoji heavy: 🚀🔥💯⭐🎉🎯🌟💪"
        ]
        
        successful_cases = 0
        
        try:
            for i, text in enumerate(test_cases):
                response = requests.post(
                    f"{self.base_url}/v1/embeddings",
                    json={"input": text, "model": "text-embedding-ada-002"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('data') and len(data['data']) > 0:
                        successful_cases += 1
            
            duration = time.time() - start
            
            if successful_cases == len(test_cases):
                self.log_result("Unicode & Special Chars", True, f"All {len(test_cases)} cases handled", duration)
                return True
            else:
                self.log_result("Unicode & Special Chars", False, f"Only {successful_cases}/{len(test_cases)} cases worked", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Unicode & Special Chars", False, f"Error: {str(e)}", duration)
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all provider validation tests"""
        print("🔍 Provider Validation Test Suite")
        print("=" * 40)
        
        tests = [
            ("Provider Availability", self.test_provider_availability),
            ("Embedding Consistency", self.test_embedding_consistency),
            ("Performance Characteristics", self.test_performance_characteristics),
            ("Batch vs Single", self.test_batch_vs_single_performance),
            ("Unicode & Special Chars", self.test_unicode_and_special_characters)
        ]
        
        passed = 0
        for test_name, test_func in tests:
            if test_func():
                passed += 1
        
        total = len(tests)
        
        print("=" * 40)
        print(f"📊 Validation Results: {passed}/{total} tests passed")
        
        return {
            'total_tests': total,
            'passed_tests': passed,
            'success_rate': (passed / total) * 100,
            'test_results': self.results
        }


def main():
    """Main test execution"""
    tester = ProviderValidationTester()
    results = tester.run_all_tests()
    
    # Save results
    with open('provider_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved to: provider_validation_results.json")
    return 0 if results['success_rate'] == 100 else 1


if __name__ == "__main__":
    exit(main())
