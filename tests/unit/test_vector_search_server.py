#!/usr/bin/env python3
"""
Vector Search MCP Server Test Suite

Comprehensive tests for vector search capabilities including:
- Collection management
- Document addition and search
- Similarity matching
- Performance testing
"""

import time
import json
import requests
from typing import Dict, List, Any
from datetime import datetime

class VectorSearchTester:
    """Test vector search server capabilities"""
    
    def __init__(self, base_url: str = "http://localhost:8004"):
        self.base_url = base_url
        self.results = []
        self.test_collection = "test_collection"
    
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
    
    def test_server_health(self) -> bool:
        """Test server health endpoint"""
        start = time.time()
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            duration = time.time() - start
            
            if response.status_code == 200:
                health_data = response.json()
                
                if health_data.get('status') == 'healthy':
                    performance = health_data.get('performance', {})
                    uptime = performance.get('uptime_seconds', 0)
                    details = f"Status: healthy, Uptime: {uptime}s"
                    self.log_result("Server Health", True, details, duration)
                    return True
                else:
                    self.log_result("Server Health", False, f"Unhealthy status", duration)
                    return False
            else:
                self.log_result("Server Health", False, f"HTTP {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Server Health", False, f"Error: {str(e)}", duration)
            return False
    
    def test_root_endpoint(self) -> bool:
        """Test root endpoint information"""
        start = time.time()
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            duration = time.time() - start
            
            if response.status_code == 200:
                root_data = response.json()
                
                service = root_data.get('service', '')
                capabilities = root_data.get('capabilities', [])
                endpoints = root_data.get('endpoints', {})
                
                if 'Vector Search' in service and len(capabilities) > 0:
                    details = f"Service: {service}, Capabilities: {len(capabilities)}, Endpoints: {len(endpoints)}"
                    self.log_result("Root Endpoint", True, details, duration)
                    return True
                else:
                    self.log_result("Root Endpoint", False, "Missing service info or capabilities", duration)
                    return False
            else:
                self.log_result("Root Endpoint", False, f"HTTP {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Root Endpoint", False, f"Error: {str(e)}", duration)
            return False
    
    def test_collection_creation(self) -> bool:
        """Test creating a vector collection"""
        start = time.time()
        try:
            payload = {
                "name": self.test_collection,
                "dimension": 384
            }
            
            response = requests.post(
                f"{self.base_url}/v1/collections",
                json=payload,
                timeout=30
            )
            
            duration = time.time() - start
            
            if response.status_code == 200:
                result_data = response.json()
                
                if result_data.get('success'):
                    collection_name = result_data.get('data', {}).get('collection_name')
                    dimension = result_data.get('data', {}).get('dimension')
                    details = f"Created collection: {collection_name}, Dimension: {dimension}"
                    self.log_result("Collection Creation", True, details, duration)
                    return True
                else:
                    message = result_data.get('message', 'Unknown error')
                    self.log_result("Collection Creation", False, f"Creation failed: {message}", duration)
                    return False
            else:
                self.log_result("Collection Creation", False, f"HTTP {response.status_code}: {response.text}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Collection Creation", False, f"Error: {str(e)}", duration)
            return False
    
    def test_list_collections(self) -> bool:
        """Test listing collections"""
        start = time.time()
        try:
            response = requests.get(f"{self.base_url}/v1/collections", timeout=10)
            duration = time.time() - start
            
            if response.status_code == 200:
                collections = response.json()
                
                # Should be a list
                if isinstance(collections, list):
                    # Look for our test collection
                    test_collection_found = any(
                        coll.get('name') == self.test_collection for coll in collections
                    )
                    
                    if test_collection_found:
                        details = f"Found {len(collections)} collections including test collection"
                        self.log_result("List Collections", True, details, duration)
                        return True
                    else:
                        details = f"Found {len(collections)} collections but test collection missing"
                        self.log_result("List Collections", False, details, duration)
                        return False
                else:
                    self.log_result("List Collections", False, "Response is not a list", duration)
                    return False
            else:
                self.log_result("List Collections", False, f"HTTP {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("List Collections", False, f"Error: {str(e)}", duration)
            return False
    
    def test_add_documents(self) -> bool:
        """Test adding documents to a collection"""
        start = time.time()
        try:
            # Sample documents
            documents = [
                {
                    "id": "doc1",
                    "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
                    "metadata": {"category": "AI", "type": "definition"}
                },
                {
                    "id": "doc2", 
                    "content": "Natural language processing enables computers to understand human language.",
                    "metadata": {"category": "NLP", "type": "definition"}
                },
                {
                    "id": "doc3",
                    "content": "Deep learning uses neural networks with multiple layers to learn patterns.",
                    "metadata": {"category": "Deep Learning", "type": "definition"}
                }
            ]
            
            payload = {
                "collection": self.test_collection,
                "documents": documents
            }
            
            response = requests.post(
                f"{self.base_url}/v1/collections/{self.test_collection}/documents",
                json=payload,
                timeout=30
            )
            
            duration = time.time() - start
            
            if response.status_code == 200:
                result_data = response.json()
                
                if result_data.get('success'):
                    added_count = result_data.get('data', {}).get('added_count', 0)
                    details = f"Added {added_count} documents to collection"
                    self.log_result("Add Documents", True, details, duration)
                    return True
                else:
                    message = result_data.get('message', 'Unknown error')
                    self.log_result("Add Documents", False, f"Addition failed: {message}", duration)
                    return False
            else:
                self.log_result("Add Documents", False, f"HTTP {response.status_code}: {response.text}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Add Documents", False, f"Error: {str(e)}", duration)
            return False
    
    def test_semantic_search(self) -> bool:
        """Test semantic search functionality"""
        start = time.time()
        try:
            # Search for AI-related content
            payload = {
                "collection": self.test_collection,
                "query_text": "artificial intelligence and machine learning algorithms",
                "limit": 5,
                "similarity_threshold": 0.1
            }
            
            response = requests.post(
                f"{self.base_url}/v1/collections/{self.test_collection}/search",
                json=payload,
                timeout=30
            )
            
            duration = time.time() - start
            
            if response.status_code == 200:
                search_results = response.json()
                
                if isinstance(search_results, list) and len(search_results) > 0:
                    # Check if results have required fields
                    first_result = search_results[0]
                    required_fields = ['document_id', 'content', 'metadata', 'similarity_score']
                    missing_fields = [field for field in required_fields if field not in first_result]
                    
                    if not missing_fields:
                        best_score = max(result['similarity_score'] for result in search_results)
                        details = f"Found {len(search_results)} results, Best score: {best_score:.3f}"
                        self.log_result("Semantic Search", True, details, duration)
                        return True
                    else:
                        self.log_result("Semantic Search", False, f"Missing fields: {missing_fields}", duration)
                        return False
                else:
                    self.log_result("Semantic Search", False, "No search results returned", duration)
                    return False
            else:
                self.log_result("Semantic Search", False, f"HTTP {response.status_code}: {response.text}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Semantic Search", False, f"Error: {str(e)}", duration)
            return False
    
    def test_similarity_threshold(self) -> bool:
        """Test similarity threshold filtering"""
        start = time.time()
        try:
            # Search with high threshold (should return fewer results)
            payload = {
                "collection": self.test_collection,
                "query_text": "completely unrelated topic about cooking recipes",
                "limit": 10,
                "similarity_threshold": 0.8  # High threshold
            }
            
            response = requests.post(
                f"{self.base_url}/v1/collections/{self.test_collection}/search",
                json=payload,
                timeout=30
            )
            
            duration = time.time() - start
            
            if response.status_code == 200:
                search_results = response.json()
                
                # Should return fewer or no results due to high threshold
                if isinstance(search_results, list):
                    # Check that all results meet the threshold
                    valid_results = all(
                        result.get('similarity_score', 0) >= 0.8 
                        for result in search_results
                    )
                    
                    if valid_results:
                        details = f"Threshold filtering working: {len(search_results)} results above 0.8"
                        self.log_result("Similarity Threshold", True, details, duration)
                        return True
                    else:
                        self.log_result("Similarity Threshold", False, "Results below threshold returned", duration)
                        return False
                else:
                    self.log_result("Similarity Threshold", False, "Invalid response format", duration)
                    return False
            else:
                self.log_result("Similarity Threshold", False, f"HTTP {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Similarity Threshold", False, f"Error: {str(e)}", duration)
            return False
    
    def test_performance_multiple_searches(self) -> bool:
        """Test performance with multiple searches"""
        start = time.time()
        try:
            search_queries = [
                "machine learning algorithms",
                "natural language processing",
                "deep learning neural networks",
                "artificial intelligence applications",
                "computer vision techniques"
            ]
            
            total_results = 0
            search_times = []
            
            for query in search_queries:
                search_start = time.time()
                
                payload = {
                    "collection": self.test_collection,
                    "query_text": query,
                    "limit": 3
                }
                
                response = requests.post(
                    f"{self.base_url}/v1/collections/{self.test_collection}/search",
                    json=payload,
                    timeout=30
                )
                
                search_time = time.time() - search_start
                search_times.append(search_time)
                
                if response.status_code == 200:
                    results = response.json()
                    total_results += len(results)
                else:
                    raise Exception(f"Search failed with status {response.status_code}")
            
            duration = time.time() - start
            avg_search_time = sum(search_times) / len(search_times)
            
            details = f"Completed {len(search_queries)} searches, Avg time: {avg_search_time*1000:.1f}ms, Total results: {total_results}"
            self.log_result("Performance Multiple Searches", True, details, duration)
            return True
            
        except Exception as e:
            duration = time.time() - start
            self.log_result("Performance Multiple Searches", False, f"Error: {str(e)}", duration)
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all vector search tests"""
        print("🔍 Vector Search MCP Server Test Suite")
        print("=" * 50)
        
        tests = [
            ("Server Health", self.test_server_health),
            ("Root Endpoint", self.test_root_endpoint),
            ("Collection Creation", self.test_collection_creation),
            ("List Collections", self.test_list_collections),
            ("Add Documents", self.test_add_documents),
            ("Semantic Search", self.test_semantic_search),
            ("Similarity Threshold", self.test_similarity_threshold),
            ("Performance Multiple Searches", self.test_performance_multiple_searches)
        ]
        
        passed = 0
        for test_name, test_func in tests:
            if test_func():
                passed += 1
            print()  # Add spacing
        
        total = len(tests)
        
        print("=" * 50)
        print(f"📊 Vector Search Results: {passed}/{total} tests passed")
        
        return {
            'total_tests': total,
            'passed_tests': passed,
            'success_rate': (passed / total) * 100,
            'test_results': self.results
        }


def main():
    """Main test execution"""
    tester = VectorSearchTester()
    results = tester.run_all_tests()
    
    # Save results
    with open('vector_search_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved to: vector_search_test_results.json")
    return 0 if results['success_rate'] >= 80 else 1


if __name__ == "__main__":
    exit(main())
