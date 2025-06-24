#!/usr/bin/env python3
"""
Document Processing MCP Server Test Suite

Comprehensive tests for document processing capabilities including:
- Document download and processing
- Text extraction methods
- Quality assessment
- Content analysis
- Error handling
"""

import time
import json
import requests
from typing import Dict, List, Any
from datetime import datetime

class DocumentProcessingTester:
    """Test document processing server capabilities"""
    
    def __init__(self, base_url: str = "http://localhost:8003"):
        self.base_url = base_url
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
    
    def test_server_health(self) -> bool:
        """Test server health endpoint"""
        start = time.time()
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            duration = time.time() - start
            
            if response.status_code == 200:
                health_data = response.json()
                
                # Validate health response structure
                required_fields = ['status', 'timestamp', 'service', 'capabilities', 'performance']
                missing_fields = [field for field in required_fields if field not in health_data]
                
                if missing_fields:
                    self.log_result("Server Health", False, f"Missing fields: {missing_fields}", duration)
                    return False
                
                if health_data['status'] == 'healthy':
                    capabilities = health_data.get('capabilities', {})
                    performance = health_data.get('performance', {})
                    
                    details = f"Status: {health_data['status']}, Capabilities: {len(capabilities)}"
                    self.log_result("Server Health", True, details, duration)
                    return True
                else:
                    self.log_result("Server Health", False, f"Unhealthy status: {health_data['status']}", duration)
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
                
                # Validate root response
                required_fields = ['service', 'version', 'endpoints', 'capabilities']
                missing_fields = [field for field in required_fields if field not in root_data]
                
                if missing_fields:
                    self.log_result("Root Endpoint", False, f"Missing fields: {missing_fields}", duration)
                    return False
                
                capabilities = root_data.get('capabilities', [])
                endpoints = root_data.get('endpoints', {})
                
                details = f"Service: {root_data['service']}, Capabilities: {len(capabilities)}, Endpoints: {len(endpoints)}"
                self.log_result("Root Endpoint", True, details, duration)
                return True
            else:
                self.log_result("Root Endpoint", False, f"HTTP {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Root Endpoint", False, f"Error: {str(e)}", duration)
            return False
    
    def test_document_processing_simple(self) -> bool:
        """Test simple document processing with a known PDF"""
        start = time.time()
        try:
            # Use a simple, reliable PDF URL for testing
            test_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
            
            payload = {
                "url": test_url,
                "metadata": {
                    "test_document": True,
                    "source": "w3.org"
                },
                "extract_method": "auto",
                "quality_threshold": 0.5
            }
            
            response = requests.post(
                f"{self.base_url}/v1/documents/process",
                json=payload,
                timeout=60
            )
            
            duration = time.time() - start
            
            if response.status_code == 200:
                result_data = response.json()
                
                # Validate response structure
                required_fields = ['document_id', 'status', 'processing_time']
                missing_fields = [field for field in required_fields if field not in result_data]
                
                if missing_fields:
                    self.log_result("Simple Document Processing", False, f"Missing fields: {missing_fields}", duration)
                    return False
                
                if result_data['status'] == 'completed':
                    doc_id = result_data['document_id']
                    proc_time = result_data.get('processing_time', 0)
                    quality = result_data.get('quality_score', 0)
                    
                    details = f"Doc ID: {doc_id[:8]}..., Quality: {quality:.2f}, Proc Time: {proc_time:.2f}s"
                    self.log_result("Simple Document Processing", True, details, duration)
                    return True
                else:
                    error_msg = result_data.get('error_message', 'Unknown error')
                    self.log_result("Simple Document Processing", False, f"Processing failed: {error_msg}", duration)
                    return False
            else:
                self.log_result("Simple Document Processing", False, f"HTTP {response.status_code}: {response.text}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Simple Document Processing", False, f"Error: {str(e)}", duration)
            return False
    
    def test_document_processing_with_metadata(self) -> bool:
        """Test document processing with rich metadata"""
        start = time.time()
        try:
            # Use a different test PDF
            test_url = "https://www.adobe.com/support/products/enterprise/knowledgecenter/media/c4611_sample_explain.pdf"
            
            payload = {
                "url": test_url,
                "metadata": {
                    "title": "Adobe Sample Document",
                    "category": "technical",
                    "source": "adobe.com",
                    "test_type": "metadata_enrichment"
                },
                "extract_method": "standard",
                "quality_threshold": 0.7,
                "max_pages": 5
            }
            
            response = requests.post(
                f"{self.base_url}/v1/documents/process",
                json=payload,
                timeout=90
            )
            
            duration = time.time() - start
            
            if response.status_code == 200:
                result_data = response.json()
                
                if result_data['status'] == 'completed':
                    metadata = result_data.get('metadata', {})
                    text_content = result_data.get('text_content', '')
                    
                    # Validate metadata enrichment
                    has_analysis = 'analysis' in metadata
                    has_user_metadata = 'user_metadata' in metadata
                    has_text_content = len(text_content) > 0
                    
                    if has_analysis and has_user_metadata and has_text_content:
                        analysis = metadata['analysis']
                        categories = analysis.get('categories', [])
                        
                        details = f"Metadata enriched, Categories: {categories}, Text length: {len(text_content)}"
                        self.log_result("Metadata Enrichment", True, details, duration)
                        return True
                    else:
                        missing = []
                        if not has_analysis: missing.append("analysis")
                        if not has_user_metadata: missing.append("user_metadata")
                        if not has_text_content: missing.append("text_content")
                        
                        self.log_result("Metadata Enrichment", False, f"Missing: {missing}", duration)
                        return False
                else:
                    error_msg = result_data.get('error_message', 'Processing failed')
                    self.log_result("Metadata Enrichment", False, error_msg, duration)
                    return False
            else:
                self.log_result("Metadata Enrichment", False, f"HTTP {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Metadata Enrichment", False, f"Error: {str(e)}", duration)
            return False
    
    def test_quality_assessment(self) -> bool:
        """Test quality assessment functionality"""
        start = time.time()
        try:
            # Test with a document that should have good quality
            test_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
            
            payload = {
                "url": test_url,
                "metadata": {"test_type": "quality_assessment"},
                "extract_method": "auto",
                "quality_threshold": 0.3  # Lower threshold to test quality scoring
            }
            
            response = requests.post(
                f"{self.base_url}/v1/documents/process",
                json=payload,
                timeout=60
            )
            
            duration = time.time() - start
            
            if response.status_code == 200:
                result_data = response.json()
                
                if result_data['status'] == 'completed':
                    quality_score = result_data.get('quality_score')
                    extraction_method = result_data.get('extraction_method')
                    
                    if quality_score is not None and extraction_method:
                        # Quality score should be between 0 and 1
                        if 0 <= quality_score <= 1:
                            details = f"Quality: {quality_score:.3f}, Method: {extraction_method}"
                            self.log_result("Quality Assessment", True, details, duration)
                            return True
                        else:
                            self.log_result("Quality Assessment", False, f"Invalid quality score: {quality_score}", duration)
                            return False
                    else:
                        self.log_result("Quality Assessment", False, "Missing quality score or extraction method", duration)
                        return False
                else:
                    error_msg = result_data.get('error_message', 'Processing failed')
                    self.log_result("Quality Assessment", False, error_msg, duration)
                    return False
            else:
                self.log_result("Quality Assessment", False, f"HTTP {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Quality Assessment", False, f"Error: {str(e)}", duration)
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling with invalid inputs"""
        start = time.time()
        try:
            # Test with invalid URL
            payload = {
                "url": "https://invalid-url-that-does-not-exist.com/nonexistent.pdf",
                "metadata": {"test_type": "error_handling"}
            }
            
            response = requests.post(
                f"{self.base_url}/v1/documents/process",
                json=payload,
                timeout=30
            )
            
            duration = time.time() - start
            
            if response.status_code == 200:
                result_data = response.json()
                
                # Should return failed status with error message
                if result_data['status'] == 'failed' and 'error_message' in result_data:
                    error_msg = result_data['error_message']
                    details = f"Properly handled error: {error_msg[:50]}..."
                    self.log_result("Error Handling", True, details, duration)
                    return True
                else:
                    self.log_result("Error Handling", False, "Did not properly handle invalid URL", duration)
                    return False
            else:
                # HTTP error is also acceptable for error handling test
                self.log_result("Error Handling", True, f"HTTP error properly returned: {response.status_code}", duration)
                return True
                
        except Exception as e:
            duration = time.time() - start
            # Exception handling is also acceptable
            self.log_result("Error Handling", True, f"Exception properly caught: {str(e)[:50]}...", duration)
            return True
    
    def test_concurrent_processing(self) -> bool:
        """Test concurrent document processing"""
        start = time.time()
        try:
            import concurrent.futures
            import threading
            
            def process_document(doc_num: int) -> bool:
                try:
                    payload = {
                        "url": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
                        "metadata": {"test_type": "concurrent", "doc_number": doc_num}
                    }
                    
                    response = requests.post(
                        f"{self.base_url}/v1/documents/process",
                        json=payload,
                        timeout=60
                    )
                    
                    return response.status_code == 200 and response.json().get('status') == 'completed'
                except:
                    return False
            
            # Process 3 documents concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(process_document, i) for i in range(3)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            duration = time.time() - start
            
            successful_requests = sum(1 for result in results if result)
            
            if successful_requests >= 2:  # Allow for some failures in concurrent testing
                details = f"Successfully processed {successful_requests}/3 concurrent requests"
                self.log_result("Concurrent Processing", True, details, duration)
                return True
            else:
                self.log_result("Concurrent Processing", False, f"Only {successful_requests}/3 requests succeeded", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Concurrent Processing", False, f"Error: {str(e)}", duration)
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all document processing tests"""
        print("📄 Document Processing MCP Server Test Suite")
        print("=" * 50)
        
        tests = [
            ("Server Health", self.test_server_health),
            ("Root Endpoint", self.test_root_endpoint),
            ("Simple Document Processing", self.test_document_processing_simple),
            ("Metadata Enrichment", self.test_document_processing_with_metadata),
            ("Quality Assessment", self.test_quality_assessment),
            ("Error Handling", self.test_error_handling),
            ("Concurrent Processing", self.test_concurrent_processing)
        ]
        
        passed = 0
        for test_name, test_func in tests:
            if test_func():
                passed += 1
            print()  # Add spacing
        
        total = len(tests)
        
        print("=" * 50)
        print(f"📊 Document Processing Results: {passed}/{total} tests passed")
        
        return {
            'total_tests': total,
            'passed_tests': passed,
            'success_rate': (passed / total) * 100,
            'test_results': self.results
        }


def main():
    """Main test execution"""
    tester = DocumentProcessingTester()
    results = tester.run_all_tests()
    
    # Save results
    with open('document_processing_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved to: document_processing_test_results.json")
    return 0 if results['success_rate'] >= 80 else 1


if __name__ == "__main__":
    exit(main())
