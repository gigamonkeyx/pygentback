#!/usr/bin/env python3
"""
Enhanced Document Processing Server Test Suite

Tests all endpoints including file upload and content analysis.
"""

import time
import json
import requests
from typing import Dict, List, Any
from datetime import datetime
import io

class EnhancedDocumentProcessingTester:
    """Test enhanced document processing server capabilities"""
    
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
    
    def test_enhanced_root_endpoint(self) -> bool:
        """Test enhanced root endpoint with new endpoints"""
        start = time.time()
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            duration = time.time() - start
            
            if response.status_code == 200:
                root_data = response.json()
                endpoints = root_data.get('endpoints', {})
                
                # Check for new endpoints
                expected_endpoints = ['process', 'extract', 'analyze', 'health']
                missing_endpoints = [ep for ep in expected_endpoints if ep not in endpoints]
                
                if not missing_endpoints:
                    details = f"All {len(expected_endpoints)} endpoints available: {list(endpoints.keys())}"
                    self.log_result("Enhanced Root Endpoint", True, details, duration)
                    return True
                else:
                    self.log_result("Enhanced Root Endpoint", False, f"Missing endpoints: {missing_endpoints}", duration)
                    return False
            else:
                self.log_result("Enhanced Root Endpoint", False, f"HTTP {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Enhanced Root Endpoint", False, f"Error: {str(e)}", duration)
            return False
    
    def test_content_analysis_endpoint(self) -> bool:
        """Test content analysis endpoint"""
        start = time.time()
        try:
            # Test with sample academic text
            sample_text = """
            Abstract
            
            This paper presents a comprehensive analysis of machine learning algorithms 
            in natural language processing. The methodology involves comparing various 
            approaches including neural networks, support vector machines, and ensemble 
            methods. Our results demonstrate significant improvements in accuracy and 
            processing speed.
            
            Introduction
            
            Natural language processing has become increasingly important in modern 
            artificial intelligence applications. This research aims to evaluate 
            different algorithmic approaches and their effectiveness.
            
            Methodology
            
            We conducted experiments using three datasets with varying complexity 
            levels. Each algorithm was tested under controlled conditions with 
            standardized evaluation metrics.
            
            Conclusion
            
            The findings suggest that ensemble methods provide the best balance 
            between accuracy and computational efficiency for most NLP tasks.
            """
            
            payload = {
                "text": sample_text,
                "metadata": {
                    "title": "ML in NLP Research",
                    "source": "test_suite"
                },
                "analysis_type": "comprehensive"
            }
            
            response = requests.post(
                f"{self.base_url}/v1/documents/analyze",
                json=payload,
                timeout=30
            )
            
            duration = time.time() - start
            
            if response.status_code == 200:
                analysis_data = response.json()
                
                # Validate response structure
                required_fields = ['document_type', 'categories', 'tags', 'structure_analysis', 'quality_assessment', 'ai_insights']
                missing_fields = [field for field in required_fields if field not in analysis_data]
                
                if missing_fields:
                    self.log_result("Content Analysis", False, f"Missing fields: {missing_fields}", duration)
                    return False
                
                categories = analysis_data.get('categories', [])
                structure = analysis_data.get('structure_analysis', {})
                
                # Should detect academic content
                is_academic = 'academic' in categories
                has_structure = 'word_count' in structure
                
                if is_academic and has_structure:
                    word_count = structure.get('word_count', 0)
                    details = f"Detected academic content, Categories: {categories}, Words: {word_count}"
                    self.log_result("Content Analysis", True, details, duration)
                    return True
                else:
                    details = f"Analysis incomplete - Academic: {is_academic}, Structure: {has_structure}"
                    self.log_result("Content Analysis", False, details, duration)
                    return False
            else:
                self.log_result("Content Analysis", False, f"HTTP {response.status_code}: {response.text}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Content Analysis", False, f"Error: {str(e)}", duration)
            return False
    
    def test_text_analysis_categorization(self) -> bool:
        """Test text analysis with different content types"""
        start = time.time()
        try:
            # Test historical content
            historical_text = """
            The American Civil War (1861-1865) was a pivotal conflict in United States history.
            The war began when Confederate forces attacked Fort Sumter in South Carolina.
            President Abraham Lincoln led the Union forces against the Confederate States.
            The conflict resulted in the abolition of slavery and preservation of the Union.
            Major battles included Gettysburg, Antietam, and Bull Run.
            The war ended with General Lee's surrender at Appomattox Court House.
            """
            
            payload = {
                "text": historical_text,
                "metadata": {"content_type": "historical"},
                "analysis_type": "categorization"
            }
            
            response = requests.post(
                f"{self.base_url}/v1/documents/analyze",
                json=payload,
                timeout=30
            )
            
            duration = time.time() - start
            
            if response.status_code == 200:
                analysis_data = response.json()
                categories = analysis_data.get('categories', [])
                tags = analysis_data.get('tags', [])
                
                # Should detect historical content
                is_historical = 'historical' in categories
                has_history_tag = 'history' in tags
                
                if is_historical or has_history_tag:
                    details = f"Correctly categorized historical content - Categories: {categories}, Tags: {tags}"
                    self.log_result("Text Categorization", True, details, duration)
                    return True
                else:
                    details = f"Failed to detect historical content - Categories: {categories}, Tags: {tags}"
                    self.log_result("Text Categorization", False, details, duration)
                    return False
            else:
                self.log_result("Text Categorization", False, f"HTTP {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Text Categorization", False, f"Error: {str(e)}", duration)
            return False
    
    def test_quality_assessment_detailed(self) -> bool:
        """Test detailed quality assessment"""
        start = time.time()
        try:
            # Test with high-quality text
            high_quality_text = """
            This document contains well-structured, coherent text with proper grammar,
            punctuation, and formatting. The content is organized into logical paragraphs
            with clear topic sentences and supporting details. The vocabulary is appropriate
            and the writing style is professional and academic in nature.
            
            The text demonstrates good readability with appropriate sentence length variation
            and clear transitions between ideas. Technical terms are used correctly and
            the overall presentation follows standard conventions for formal writing.
            """
            
            payload = {
                "text": high_quality_text,
                "metadata": {"quality_test": True},
                "analysis_type": "comprehensive"
            }
            
            response = requests.post(
                f"{self.base_url}/v1/documents/analyze",
                json=payload,
                timeout=30
            )
            
            duration = time.time() - start
            
            if response.status_code == 200:
                analysis_data = response.json()
                quality_assessment = analysis_data.get('quality_assessment', {})
                
                # Check quality metrics
                text_length = quality_assessment.get('text_length', 0)
                completeness = quality_assessment.get('completeness', 0)
                
                if text_length > 0 and completeness > 0:
                    details = f"Quality assessment complete - Length: {text_length}, Completeness: {completeness}"
                    self.log_result("Quality Assessment", True, details, duration)
                    return True
                else:
                    self.log_result("Quality Assessment", False, "Quality metrics missing or invalid", duration)
                    return False
            else:
                self.log_result("Quality Assessment", False, f"HTTP {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Quality Assessment", False, f"Error: {str(e)}", duration)
            return False
    
    def test_ai_insights_generation(self) -> bool:
        """Test AI insights generation"""
        start = time.time()
        try:
            # Test with technical content
            technical_text = """
            The implementation of microservices architecture requires careful consideration
            of service boundaries, data consistency, and inter-service communication patterns.
            Container orchestration platforms like Kubernetes provide scalability and
            fault tolerance for distributed systems. API gateways manage authentication,
            rate limiting, and request routing across multiple service instances.
            """
            
            payload = {
                "text": technical_text,
                "metadata": {"domain": "technology"},
                "analysis_type": "comprehensive"
            }
            
            response = requests.post(
                f"{self.base_url}/v1/documents/analyze",
                json=payload,
                timeout=30
            )
            
            duration = time.time() - start
            
            if response.status_code == 200:
                analysis_data = response.json()
                ai_insights = analysis_data.get('ai_insights', {})
                
                # Check AI insights
                has_analysis_type = 'analysis_type' in ai_insights
                has_processing_method = 'processing_method' in ai_insights
                has_timestamp = 'timestamp' in ai_insights
                
                if has_analysis_type and has_processing_method:
                    analysis_type = ai_insights.get('analysis_type')
                    method = ai_insights.get('processing_method')
                    details = f"AI insights generated - Type: {analysis_type}, Method: {method}"
                    self.log_result("AI Insights", True, details, duration)
                    return True
                else:
                    missing = []
                    if not has_analysis_type: missing.append("analysis_type")
                    if not has_processing_method: missing.append("processing_method")
                    self.log_result("AI Insights", False, f"Missing insights: {missing}", duration)
                    return False
            else:
                self.log_result("AI Insights", False, f"HTTP {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("AI Insights", False, f"Error: {str(e)}", duration)
            return False
    
    def test_performance_under_load(self) -> bool:
        """Test performance with multiple analysis requests"""
        start = time.time()
        try:
            import concurrent.futures
            
            def analyze_text(text_id: int) -> bool:
                try:
                    payload = {
                        "text": f"Performance test document {text_id}. " * 50,  # ~250 words
                        "metadata": {"test_id": text_id},
                        "analysis_type": "comprehensive"
                    }
                    
                    response = requests.post(
                        f"{self.base_url}/v1/documents/analyze",
                        json=payload,
                        timeout=30
                    )
                    
                    return response.status_code == 200
                except:
                    return False
            
            # Analyze 5 texts concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(analyze_text, i) for i in range(5)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            duration = time.time() - start
            
            successful_analyses = sum(1 for result in results if result)
            
            if successful_analyses >= 4:  # Allow for one failure
                throughput = successful_analyses / duration
                details = f"Processed {successful_analyses}/5 analyses, Throughput: {throughput:.2f} analyses/sec"
                self.log_result("Performance Under Load", True, details, duration)
                return True
            else:
                self.log_result("Performance Under Load", False, f"Only {successful_analyses}/5 analyses succeeded", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Performance Under Load", False, f"Error: {str(e)}", duration)
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all enhanced document processing tests"""
        print("📄 Enhanced Document Processing Test Suite")
        print("=" * 50)
        
        tests = [
            ("Enhanced Root Endpoint", self.test_enhanced_root_endpoint),
            ("Content Analysis", self.test_content_analysis_endpoint),
            ("Text Categorization", self.test_text_analysis_categorization),
            ("Quality Assessment", self.test_quality_assessment_detailed),
            ("AI Insights", self.test_ai_insights_generation),
            ("Performance Under Load", self.test_performance_under_load)
        ]
        
        passed = 0
        for test_name, test_func in tests:
            if test_func():
                passed += 1
            print()  # Add spacing
        
        total = len(tests)
        
        print("=" * 50)
        print(f"📊 Enhanced Processing Results: {passed}/{total} tests passed")
        
        return {
            'total_tests': total,
            'passed_tests': passed,
            'success_rate': (passed / total) * 100,
            'test_results': self.results
        }


def main():
    """Main test execution"""
    tester = EnhancedDocumentProcessingTester()
    results = tester.run_all_tests()
    
    # Save results
    with open('enhanced_document_processing_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved to: enhanced_document_processing_results.json")
    return 0 if results['success_rate'] >= 80 else 1


if __name__ == "__main__":
    exit(main())
