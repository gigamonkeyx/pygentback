"""
Test Interpretation Module

Modular components for interpreting test results and providing insights.
"""

import logging
import time
import statistics
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .core import NLPProcessor, ConfidenceCalculator, ProcessingResult
from .models import TestResult, TestStatus, InterpretationResult, TestMetrics

logger = logging.getLogger(__name__)


class TestPatterns:
    """Test result analysis patterns"""
    
    ERROR_PATTERNS = {
        'timeout': [r'timeout', r'time.*out', r'exceeded.*time'],
        'memory': [r'memory.*error', r'out.*of.*memory', r'memory.*leak'],
        'network': [r'network.*error', r'connection.*failed', r'dns.*error'],
        'permission': [r'permission.*denied', r'access.*denied', r'unauthorized'],
        'syntax': [r'syntax.*error', r'invalid.*syntax', r'parse.*error'],
        'runtime': [r'runtime.*error', r'null.*pointer', r'index.*out.*of.*bounds']
    }
    
    PERFORMANCE_PATTERNS = {
        'slow': [r'slow', r'performance.*issue', r'taking.*too.*long'],
        'fast': [r'fast', r'quick', r'efficient', r'optimized'],
        'memory_heavy': [r'high.*memory', r'memory.*intensive', r'large.*footprint'],
        'cpu_heavy': [r'high.*cpu', r'cpu.*intensive', r'processing.*heavy']
    }
    
    SUCCESS_INDICATORS = [
        r'passed', r'success', r'ok', r'completed', r'finished', r'done'
    ]
    
    FAILURE_INDICATORS = [
        r'failed', r'error', r'exception', r'crash', r'abort', r'timeout'
    ]


class ResultAnalyzer:
    """
    Analyzes test results and provides detailed insights.
    """
    
    def __init__(self):
        self.performance_thresholds = {
            'execution_time_ms': {'good': 1000, 'warning': 5000, 'critical': 10000},
            'memory_usage_mb': {'good': 100, 'warning': 500, 'critical': 1000},
            'cpu_usage_percent': {'good': 50, 'warning': 80, 'critical': 95},
            'error_rate': {'good': 0.01, 'warning': 0.05, 'critical': 0.1}
        }
    
    def analyze_test_suite(self, test_results: List[TestResult]) -> Dict[str, Any]:
        """Analyze a complete test suite"""
        if not test_results:
            return {'error': 'No test results provided'}
        
        analysis = {
            'total_tests': len(test_results),
            'status_distribution': self._analyze_status_distribution(test_results),
            'performance_analysis': self._analyze_performance(test_results),
            'error_analysis': self._analyze_errors(test_results),
            'trends': self._analyze_trends(test_results),
            'recommendations': self._generate_recommendations(test_results)
        }
        
        return analysis
    
    def _analyze_status_distribution(self, test_results: List[TestResult]) -> Dict[str, Any]:
        """Analyze distribution of test statuses"""
        status_counts = {}
        for status in TestStatus:
            status_counts[status.value] = sum(1 for test in test_results if test.status == status)
        
        total = len(test_results)
        status_percentages = {
            status: (count / total * 100) if total > 0 else 0
            for status, count in status_counts.items()
        }
        
        success_rate = status_percentages.get('passed', 0)
        
        return {
            'counts': status_counts,
            'percentages': status_percentages,
            'success_rate': success_rate,
            'failure_rate': 100 - success_rate
        }
    
    def _analyze_performance(self, test_results: List[TestResult]) -> Dict[str, Any]:
        """Analyze performance metrics"""
        execution_times = [test.metrics.execution_time_ms for test in test_results if test.metrics.execution_time_ms > 0]
        memory_usage = [test.metrics.memory_usage_mb for test in test_results if test.metrics.memory_usage_mb > 0]
        cpu_usage = [test.metrics.cpu_usage_percent for test in test_results if test.metrics.cpu_usage_percent > 0]
        
        performance_analysis = {}
        
        if execution_times:
            performance_analysis['execution_time'] = {
                'avg': statistics.mean(execution_times),
                'median': statistics.median(execution_times),
                'min': min(execution_times),
                'max': max(execution_times),
                'std': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                'assessment': self._assess_metric(statistics.mean(execution_times), 'execution_time_ms')
            }
        
        if memory_usage:
            performance_analysis['memory_usage'] = {
                'avg': statistics.mean(memory_usage),
                'median': statistics.median(memory_usage),
                'min': min(memory_usage),
                'max': max(memory_usage),
                'assessment': self._assess_metric(statistics.mean(memory_usage), 'memory_usage_mb')
            }
        
        if cpu_usage:
            performance_analysis['cpu_usage'] = {
                'avg': statistics.mean(cpu_usage),
                'median': statistics.median(cpu_usage),
                'min': min(cpu_usage),
                'max': max(cpu_usage),
                'assessment': self._assess_metric(statistics.mean(cpu_usage), 'cpu_usage_percent')
            }
        
        return performance_analysis
    
    def _assess_metric(self, value: float, metric_type: str) -> str:
        """Assess a performance metric"""
        thresholds = self.performance_thresholds.get(metric_type, {})
        
        if value <= thresholds.get('good', 0):
            return 'good'
        elif value <= thresholds.get('warning', float('inf')):
            return 'warning'
        elif value <= thresholds.get('critical', float('inf')):
            return 'critical'
        else:
            return 'severe'
    
    def _analyze_errors(self, test_results: List[TestResult]) -> Dict[str, Any]:
        """Analyze error patterns"""
        failed_tests = [test for test in test_results if test.status in [TestStatus.FAILED, TestStatus.ERROR]]
        
        if not failed_tests:
            return {'error_count': 0, 'patterns': {}}
        
        error_patterns = {}
        for pattern_name, patterns in TestPatterns.ERROR_PATTERNS.items():
            count = 0
            for test in failed_tests:
                error_text = (test.error_details or test.message or '').lower()
                for pattern in patterns:
                    if pattern in error_text:
                        count += 1
                        break
            
            if count > 0:
                error_patterns[pattern_name] = count
        
        return {
            'error_count': len(failed_tests),
            'error_rate': len(failed_tests) / len(test_results) * 100,
            'patterns': error_patterns,
            'most_common_error': max(error_patterns.keys(), key=error_patterns.get) if error_patterns else None
        }
    
    def _analyze_trends(self, test_results: List[TestResult]) -> Dict[str, Any]:
        """Analyze trends in test results"""
        # Sort by timestamp
        sorted_tests = sorted(test_results, key=lambda t: t.timestamp)
        
        if len(sorted_tests) < 2:
            return {'insufficient_data': True}
        
        # Analyze success rate trend
        recent_tests = sorted_tests[-10:]  # Last 10 tests
        older_tests = sorted_tests[:-10] if len(sorted_tests) > 10 else []
        
        trends = {}
        
        if older_tests:
            recent_success_rate = sum(1 for test in recent_tests if test.status == TestStatus.PASSED) / len(recent_tests)
            older_success_rate = sum(1 for test in older_tests if test.status == TestStatus.PASSED) / len(older_tests)
            
            trends['success_rate_trend'] = {
                'recent': recent_success_rate * 100,
                'previous': older_success_rate * 100,
                'direction': 'improving' if recent_success_rate > older_success_rate else 'declining'
            }
        
        # Analyze performance trends
        if len(sorted_tests) >= 5:
            recent_times = [test.metrics.execution_time_ms for test in sorted_tests[-5:] if test.metrics.execution_time_ms > 0]
            older_times = [test.metrics.execution_time_ms for test in sorted_tests[:-5] if test.metrics.execution_time_ms > 0]
            
            if recent_times and older_times:
                recent_avg = statistics.mean(recent_times)
                older_avg = statistics.mean(older_times)
                
                trends['performance_trend'] = {
                    'recent_avg_ms': recent_avg,
                    'previous_avg_ms': older_avg,
                    'direction': 'improving' if recent_avg < older_avg else 'declining'
                }
        
        return trends
    
    def _generate_recommendations(self, test_results: List[TestResult]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Analyze failure rate
        failed_count = sum(1 for test in test_results if test.status in [TestStatus.FAILED, TestStatus.ERROR])
        failure_rate = failed_count / len(test_results) * 100
        
        if failure_rate > 20:
            recommendations.append("High failure rate detected - review test implementation and environment")
        elif failure_rate > 10:
            recommendations.append("Moderate failure rate - investigate common failure patterns")
        
        # Analyze performance
        slow_tests = [test for test in test_results if test.metrics.execution_time_ms > 5000]
        if len(slow_tests) > len(test_results) * 0.2:
            recommendations.append("Many slow tests detected - consider performance optimization")
        
        # Analyze memory usage
        memory_heavy_tests = [test for test in test_results if test.metrics.memory_usage_mb > 500]
        if memory_heavy_tests:
            recommendations.append("High memory usage detected - review memory management")
        
        # Analyze error patterns
        timeout_errors = sum(1 for test in test_results 
                           if test.status == TestStatus.TIMEOUT or 
                           'timeout' in (test.error_details or test.message or '').lower())
        
        if timeout_errors > 0:
            recommendations.append("Timeout errors detected - review test timeouts and performance")
        
        return recommendations


class TestInterpreter(NLPProcessor):
    """
    Natural language interpreter for test results.
    """
    
    def __init__(self):
        super().__init__("TestInterpreter")
        self.analyzer = ResultAnalyzer()
        self._setup_patterns()
    
    def _setup_patterns(self):
        """Setup pattern matcher for test interpretation"""
        # Add error patterns
        for error_type, patterns in TestPatterns.ERROR_PATTERNS.items():
            for i, pattern in enumerate(patterns):
                self.pattern_matcher.add_pattern(f"error_{error_type}_{i}", pattern)
        
        # Add performance patterns
        for perf_type, patterns in TestPatterns.PERFORMANCE_PATTERNS.items():
            for i, pattern in enumerate(patterns):
                self.pattern_matcher.add_pattern(f"perf_{perf_type}_{i}", pattern)
        
        # Add success/failure indicators
        for i, pattern in enumerate(TestPatterns.SUCCESS_INDICATORS):
            self.pattern_matcher.add_pattern(f"success_{i}", pattern)
        
        for i, pattern in enumerate(TestPatterns.FAILURE_INDICATORS):
            self.pattern_matcher.add_pattern(f"failure_{i}", pattern)
    
    async def process(self, text: str, context: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Process test results and return interpretation"""
        start_time = time.time()
        
        try:
            # Extract test results from context or text
            test_results = self._extract_test_results(text, context)
            
            # Interpret results
            interpretation = await self.interpret_results(test_results)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update statistics
            self.update_stats(True, interpretation.interpretation_confidence, processing_time)
            
            return ProcessingResult(
                success=True,
                confidence=interpretation.interpretation_confidence,
                processing_time_ms=processing_time,
                metadata={'interpretation': interpretation.__dict__}
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.update_stats(False, 0.0, processing_time)
            
            logger.error(f"Test interpretation failed: {e}")
            
            return ProcessingResult(
                success=False,
                confidence=0.0,
                processing_time_ms=processing_time,
                metadata={'error': str(e)}
            )
    
    def _extract_test_results(self, text: str, context: Optional[Dict[str, Any]]) -> List[TestResult]:
        """Extract test results from text or context"""
        test_results = []
        
        # Try to get from context first
        if context and 'test_results' in context:
            return context['test_results']
        
        # Parse from text (simplified)
        lines = text.split('\n')
        for line in lines:
            if any(indicator in line.lower() for indicator in ['test', 'passed', 'failed', 'error']):
                # Simple parsing - in practice would be more sophisticated
                status = TestStatus.UNKNOWN
                if any(pattern in line.lower() for pattern in TestPatterns.SUCCESS_INDICATORS):
                    status = TestStatus.PASSED
                elif any(pattern in line.lower() for pattern in TestPatterns.FAILURE_INDICATORS):
                    status = TestStatus.FAILED
                
                test_result = TestResult(
                    test_name=line.strip()[:50],  # Use line as test name
                    status=status,
                    message=line.strip()
                )
                test_results.append(test_result)
        
        return test_results
    
    async def interpret_results(self, test_results: List[TestResult]) -> InterpretationResult:
        """Interpret test results and provide insights"""
        if not test_results:
            return InterpretationResult(
                summary="No test results to interpret",
                success_rate=0.0,
                performance_assessment="Unknown",
                interpretation_confidence=0.0
            )
        
        # Analyze results
        analysis = self.analyzer.analyze_test_suite(test_results)
        
        # Generate summary
        summary = self._generate_summary(analysis, test_results)
        
        # Calculate success rate
        success_rate = analysis['status_distribution']['success_rate']
        
        # Assess performance
        performance_assessment = self._assess_overall_performance(analysis)
        
        # Extract details
        passed_tests = [test.test_name for test in test_results if test.status == TestStatus.PASSED]
        failed_tests = [test.test_name for test in test_results if test.status in [TestStatus.FAILED, TestStatus.ERROR]]
        
        # Identify error patterns
        error_patterns = list(analysis['error_analysis'].get('patterns', {}).keys())
        
        # Identify performance issues
        performance_issues = self._identify_performance_issues(analysis)
        
        # Get recommendations
        recommendations = analysis.get('recommendations', [])
        
        # Calculate confidence
        confidence = self._calculate_interpretation_confidence(test_results, analysis)
        
        return InterpretationResult(
            summary=summary,
            success_rate=success_rate,
            performance_assessment=performance_assessment,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            error_patterns=error_patterns,
            performance_issues=performance_issues,
            recommendations=recommendations,
            interpretation_confidence=confidence
        )
    
    def _generate_summary(self, analysis: Dict[str, Any], test_results: List[TestResult]) -> str:
        """Generate natural language summary"""
        total_tests = len(test_results)
        success_rate = analysis['status_distribution']['success_rate']
        
        if success_rate >= 95:
            summary = f"Excellent test results: {total_tests} tests with {success_rate:.1f}% success rate."
        elif success_rate >= 80:
            summary = f"Good test results: {total_tests} tests with {success_rate:.1f}% success rate."
        elif success_rate >= 60:
            summary = f"Moderate test results: {total_tests} tests with {success_rate:.1f}% success rate."
        else:
            summary = f"Poor test results: {total_tests} tests with only {success_rate:.1f}% success rate."
        
        # Add performance note
        perf_analysis = analysis.get('performance_analysis', {})
        if 'execution_time' in perf_analysis:
            avg_time = perf_analysis['execution_time']['avg']
            if avg_time > 5000:
                summary += " Performance is concerning with slow execution times."
            elif avg_time > 2000:
                summary += " Performance could be improved."
            else:
                summary += " Performance is acceptable."
        
        return summary
    
    def _assess_overall_performance(self, analysis: Dict[str, Any]) -> str:
        """Assess overall performance"""
        perf_analysis = analysis.get('performance_analysis', {})
        
        assessments = []
        for metric, data in perf_analysis.items():
            if 'assessment' in data:
                assessments.append(data['assessment'])
        
        if not assessments:
            return "Unknown"
        
        # Determine overall assessment
        if all(a == 'good' for a in assessments):
            return "Excellent"
        elif any(a in ['critical', 'severe'] for a in assessments):
            return "Poor"
        elif any(a == 'warning' for a in assessments):
            return "Moderate"
        else:
            return "Good"
    
    def _identify_performance_issues(self, analysis: Dict[str, Any]) -> List[str]:
        """Identify specific performance issues"""
        issues = []
        perf_analysis = analysis.get('performance_analysis', {})
        
        for metric, data in perf_analysis.items():
            assessment = data.get('assessment', 'unknown')
            if assessment in ['warning', 'critical', 'severe']:
                if metric == 'execution_time':
                    issues.append(f"Slow execution times (avg: {data['avg']:.1f}ms)")
                elif metric == 'memory_usage':
                    issues.append(f"High memory usage (avg: {data['avg']:.1f}MB)")
                elif metric == 'cpu_usage':
                    issues.append(f"High CPU usage (avg: {data['avg']:.1f}%)")
        
        return issues
    
    def _calculate_interpretation_confidence(self, test_results: List[TestResult], 
                                           analysis: Dict[str, Any]) -> float:
        """Calculate confidence in interpretation"""
        factors = []
        
        # Data quality factor
        data_quality = min(1.0, len(test_results) / 10.0)  # More tests = higher confidence
        factors.append(data_quality)
        
        # Result consistency factor
        status_dist = analysis['status_distribution']['percentages']
        # High confidence if results are consistent (mostly pass or mostly fail)
        max_percentage = max(status_dist.values()) if status_dist else 0
        consistency = max_percentage / 100.0
        factors.append(consistency)
        
        # Analysis completeness factor
        has_performance = bool(analysis.get('performance_analysis'))
        has_errors = bool(analysis.get('error_analysis', {}).get('patterns'))
        completeness = (has_performance + has_errors) / 2.0
        factors.append(completeness)
        
        return ConfidenceCalculator.combined_confidence(factors, [0.4, 0.3, 0.3])
