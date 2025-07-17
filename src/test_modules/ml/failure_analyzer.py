"""
Failure Pattern Analyzer

ML-based analysis of test failures to identify patterns, root causes, and provide
intelligent recommendations for fixing recurring issues.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import statistics
from enum import Enum

logger = logging.getLogger(__name__)


class FailureCategory(Enum):
    """Categories of test failures"""
    TIMEOUT = "timeout"
    ASSERTION = "assertion"
    EXCEPTION = "exception"
    DEPENDENCY = "dependency"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    RESOURCE = "resource"
    LOGIC = "logic"
    UNKNOWN = "unknown"


@dataclass
class FailurePattern:
    """Identified failure pattern"""
    pattern_id: str
    category: FailureCategory
    description: str
    frequency: int
    confidence_score: float
    affected_tests: List[str]
    error_signatures: List[str]
    root_cause_hypothesis: str
    recommended_actions: List[str]
    first_seen: datetime
    last_seen: datetime


@dataclass
class FailureAnalysis:
    """Analysis result for a specific failure"""
    test_id: str
    error_message: str
    stack_trace: Optional[str]
    category: FailureCategory
    confidence: float
    similar_failures: List[str]
    root_cause: str
    recommendations: List[str]
    pattern_matches: List[str]


@dataclass
class TrendAnalysis:
    """Trend analysis of failures over time"""
    time_period: str
    total_failures: int
    failure_rate_trend: str  # increasing, decreasing, stable
    most_common_categories: List[Tuple[FailureCategory, int]]
    emerging_patterns: List[str]
    resolved_patterns: List[str]
    stability_score: float


class FailurePatternAnalyzer:
    """
    ML-based Failure Pattern Analysis System.
    
    Analyzes test failures to identify recurring patterns, categorize issues,
    and provide intelligent recommendations for resolution.
    """
    
    def __init__(self):
        # Pattern storage
        self.known_patterns: Dict[str, FailurePattern] = {}
        self.failure_history: List[Dict[str, Any]] = []
        
        # Analysis configuration
        self.min_pattern_frequency = 3
        self.confidence_threshold = 0.7
        self.pattern_similarity_threshold = 0.8
        
        # Error signature patterns
        self.error_patterns = self._initialize_error_patterns()
        
        # ML model placeholders (would be actual models in production)
        self.classification_model = None
        self.clustering_model = None
    
    def _initialize_error_patterns(self) -> Dict[FailureCategory, List[str]]:
        """Initialize common error patterns for categorization"""
        return {
            FailureCategory.TIMEOUT: [
                r"timeout",
                r"timed out",
                r"connection timeout",
                r"read timeout",
                r"operation timeout",
                r"deadline exceeded"
            ],
            FailureCategory.ASSERTION: [
                r"assertion.*failed",
                r"expected.*but.*was",
                r"assert.*error",
                r"comparison failed",
                r"value mismatch"
            ],
            FailureCategory.EXCEPTION: [
                r"null pointer",
                r"index out of bounds",
                r"key error",
                r"attribute error",
                r"type error",
                r"value error"
            ],
            FailureCategory.DEPENDENCY: [
                r"module not found",
                r"import error",
                r"dependency.*missing",
                r"package.*not.*found",
                r"library.*error"
            ],
            FailureCategory.CONFIGURATION: [
                r"configuration.*error",
                r"config.*missing",
                r"setting.*invalid",
                r"property.*not.*set",
                r"environment.*variable"
            ],
            FailureCategory.NETWORK: [
                r"connection.*refused",
                r"network.*error",
                r"host.*unreachable",
                r"dns.*resolution",
                r"socket.*error"
            ],
            FailureCategory.RESOURCE: [
                r"out of memory",
                r"disk.*full",
                r"resource.*exhausted",
                r"file.*not.*found",
                r"permission.*denied"
            ]
        }
    
    def analyze_failure(self, test_id: str, error_message: str, 
                       stack_trace: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> FailureAnalysis:
        """
        Analyze a single test failure.
        
        Args:
            test_id: Identifier for the failed test
            error_message: Error message from the failure
            stack_trace: Optional stack trace
            metadata: Additional metadata about the test
            
        Returns:
            FailureAnalysis with categorization and recommendations
        """
        # Store failure in history
        failure_record = {
            "test_id": test_id,
            "error_message": error_message,
            "stack_trace": stack_trace,
            "timestamp": datetime.utcnow(),
            "metadata": metadata or {}
        }
        self.failure_history.append(failure_record)
        
        # Categorize the failure
        category, confidence = self._categorize_failure(error_message, stack_trace)
        
        # Find similar failures
        similar_failures = self._find_similar_failures(error_message, stack_trace)
        
        # Determine root cause
        root_cause = self._determine_root_cause(error_message, stack_trace, category)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(category, error_message, similar_failures)
        
        # Find matching patterns
        pattern_matches = self._find_pattern_matches(error_message, stack_trace)
        
        analysis = FailureAnalysis(
            test_id=test_id,
            error_message=error_message,
            stack_trace=stack_trace,
            category=category,
            confidence=confidence,
            similar_failures=similar_failures,
            root_cause=root_cause,
            recommendations=recommendations,
            pattern_matches=pattern_matches
        )
        
        # Update patterns
        self._update_patterns(analysis)
        
        return analysis
    
    def _categorize_failure(self, error_message: str, stack_trace: Optional[str]) -> Tuple[FailureCategory, float]:
        """Categorize failure based on error message and stack trace"""
        text_to_analyze = error_message.lower()
        if stack_trace:
            text_to_analyze += " " + stack_trace.lower()
        
        category_scores = {}
        
        for category, patterns in self.error_patterns.items():
            score = 0.0
            matches = 0
            
            for pattern in patterns:
                if re.search(pattern, text_to_analyze, re.IGNORECASE):
                    matches += 1
                    score += 1.0
            
            if matches > 0:
                # Normalize score by number of patterns
                category_scores[category] = score / len(patterns)
        
        if not category_scores:
            return FailureCategory.UNKNOWN, 0.5
        
        # Find category with highest score
        best_category = max(category_scores.items(), key=lambda x: x[1])
        return best_category[0], min(best_category[1], 1.0)
    
    def _find_similar_failures(self, error_message: str, stack_trace: Optional[str]) -> List[str]:
        """Find similar failures in history"""
        similar_failures = []
        
        for failure in self.failure_history[-100:]:  # Check last 100 failures
            similarity = self._calculate_similarity(
                error_message, failure["error_message"],
                stack_trace, failure.get("stack_trace")
            )
            
            if similarity > self.pattern_similarity_threshold:
                similar_failures.append(failure["test_id"])
        
        return similar_failures
    
    def _calculate_similarity(self, msg1: str, msg2: str, 
                            trace1: Optional[str], trace2: Optional[str]) -> float:
        """Calculate similarity between two failures"""
        # Simple similarity based on common words
        words1 = set(re.findall(r'\w+', msg1.lower()))
        words2 = set(re.findall(r'\w+', msg2.lower()))
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0.0
        
        # Boost similarity if stack traces are similar
        if trace1 and trace2:
            trace_words1 = set(re.findall(r'\w+', trace1.lower()))
            trace_words2 = set(re.findall(r'\w+', trace2.lower()))
            
            if trace_words1 and trace_words2:
                trace_intersection = len(trace_words1.intersection(trace_words2))
                trace_union = len(trace_words1.union(trace_words2))
                trace_similarity = trace_intersection / trace_union if trace_union > 0 else 0.0
                
                # Weight message similarity more than stack trace
                similarity = 0.7 * similarity + 0.3 * trace_similarity
        
        return similarity
    
    def _determine_root_cause(self, error_message: str, stack_trace: Optional[str], 
                            category: FailureCategory) -> str:
        """Determine likely root cause of failure"""
        root_causes = {
            FailureCategory.TIMEOUT: "Operation exceeded time limit - check for slow operations or increase timeout",
            FailureCategory.ASSERTION: "Test assertion failed - verify expected vs actual values",
            FailureCategory.EXCEPTION: "Runtime exception occurred - check for null values or invalid operations",
            FailureCategory.DEPENDENCY: "Missing or incompatible dependency - verify installation and versions",
            FailureCategory.CONFIGURATION: "Configuration issue - check settings and environment variables",
            FailureCategory.NETWORK: "Network connectivity issue - verify network access and endpoints",
            FailureCategory.RESOURCE: "Resource limitation - check memory, disk space, or file permissions",
            FailureCategory.LOGIC: "Logic error in test or code - review test implementation",
            FailureCategory.UNKNOWN: "Unable to determine root cause - manual investigation required"
        }
        
        base_cause = root_causes.get(category, root_causes[FailureCategory.UNKNOWN])
        
        # Add specific details from error message
        if "permission denied" in error_message.lower():
            base_cause += " - Check file/directory permissions"
        elif "connection refused" in error_message.lower():
            base_cause += " - Verify service is running and accessible"
        elif "out of memory" in error_message.lower():
            base_cause += " - Increase memory allocation or optimize memory usage"
        
        return base_cause
    
    def _generate_recommendations(self, category: FailureCategory, error_message: str, 
                                similar_failures: List[str]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Category-specific recommendations
        category_recommendations = {
            FailureCategory.TIMEOUT: [
                "Increase timeout values in test configuration",
                "Optimize slow operations or database queries",
                "Check for deadlocks or blocking operations",
                "Consider using asynchronous operations"
            ],
            FailureCategory.ASSERTION: [
                "Review test data and expected values",
                "Add more detailed assertion messages",
                "Check for timing issues in dynamic data",
                "Verify test setup and preconditions"
            ],
            FailureCategory.EXCEPTION: [
                "Add null checks and input validation",
                "Review error handling in the code",
                "Check for edge cases and boundary conditions",
                "Add defensive programming practices"
            ],
            FailureCategory.DEPENDENCY: [
                "Update dependency versions",
                "Check package installation and imports",
                "Verify virtual environment setup",
                "Review dependency compatibility matrix"
            ],
            FailureCategory.CONFIGURATION: [
                "Validate configuration files and settings",
                "Check environment variable setup",
                "Review default values and fallbacks",
                "Ensure configuration is environment-appropriate"
            ],
            FailureCategory.NETWORK: [
                "Check network connectivity and firewall rules",
                "Verify service endpoints and ports",
                "Add retry logic for transient failures",
                "Consider using mock services for testing"
            ],
            FailureCategory.RESOURCE: [
                "Monitor and increase resource limits",
                "Optimize resource usage and cleanup",
                "Check file system permissions",
                "Review resource allocation policies"
            ]
        }
        
        recommendations.extend(category_recommendations.get(category, []))
        
        # Add recommendations based on similar failures
        if len(similar_failures) > 2:
            recommendations.append("This is a recurring issue - consider systematic investigation")
            recommendations.append("Review recent changes that might have introduced this pattern")
        
        # Add specific recommendations based on error content
        if "database" in error_message.lower():
            recommendations.append("Check database connection and query syntax")
        if "api" in error_message.lower():
            recommendations.append("Verify API endpoint availability and response format")
        if "file" in error_message.lower():
            recommendations.append("Check file existence and access permissions")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _find_pattern_matches(self, error_message: str, stack_trace: Optional[str]) -> List[str]:
        """Find matching known patterns"""
        matches = []
        
        for pattern_id, pattern in self.known_patterns.items():
            for signature in pattern.error_signatures:
                if re.search(signature, error_message, re.IGNORECASE):
                    matches.append(pattern_id)
                    break
        
        return matches
    
    def _update_patterns(self, analysis: FailureAnalysis):
        """Update known patterns based on new analysis"""
        # Create error signature from the failure
        error_signature = self._create_error_signature(analysis.error_message)
        
        # Check if this matches an existing pattern
        matching_pattern = None
        for pattern in self.known_patterns.values():
            if (pattern.category == analysis.category and 
                any(re.search(sig, analysis.error_message, re.IGNORECASE) 
                    for sig in pattern.error_signatures)):
                matching_pattern = pattern
                break
        
        if matching_pattern:
            # Update existing pattern
            matching_pattern.frequency += 1
            matching_pattern.last_seen = datetime.utcnow()
            if analysis.test_id not in matching_pattern.affected_tests:
                matching_pattern.affected_tests.append(analysis.test_id)
        else:
            # Create new pattern if we have enough similar failures
            if len(analysis.similar_failures) >= self.min_pattern_frequency:
                pattern_id = f"pattern_{len(self.known_patterns) + 1}"
                new_pattern = FailurePattern(
                    pattern_id=pattern_id,
                    category=analysis.category,
                    description=f"Recurring {analysis.category.value} failure",
                    frequency=len(analysis.similar_failures) + 1,
                    confidence_score=analysis.confidence,
                    affected_tests=[analysis.test_id] + analysis.similar_failures,
                    error_signatures=[error_signature],
                    root_cause_hypothesis=analysis.root_cause,
                    recommended_actions=analysis.recommendations,
                    first_seen=datetime.utcnow(),
                    last_seen=datetime.utcnow()
                )
                self.known_patterns[pattern_id] = new_pattern
    
    def _create_error_signature(self, error_message: str) -> str:
        """Create a regex signature from an error message"""
        # Normalize the error message
        signature = error_message.lower()
        
        # Replace specific values with wildcards
        signature = re.sub(r'\d+', r'\\d+', signature)  # Numbers
        signature = re.sub(r'["\'].*?["\']', r'["\'].*?["\']', signature)  # Quoted strings
        signature = re.sub(r'\b\w+\.\w+\.\w+\b', r'\\w+\\.\\w+\\.\\w+', signature)  # Package names
        
        # Escape special regex characters
        signature = re.escape(signature)
        
        # Restore wildcards
        signature = signature.replace(r'\\d\+', r'\\d+')
        signature = signature.replace(r'\[\"\']\.\*\?\[\"\']\', r'["\'].*?["\']')
        signature = signature.replace(r'\\w\+\\\\\.\\\\w\+\\\\\.\\\\w\+', r'\\w+\\.\\w+\\.\\w+')
        
        return signature
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of identified patterns"""
        if not self.known_patterns:
            return {"total_patterns": 0, "patterns": []}
        
        patterns_summary = []
        for pattern in self.known_patterns.values():
            patterns_summary.append({
                "pattern_id": pattern.pattern_id,
                "category": pattern.category.value,
                "frequency": pattern.frequency,
                "confidence": pattern.confidence_score,
                "affected_tests": len(pattern.affected_tests),
                "description": pattern.description
            })
        
        # Sort by frequency
        patterns_summary.sort(key=lambda x: x["frequency"], reverse=True)
        
        return {
            "total_patterns": len(self.known_patterns),
            "patterns": patterns_summary,
            "most_common_category": self._get_most_common_category(),
            "total_failures_analyzed": len(self.failure_history)
        }
    
    def _get_most_common_category(self) -> str:
        """Get the most common failure category"""
        if not self.known_patterns:
            return "none"
        
        category_counts = Counter(pattern.category for pattern in self.known_patterns.values())
        most_common = category_counts.most_common(1)
        return most_common[0][0].value if most_common else "none"
    
    def analyze_trends(self, days: int = 7) -> TrendAnalysis:
        """Analyze failure trends over time"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_failures = [f for f in self.failure_history if f["timestamp"] > cutoff_date]
        
        if not recent_failures:
            return TrendAnalysis(
                time_period=f"last_{days}_days",
                total_failures=0,
                failure_rate_trend="stable",
                most_common_categories=[],
                emerging_patterns=[],
                resolved_patterns=[],
                stability_score=1.0
            )
        
        # Analyze categories
        categories = []
        for failure in recent_failures:
            # Re-categorize each failure
            category, _ = self._categorize_failure(
                failure["error_message"], 
                failure.get("stack_trace")
            )
            categories.append(category)
        
        category_counts = Counter(categories)
        most_common_categories = [(cat, count) for cat, count in category_counts.most_common(3)]
        
        # Calculate stability score (inverse of failure rate)
        total_tests_estimated = len(recent_failures) * 2  # Rough estimate
        failure_rate = len(recent_failures) / max(total_tests_estimated, 1)
        stability_score = max(0.0, 1.0 - failure_rate)
        
        return TrendAnalysis(
            time_period=f"last_{days}_days",
            total_failures=len(recent_failures),
            failure_rate_trend="stable",  # Would need historical data to determine trend
            most_common_categories=most_common_categories,
            emerging_patterns=[],  # Would need pattern comparison over time
            resolved_patterns=[],  # Would need pattern resolution tracking
            stability_score=stability_score
        )
