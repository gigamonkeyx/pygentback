"""
Recipe Scoring System

This module provides comprehensive scoring and metrics for Agent + MCP recipe
test results, including performance, reliability, and quality assessments.
"""

import logging
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from ..core.framework import RecipeTestResult
from ..recipes.schema import RecipeDefinition


logger = logging.getLogger(__name__)


class ScoringMetrics(Enum):
    """Available scoring metrics"""
    SUCCESS_RATE = "success_rate"
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    EFFICIENCY = "efficiency"
    QUALITY = "quality"
    CONSISTENCY = "consistency"
    ROBUSTNESS = "robustness"
    COMPOSITE = "composite"


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of a score"""
    total_score: float
    component_scores: Dict[str, float] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    penalties: Dict[str, float] = field(default_factory=dict)
    bonuses: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""


@dataclass
class ScoringConfig:
    """Configuration for scoring system"""
    # Component weights (must sum to 1.0)
    success_weight: float = 0.30
    performance_weight: float = 0.25
    reliability_weight: float = 0.20
    efficiency_weight: float = 0.15
    quality_weight: float = 0.10
    
    # Performance thresholds
    excellent_time_ms: int = 1000
    good_time_ms: int = 5000
    acceptable_time_ms: int = 15000
    
    # Memory thresholds
    excellent_memory_mb: float = 128.0
    good_memory_mb: float = 512.0
    acceptable_memory_mb: float = 1024.0
    
    # Reliability thresholds
    excellent_success_rate: float = 0.95
    good_success_rate: float = 0.85
    acceptable_success_rate: float = 0.70
    
    # Quality thresholds
    excellent_scenario_pass_rate: float = 0.90
    good_scenario_pass_rate: float = 0.75
    acceptable_scenario_pass_rate: float = 0.60


class RecipeScorer:
    """
    Comprehensive scoring system for Agent + MCP recipe test results.
    
    Provides multiple scoring metrics including success rate, performance,
    reliability, efficiency, and composite scores with detailed breakdowns.
    """
    
    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or ScoringConfig()
        
        # Validate weights
        total_weight = (
            self.config.success_weight + 
            self.config.performance_weight + 
            self.config.reliability_weight + 
            self.config.efficiency_weight + 
            self.config.quality_weight
        )
        
        if abs(total_weight - 1.0) > 0.001:
            logger.warning(f"Scoring weights don't sum to 1.0 (sum: {total_weight})")
            # Normalize weights
            self.config.success_weight /= total_weight
            self.config.performance_weight /= total_weight
            self.config.reliability_weight /= total_weight
            self.config.efficiency_weight /= total_weight
            self.config.quality_weight /= total_weight
    
    def score_single_result(self, 
                           result: RecipeTestResult,
                           recipe: Optional[RecipeDefinition] = None,
                           metric: ScoringMetrics = ScoringMetrics.COMPOSITE) -> ScoreBreakdown:
        """
        Score a single test result.
        
        Args:
            result: Test result to score
            recipe: Optional recipe definition for context
            metric: Scoring metric to use
            
        Returns:
            Score breakdown with detailed analysis
        """
        if metric == ScoringMetrics.SUCCESS_RATE:
            return self._score_success_rate([result])
        elif metric == ScoringMetrics.PERFORMANCE:
            return self._score_performance([result], recipe)
        elif metric == ScoringMetrics.RELIABILITY:
            return self._score_reliability([result])
        elif metric == ScoringMetrics.EFFICIENCY:
            return self._score_efficiency([result], recipe)
        elif metric == ScoringMetrics.QUALITY:
            return self._score_quality([result])
        elif metric == ScoringMetrics.CONSISTENCY:
            return self._score_consistency([result])
        elif metric == ScoringMetrics.ROBUSTNESS:
            return self._score_robustness([result])
        else:  # COMPOSITE
            return self._score_composite([result], recipe)
    
    def score_multiple_results(self, 
                              results: List[RecipeTestResult],
                              recipes: Optional[List[RecipeDefinition]] = None,
                              metric: ScoringMetrics = ScoringMetrics.COMPOSITE) -> ScoreBreakdown:
        """
        Score multiple test results.
        
        Args:
            results: List of test results to score
            recipes: Optional list of recipe definitions
            metric: Scoring metric to use
            
        Returns:
            Aggregated score breakdown
        """
        if not results:
            return ScoreBreakdown(
                total_score=0.0,
                explanation="No test results to score"
            )
        
        if metric == ScoringMetrics.SUCCESS_RATE:
            return self._score_success_rate(results)
        elif metric == ScoringMetrics.PERFORMANCE:
            return self._score_performance(results, recipes)
        elif metric == ScoringMetrics.RELIABILITY:
            return self._score_reliability(results)
        elif metric == ScoringMetrics.EFFICIENCY:
            return self._score_efficiency(results, recipes)
        elif metric == ScoringMetrics.QUALITY:
            return self._score_quality(results)
        elif metric == ScoringMetrics.CONSISTENCY:
            return self._score_consistency(results)
        elif metric == ScoringMetrics.ROBUSTNESS:
            return self._score_robustness(results)
        else:  # COMPOSITE
            return self._score_composite(results, recipes)
    
    def _score_success_rate(self, results: List[RecipeTestResult]) -> ScoreBreakdown:
        """Score based on success rate"""
        successful = sum(1 for r in results if r.success)
        total = len(results)
        success_rate = successful / total if total > 0 else 0.0
        
        # Convert to 0-1 score
        if success_rate >= self.config.excellent_success_rate:
            score = 1.0
            grade = "Excellent"
        elif success_rate >= self.config.good_success_rate:
            score = 0.8 + 0.2 * (success_rate - self.config.good_success_rate) / (self.config.excellent_success_rate - self.config.good_success_rate)
            grade = "Good"
        elif success_rate >= self.config.acceptable_success_rate:
            score = 0.6 + 0.2 * (success_rate - self.config.acceptable_success_rate) / (self.config.good_success_rate - self.config.acceptable_success_rate)
            grade = "Acceptable"
        else:
            score = 0.6 * success_rate / self.config.acceptable_success_rate
            grade = "Poor"
        
        return ScoreBreakdown(
            total_score=score,
            component_scores={"success_rate": success_rate},
            explanation=f"Success rate: {success_rate:.1%} ({successful}/{total}) - {grade}"
        )
    
    def _score_performance(self, 
                          results: List[RecipeTestResult],
                          recipes: Optional[List[RecipeDefinition]] = None) -> ScoreBreakdown:
        """Score based on performance metrics"""
        if not results:
            return ScoreBreakdown(total_score=0.0, explanation="No results for performance scoring")
        
        # Calculate average execution time
        avg_time = sum(r.execution_time_ms for r in results) / len(results)
        avg_memory = sum(r.memory_usage_mb for r in results) / len(results)
        
        # Score execution time
        if avg_time <= self.config.excellent_time_ms:
            time_score = 1.0
            time_grade = "Excellent"
        elif avg_time <= self.config.good_time_ms:
            time_score = 0.8 + 0.2 * (self.config.good_time_ms - avg_time) / (self.config.good_time_ms - self.config.excellent_time_ms)
            time_grade = "Good"
        elif avg_time <= self.config.acceptable_time_ms:
            time_score = 0.6 + 0.2 * (self.config.acceptable_time_ms - avg_time) / (self.config.acceptable_time_ms - self.config.good_time_ms)
            time_grade = "Acceptable"
        else:
            time_score = max(0.0, 0.6 * self.config.acceptable_time_ms / avg_time)
            time_grade = "Poor"
        
        # Score memory usage
        if avg_memory <= self.config.excellent_memory_mb:
            memory_score = 1.0
            memory_grade = "Excellent"
        elif avg_memory <= self.config.good_memory_mb:
            memory_score = 0.8 + 0.2 * (self.config.good_memory_mb - avg_memory) / (self.config.good_memory_mb - self.config.excellent_memory_mb)
            memory_grade = "Good"
        elif avg_memory <= self.config.acceptable_memory_mb:
            memory_score = 0.6 + 0.2 * (self.config.acceptable_memory_mb - avg_memory) / (self.config.acceptable_memory_mb - self.config.good_memory_mb)
            memory_grade = "Acceptable"
        else:
            memory_score = max(0.0, 0.6 * self.config.acceptable_memory_mb / avg_memory)
            memory_grade = "Poor"
        
        # Combine scores (70% time, 30% memory)
        total_score = 0.7 * time_score + 0.3 * memory_score
        
        return ScoreBreakdown(
            total_score=total_score,
            component_scores={
                "execution_time": time_score,
                "memory_usage": memory_score,
                "avg_time_ms": avg_time,
                "avg_memory_mb": avg_memory
            },
            weights={"execution_time": 0.7, "memory_usage": 0.3},
            explanation=f"Performance: Time {avg_time:.0f}ms ({time_grade}), Memory {avg_memory:.0f}MB ({memory_grade})"
        )
    
    def _score_reliability(self, results: List[RecipeTestResult]) -> ScoreBreakdown:
        """Score based on reliability metrics"""
        if not results:
            return ScoreBreakdown(total_score=0.0, explanation="No results for reliability scoring")
        
        # Calculate reliability metrics
        successful = sum(1 for r in results if r.success)
        total = len(results)
        success_rate = successful / total
        
        # Calculate consistency (low variance in execution time)
        times = [r.execution_time_ms for r in results]
        avg_time = sum(times) / len(times)
        variance = sum((t - avg_time) ** 2 for t in times) / len(times)
        cv = math.sqrt(variance) / avg_time if avg_time > 0 else 0  # Coefficient of variation
        
        # Consistency score (lower CV is better)
        consistency_score = max(0.0, 1.0 - cv / 2.0)  # CV > 2.0 gives 0 score
        
        # Error rate
        error_rate = 1.0 - success_rate
        error_score = 1.0 - error_rate
        
        # Combine scores
        total_score = 0.6 * error_score + 0.4 * consistency_score
        
        return ScoreBreakdown(
            total_score=total_score,
            component_scores={
                "success_rate": success_rate,
                "consistency": consistency_score,
                "error_rate": error_rate,
                "coefficient_of_variation": cv
            },
            weights={"error_score": 0.6, "consistency": 0.4},
            explanation=f"Reliability: {success_rate:.1%} success, CV={cv:.2f} (consistency)"
        )
    
    def _score_efficiency(self, 
                         results: List[RecipeTestResult],
                         recipes: Optional[List[RecipeDefinition]] = None) -> ScoreBreakdown:
        """Score based on efficiency metrics"""
        if not results:
            return ScoreBreakdown(total_score=0.0, explanation="No results for efficiency scoring")
        
        # Resource efficiency (time and memory per successful operation)
        successful_results = [r for r in results if r.success]
        if not successful_results:
            return ScoreBreakdown(total_score=0.0, explanation="No successful results for efficiency scoring")
        
        avg_time_per_success = sum(r.execution_time_ms for r in successful_results) / len(successful_results)
        avg_memory_per_success = sum(r.memory_usage_mb for r in successful_results) / len(successful_results)
        
        # Scenario efficiency (scenarios passed per unit time)
        total_scenarios_passed = sum(r.test_scenarios_passed for r in successful_results)
        total_time = sum(r.execution_time_ms for r in successful_results)
        scenarios_per_second = (total_scenarios_passed * 1000) / total_time if total_time > 0 else 0
        
        # Score components
        time_efficiency = min(1.0, self.config.excellent_time_ms / avg_time_per_success) if avg_time_per_success > 0 else 1.0
        memory_efficiency = min(1.0, self.config.excellent_memory_mb / avg_memory_per_success) if avg_memory_per_success > 0 else 1.0
        throughput_score = min(1.0, scenarios_per_second / 10.0)  # 10 scenarios/second = perfect
        
        # Combine scores
        total_score = 0.4 * time_efficiency + 0.3 * memory_efficiency + 0.3 * throughput_score
        
        return ScoreBreakdown(
            total_score=total_score,
            component_scores={
                "time_efficiency": time_efficiency,
                "memory_efficiency": memory_efficiency,
                "throughput": throughput_score,
                "scenarios_per_second": scenarios_per_second
            },
            weights={"time": 0.4, "memory": 0.3, "throughput": 0.3},
            explanation=f"Efficiency: {avg_time_per_success:.0f}ms/success, {scenarios_per_second:.1f} scenarios/sec"
        )
    
    def _score_quality(self, results: List[RecipeTestResult]) -> ScoreBreakdown:
        """Score based on quality metrics"""
        if not results:
            return ScoreBreakdown(total_score=0.0, explanation="No results for quality scoring")
        
        # Calculate quality metrics
        total_scenarios = sum(r.test_scenarios_total for r in results)
        passed_scenarios = sum(r.test_scenarios_passed for r in results)
        scenario_pass_rate = passed_scenarios / total_scenarios if total_scenarios > 0 else 0.0
        
        # Average score from individual results
        avg_individual_score = sum(r.score for r in results) / len(results)
        
        # Quality score based on scenario pass rate
        if scenario_pass_rate >= self.config.excellent_scenario_pass_rate:
            quality_score = 1.0
        elif scenario_pass_rate >= self.config.good_scenario_pass_rate:
            quality_score = 0.8 + 0.2 * (scenario_pass_rate - self.config.good_scenario_pass_rate) / (self.config.excellent_scenario_pass_rate - self.config.good_scenario_pass_rate)
        elif scenario_pass_rate >= self.config.acceptable_scenario_pass_rate:
            quality_score = 0.6 + 0.2 * (scenario_pass_rate - self.config.acceptable_scenario_pass_rate) / (self.config.good_scenario_pass_rate - self.config.acceptable_scenario_pass_rate)
        else:
            quality_score = 0.6 * scenario_pass_rate / self.config.acceptable_scenario_pass_rate
        
        # Combine with individual scores
        total_score = 0.6 * quality_score + 0.4 * avg_individual_score
        
        return ScoreBreakdown(
            total_score=total_score,
            component_scores={
                "scenario_pass_rate": scenario_pass_rate,
                "average_individual_score": avg_individual_score,
                "quality_score": quality_score
            },
            weights={"scenario_quality": 0.6, "individual_scores": 0.4},
            explanation=f"Quality: {scenario_pass_rate:.1%} scenarios passed, avg score {avg_individual_score:.2f}"
        )
    
    def _score_consistency(self, results: List[RecipeTestResult]) -> ScoreBreakdown:
        """Score based on consistency across multiple runs"""
        if len(results) < 2:
            return ScoreBreakdown(
                total_score=1.0 if results and results[0].success else 0.0,
                explanation="Single result - consistency not applicable"
            )
        
        # Calculate variance in scores
        scores = [r.score for r in results]
        avg_score = sum(scores) / len(scores)
        variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
        std_dev = math.sqrt(variance)
        
        # Calculate variance in execution times
        times = [r.execution_time_ms for r in results]
        avg_time = sum(times) / len(times)
        time_variance = sum((t - avg_time) ** 2 for t in times) / len(times)
        time_cv = math.sqrt(time_variance) / avg_time if avg_time > 0 else 0
        
        # Consistency scores (lower variance is better)
        score_consistency = max(0.0, 1.0 - std_dev / 0.5)  # std_dev > 0.5 gives 0 score
        time_consistency = max(0.0, 1.0 - time_cv / 1.0)   # CV > 1.0 gives 0 score
        
        # Success consistency
        success_rate = sum(1 for r in results if r.success) / len(results)
        success_consistency = 1.0 if success_rate in [0.0, 1.0] else 1.0 - abs(success_rate - 0.5) * 2
        
        # Combine scores
        total_score = 0.4 * score_consistency + 0.3 * time_consistency + 0.3 * success_consistency
        
        return ScoreBreakdown(
            total_score=total_score,
            component_scores={
                "score_consistency": score_consistency,
                "time_consistency": time_consistency,
                "success_consistency": success_consistency,
                "score_std_dev": std_dev,
                "time_cv": time_cv
            },
            weights={"score": 0.4, "time": 0.3, "success": 0.3},
            explanation=f"Consistency: Score σ={std_dev:.3f}, Time CV={time_cv:.3f}, Success={success_rate:.1%}"
        )
    
    def _score_robustness(self, results: List[RecipeTestResult]) -> ScoreBreakdown:
        """Score based on robustness (handling edge cases and errors)"""
        if not results:
            return ScoreBreakdown(total_score=0.0, explanation="No results for robustness scoring")
        
        # Count different types of scenarios
        edge_case_results = []
        normal_case_results = []
        
        for result in results:
            # Simple heuristic: if execution time is very short or very long, or if it failed, consider it an edge case
            if (result.execution_time_ms < 100 or 
                result.execution_time_ms > 30000 or 
                not result.success or
                result.memory_usage_mb > 2048):
                edge_case_results.append(result)
            else:
                normal_case_results.append(result)
        
        # Robustness metrics
        total_tests = len(results)
        edge_case_count = len(edge_case_results)
        edge_case_success = sum(1 for r in edge_case_results if r.success)
        normal_case_success = sum(1 for r in normal_case_results if r.success)
        
        # Edge case handling score
        edge_case_score = edge_case_success / edge_case_count if edge_case_count > 0 else 1.0
        
        # Normal case score
        normal_case_score = normal_case_success / len(normal_case_results) if normal_case_results else 1.0
        
        # Error recovery (how well it handles failures)
        failed_results = [r for r in results if not r.success]
        graceful_failures = sum(1 for r in failed_results if r.error_message and len(r.error_message) > 10)
        error_recovery_score = graceful_failures / len(failed_results) if failed_results else 1.0
        
        # Combine scores
        total_score = 0.5 * edge_case_score + 0.3 * normal_case_score + 0.2 * error_recovery_score
        
        return ScoreBreakdown(
            total_score=total_score,
            component_scores={
                "edge_case_success_rate": edge_case_score,
                "normal_case_success_rate": normal_case_score,
                "error_recovery_score": error_recovery_score,
                "edge_cases_tested": edge_case_count,
                "total_tests": total_tests
            },
            weights={"edge_cases": 0.5, "normal_cases": 0.3, "error_recovery": 0.2},
            explanation=f"Robustness: {edge_case_count} edge cases, {edge_case_score:.1%} edge success, {error_recovery_score:.1%} graceful failures"
        )
    
    def _score_composite(self, 
                        results: List[RecipeTestResult],
                        recipes: Optional[List[RecipeDefinition]] = None) -> ScoreBreakdown:
        """Calculate composite score using all metrics"""
        if not results:
            return ScoreBreakdown(total_score=0.0, explanation="No results for composite scoring")
        
        # Calculate component scores
        success_breakdown = self._score_success_rate(results)
        performance_breakdown = self._score_performance(results, recipes)
        reliability_breakdown = self._score_reliability(results)
        efficiency_breakdown = self._score_efficiency(results, recipes)
        quality_breakdown = self._score_quality(results)
        
        # Calculate weighted composite score
        composite_score = (
            success_breakdown.total_score * self.config.success_weight +
            performance_breakdown.total_score * self.config.performance_weight +
            reliability_breakdown.total_score * self.config.reliability_weight +
            efficiency_breakdown.total_score * self.config.efficiency_weight +
            quality_breakdown.total_score * self.config.quality_weight
        )
        
        # Combine component scores
        component_scores = {
            "success": success_breakdown.total_score,
            "performance": performance_breakdown.total_score,
            "reliability": reliability_breakdown.total_score,
            "efficiency": efficiency_breakdown.total_score,
            "quality": quality_breakdown.total_score
        }
        
        weights = {
            "success": self.config.success_weight,
            "performance": self.config.performance_weight,
            "reliability": self.config.reliability_weight,
            "efficiency": self.config.efficiency_weight,
            "quality": self.config.quality_weight
        }
        
        # Create explanation
        explanation = (
            f"Composite Score: Success {success_breakdown.total_score:.2f}, "
            f"Performance {performance_breakdown.total_score:.2f}, "
            f"Reliability {reliability_breakdown.total_score:.2f}, "
            f"Efficiency {efficiency_breakdown.total_score:.2f}, "
            f"Quality {quality_breakdown.total_score:.2f}"
        )
        
        return ScoreBreakdown(
            total_score=composite_score,
            component_scores=component_scores,
            weights=weights,
            explanation=explanation
        )
