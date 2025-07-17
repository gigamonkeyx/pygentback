"""
Recipe Analyzer

Advanced analytics for test recipe performance, patterns, and optimization opportunities.
Provides comprehensive analysis of recipe execution data and recommendations.
"""

import logging
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from enum import Enum
import json

logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """Types of analysis"""
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    EFFICIENCY = "efficiency"
    TRENDS = "trends"
    PATTERNS = "patterns"
    OPTIMIZATION = "optimization"


@dataclass
class PerformanceMetrics:
    """Performance metrics for analysis"""
    execution_time_avg: float
    execution_time_median: float
    execution_time_std: float
    success_rate: float
    failure_rate: float
    timeout_rate: float
    resource_utilization: Dict[str, float]
    throughput: float


@dataclass
class ReliabilityMetrics:
    """Reliability metrics for analysis"""
    consistency_score: float
    stability_index: float
    error_diversity: float
    recovery_rate: float
    mean_time_to_failure: float
    mean_time_to_recovery: float


@dataclass
class AnalysisResult:
    """Result of recipe analysis"""
    recipe_id: str
    analysis_type: AnalysisType
    metrics: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
    confidence_score: float
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ComparisonResult:
    """Result of recipe comparison"""
    recipe_ids: List[str]
    comparison_metrics: Dict[str, Dict[str, float]]
    winner: Optional[str]
    significant_differences: List[str]
    recommendations: List[str]


class RecipeAnalyzer:
    """
    Advanced Recipe Analysis System.
    
    Provides comprehensive analysis of test recipe performance, reliability,
    and optimization opportunities using statistical and ML techniques.
    """
    
    def __init__(self):
        # Analysis data storage
        self.execution_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.analysis_history: List[AnalysisResult] = []
        self.baseline_metrics: Dict[str, Dict[str, float]] = {}
        
        # Analysis configuration
        self.min_data_points = 5
        self.confidence_threshold = 0.7
        self.significance_threshold = 0.05
        
        # Pattern detection
        self.known_patterns: Dict[str, List[str]] = {}
        self.anomaly_threshold = 2.0  # Standard deviations
    
    def add_execution_data(self, recipe_id: str, execution_data: Dict[str, Any]):
        """Add execution data for analysis"""
        execution_data["timestamp"] = execution_data.get("timestamp", datetime.utcnow())
        self.execution_data[recipe_id].append(execution_data)
        
        # Keep only recent data (last 1000 executions)
        if len(self.execution_data[recipe_id]) > 1000:
            self.execution_data[recipe_id] = self.execution_data[recipe_id][-1000:]
        
        logger.debug(f"Added execution data for recipe {recipe_id}")
    
    def analyze_performance(self, recipe_id: str, time_window_hours: int = 24) -> AnalysisResult:
        """Analyze performance metrics for a recipe"""
        data = self._get_recent_data(recipe_id, time_window_hours)
        
        if len(data) < self.min_data_points:
            return AnalysisResult(
                recipe_id=recipe_id,
                analysis_type=AnalysisType.PERFORMANCE,
                metrics={},
                insights=[f"Insufficient data: only {len(data)} executions"],
                recommendations=["Collect more execution data for meaningful analysis"],
                confidence_score=0.0
            )
        
        # Calculate performance metrics
        execution_times = [d.get("execution_time", 0) for d in data]
        success_count = sum(1 for d in data if d.get("success", False))
        timeout_count = sum(1 for d in data if d.get("timeout", False))
        
        metrics = PerformanceMetrics(
            execution_time_avg=statistics.mean(execution_times),
            execution_time_median=statistics.median(execution_times),
            execution_time_std=statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0,
            success_rate=success_count / len(data),
            failure_rate=(len(data) - success_count) / len(data),
            timeout_rate=timeout_count / len(data),
            resource_utilization=self._calculate_resource_utilization(data),
            throughput=len(data) / time_window_hours
        )
        
        # Generate insights
        insights = self._generate_performance_insights(metrics, data)
        
        # Generate recommendations
        recommendations = self._generate_performance_recommendations(metrics, data)
        
        # Calculate confidence score
        confidence = min(1.0, len(data) / 50.0)  # Higher confidence with more data
        
        result = AnalysisResult(
            recipe_id=recipe_id,
            analysis_type=AnalysisType.PERFORMANCE,
            metrics=self._metrics_to_dict(metrics),
            insights=insights,
            recommendations=recommendations,
            confidence_score=confidence
        )
        
        self.analysis_history.append(result)
        return result
    
    def analyze_reliability(self, recipe_id: str, time_window_hours: int = 168) -> AnalysisResult:
        """Analyze reliability metrics for a recipe (default: 1 week)"""
        data = self._get_recent_data(recipe_id, time_window_hours)
        
        if len(data) < self.min_data_points:
            return AnalysisResult(
                recipe_id=recipe_id,
                analysis_type=AnalysisType.RELIABILITY,
                metrics={},
                insights=[f"Insufficient data: only {len(data)} executions"],
                recommendations=["Collect more execution data for reliability analysis"],
                confidence_score=0.0
            )
        
        # Calculate reliability metrics
        metrics = self._calculate_reliability_metrics(data)
        
        # Generate insights
        insights = self._generate_reliability_insights(metrics, data)
        
        # Generate recommendations
        recommendations = self._generate_reliability_recommendations(metrics, data)
        
        # Calculate confidence score
        confidence = min(1.0, len(data) / 100.0)
        
        result = AnalysisResult(
            recipe_id=recipe_id,
            analysis_type=AnalysisType.RELIABILITY,
            metrics=self._metrics_to_dict(metrics),
            insights=insights,
            recommendations=recommendations,
            confidence_score=confidence
        )
        
        self.analysis_history.append(result)
        return result
    
    def analyze_trends(self, recipe_id: str, time_window_hours: int = 168) -> AnalysisResult:
        """Analyze trends in recipe performance over time"""
        data = self._get_recent_data(recipe_id, time_window_hours)
        
        if len(data) < 10:  # Need more data for trend analysis
            return AnalysisResult(
                recipe_id=recipe_id,
                analysis_type=AnalysisType.TRENDS,
                metrics={},
                insights=["Insufficient data for trend analysis"],
                recommendations=["Collect more historical data"],
                confidence_score=0.0
            )
        
        # Sort data by timestamp
        sorted_data = sorted(data, key=lambda x: x.get("timestamp", datetime.min))
        
        # Calculate trends
        trends = self._calculate_trends(sorted_data)
        
        # Generate insights
        insights = self._generate_trend_insights(trends, sorted_data)
        
        # Generate recommendations
        recommendations = self._generate_trend_recommendations(trends, sorted_data)
        
        confidence = min(1.0, len(data) / 50.0)
        
        result = AnalysisResult(
            recipe_id=recipe_id,
            analysis_type=AnalysisType.TRENDS,
            metrics=trends,
            insights=insights,
            recommendations=recommendations,
            confidence_score=confidence
        )
        
        self.analysis_history.append(result)
        return result
    
    def compare_recipes(self, recipe_ids: List[str], time_window_hours: int = 24) -> ComparisonResult:
        """Compare performance of multiple recipes"""
        if len(recipe_ids) < 2:
            raise ValueError("Need at least 2 recipes for comparison")
        
        # Collect metrics for each recipe
        recipe_metrics = {}
        for recipe_id in recipe_ids:
            data = self._get_recent_data(recipe_id, time_window_hours)
            if len(data) >= self.min_data_points:
                recipe_metrics[recipe_id] = self._calculate_comparison_metrics(data)
        
        if len(recipe_metrics) < 2:
            return ComparisonResult(
                recipe_ids=recipe_ids,
                comparison_metrics={},
                winner=None,
                significant_differences=[],
                recommendations=["Insufficient data for comparison"]
            )
        
        # Find winner based on composite score
        winner = self._determine_winner(recipe_metrics)
        
        # Identify significant differences
        significant_differences = self._identify_significant_differences(recipe_metrics)
        
        # Generate recommendations
        recommendations = self._generate_comparison_recommendations(recipe_metrics, winner)
        
        return ComparisonResult(
            recipe_ids=recipe_ids,
            comparison_metrics=recipe_metrics,
            winner=winner,
            significant_differences=significant_differences,
            recommendations=recommendations
        )
    
    def detect_anomalies(self, recipe_id: str, time_window_hours: int = 24) -> List[Dict[str, Any]]:
        """Detect anomalies in recipe execution"""
        data = self._get_recent_data(recipe_id, time_window_hours)
        
        if len(data) < 10:
            return []
        
        anomalies = []
        
        # Detect execution time anomalies
        execution_times = [d.get("execution_time", 0) for d in data]
        mean_time = statistics.mean(execution_times)
        std_time = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
        
        for i, d in enumerate(data):
            exec_time = d.get("execution_time", 0)
            if std_time > 0 and abs(exec_time - mean_time) > self.anomaly_threshold * std_time:
                anomalies.append({
                    "type": "execution_time_anomaly",
                    "timestamp": d.get("timestamp"),
                    "value": exec_time,
                    "expected_range": (mean_time - std_time, mean_time + std_time),
                    "severity": "high" if abs(exec_time - mean_time) > 3 * std_time else "medium"
                })
        
        # Detect success rate anomalies
        recent_success_rate = sum(1 for d in data[-10:] if d.get("success", False)) / min(10, len(data))
        overall_success_rate = sum(1 for d in data if d.get("success", False)) / len(data)
        
        if abs(recent_success_rate - overall_success_rate) > 0.2:
            anomalies.append({
                "type": "success_rate_anomaly",
                "timestamp": datetime.utcnow(),
                "recent_rate": recent_success_rate,
                "overall_rate": overall_success_rate,
                "severity": "high" if abs(recent_success_rate - overall_success_rate) > 0.4 else "medium"
            })
        
        return anomalies
    
    def _get_recent_data(self, recipe_id: str, hours: int) -> List[Dict[str, Any]]:
        """Get recent execution data for a recipe"""
        if recipe_id not in self.execution_data:
            return []
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            d for d in self.execution_data[recipe_id]
            if d.get("timestamp", datetime.min) > cutoff_time
        ]
    
    def _calculate_resource_utilization(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate resource utilization metrics"""
        cpu_usage = [d.get("cpu_usage", 0) for d in data if "cpu_usage" in d]
        memory_usage = [d.get("memory_usage", 0) for d in data if "memory_usage" in d]
        
        return {
            "cpu_avg": statistics.mean(cpu_usage) if cpu_usage else 0.0,
            "memory_avg": statistics.mean(memory_usage) if memory_usage else 0.0,
            "cpu_max": max(cpu_usage) if cpu_usage else 0.0,
            "memory_max": max(memory_usage) if memory_usage else 0.0
        }
    
    def _calculate_reliability_metrics(self, data: List[Dict[str, Any]]) -> ReliabilityMetrics:
        """Calculate reliability metrics"""
        # Consistency score (inverse of coefficient of variation)
        execution_times = [d.get("execution_time", 0) for d in data]
        mean_time = statistics.mean(execution_times)
        std_time = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
        consistency_score = 1.0 - (std_time / mean_time) if mean_time > 0 else 0.0
        
        # Stability index (success rate over time)
        success_rates = []
        window_size = max(5, len(data) // 10)
        for i in range(0, len(data), window_size):
            window_data = data[i:i + window_size]
            success_rate = sum(1 for d in window_data if d.get("success", False)) / len(window_data)
            success_rates.append(success_rate)
        
        stability_index = 1.0 - statistics.stdev(success_rates) if len(success_rates) > 1 else 1.0
        
        # Error diversity (number of unique error types)
        error_types = set()
        for d in data:
            if not d.get("success", False) and d.get("error_type"):
                error_types.add(d["error_type"])
        error_diversity = len(error_types) / len(data) if data else 0.0
        
        # Recovery rate (success after failure)
        recovery_count = 0
        for i in range(1, len(data)):
            if not data[i-1].get("success", False) and data[i].get("success", False):
                recovery_count += 1
        
        failure_count = sum(1 for d in data if not d.get("success", False))
        recovery_rate = recovery_count / failure_count if failure_count > 0 else 1.0
        
        return ReliabilityMetrics(
            consistency_score=max(0.0, min(1.0, consistency_score)),
            stability_index=max(0.0, min(1.0, stability_index)),
            error_diversity=error_diversity,
            recovery_rate=recovery_rate,
            mean_time_to_failure=mean_time * (1.0 / (1.0 - sum(1 for d in data if d.get("success", False)) / len(data))) if data else 0.0,
            mean_time_to_recovery=mean_time * 0.5  # Simplified calculation
        )
    
    def _calculate_trends(self, sorted_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate trend metrics"""
        # Split data into time windows
        window_size = max(5, len(sorted_data) // 10)
        windows = []
        
        for i in range(0, len(sorted_data), window_size):
            window_data = sorted_data[i:i + window_size]
            if len(window_data) >= 3:  # Minimum for meaningful metrics
                window_metrics = {
                    "avg_execution_time": statistics.mean([d.get("execution_time", 0) for d in window_data]),
                    "success_rate": sum(1 for d in window_data if d.get("success", False)) / len(window_data),
                    "timestamp": window_data[len(window_data)//2].get("timestamp", datetime.utcnow())
                }
                windows.append(window_metrics)
        
        if len(windows) < 2:
            return {"trend_direction": "insufficient_data"}
        
        # Calculate trends
        execution_time_trend = self._calculate_linear_trend([w["avg_execution_time"] for w in windows])
        success_rate_trend = self._calculate_linear_trend([w["success_rate"] for w in windows])
        
        return {
            "execution_time_trend": execution_time_trend,
            "success_rate_trend": success_rate_trend,
            "trend_direction": self._determine_overall_trend(execution_time_trend, success_rate_trend),
            "windows_analyzed": len(windows),
            "data_points": len(sorted_data)
        }
    
    def _calculate_linear_trend(self, values: List[float]) -> Dict[str, float]:
        """Calculate linear trend (slope) for a series of values"""
        if len(values) < 2:
            return {"slope": 0.0, "direction": "stable"}
        
        n = len(values)
        x_values = list(range(n))
        
        # Calculate slope using least squares
        sum_x = sum(x_values)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(x_values, values))
        sum_x2 = sum(x * x for x in x_values)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Determine direction
        if abs(slope) < 0.01:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        return {"slope": slope, "direction": direction}
    
    def _determine_overall_trend(self, exec_trend: Dict[str, float], success_trend: Dict[str, float]) -> str:
        """Determine overall trend direction"""
        exec_dir = exec_trend.get("direction", "stable")
        success_dir = success_trend.get("direction", "stable")
        
        if exec_dir == "decreasing" and success_dir == "increasing":
            return "improving"
        elif exec_dir == "increasing" and success_dir == "decreasing":
            return "degrading"
        elif exec_dir == "stable" and success_dir == "stable":
            return "stable"
        else:
            return "mixed"
    
    def _generate_performance_insights(self, metrics: PerformanceMetrics, data: List[Dict[str, Any]]) -> List[str]:
        """Generate performance insights"""
        insights = []
        
        if metrics.success_rate < 0.8:
            insights.append(f"Low success rate: {metrics.success_rate:.1%}")
        
        if metrics.execution_time_std > metrics.execution_time_avg * 0.5:
            insights.append("High execution time variability detected")
        
        if metrics.timeout_rate > 0.1:
            insights.append(f"Frequent timeouts: {metrics.timeout_rate:.1%} of executions")
        
        if metrics.resource_utilization.get("cpu_avg", 0) > 80:
            insights.append("High CPU utilization detected")
        
        if metrics.throughput < 1.0:
            insights.append("Low throughput: less than 1 execution per hour")
        
        return insights
    
    def _generate_performance_recommendations(self, metrics: PerformanceMetrics, data: List[Dict[str, Any]]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        if metrics.success_rate < 0.9:
            recommendations.append("Investigate and fix failing test cases")
        
        if metrics.execution_time_avg > 60:
            recommendations.append("Consider optimizing slow operations")
        
        if metrics.timeout_rate > 0.05:
            recommendations.append("Review and adjust timeout settings")
        
        if metrics.execution_time_std > metrics.execution_time_avg * 0.3:
            recommendations.append("Investigate causes of execution time variability")
        
        return recommendations
    
    def _generate_reliability_insights(self, metrics: ReliabilityMetrics, data: List[Dict[str, Any]]) -> List[str]:
        """Generate reliability insights"""
        insights = []
        
        if metrics.consistency_score < 0.7:
            insights.append("Low consistency in execution times")
        
        if metrics.stability_index < 0.8:
            insights.append("Unstable success rate over time")
        
        if metrics.error_diversity > 0.1:
            insights.append("High diversity of error types")
        
        if metrics.recovery_rate < 0.5:
            insights.append("Poor recovery rate after failures")
        
        return insights
    
    def _generate_reliability_recommendations(self, metrics: ReliabilityMetrics, data: List[Dict[str, Any]]) -> List[str]:
        """Generate reliability recommendations"""
        recommendations = []
        
        if metrics.consistency_score < 0.8:
            recommendations.append("Implement more consistent execution environment")
        
        if metrics.stability_index < 0.9:
            recommendations.append("Investigate causes of instability")
        
        if metrics.recovery_rate < 0.7:
            recommendations.append("Improve error handling and recovery mechanisms")
        
        return recommendations
    
    def _generate_trend_insights(self, trends: Dict[str, Any], data: List[Dict[str, Any]]) -> List[str]:
        """Generate trend insights"""
        insights = []
        
        trend_direction = trends.get("trend_direction", "unknown")
        
        if trend_direction == "degrading":
            insights.append("Performance is degrading over time")
        elif trend_direction == "improving":
            insights.append("Performance is improving over time")
        elif trend_direction == "stable":
            insights.append("Performance is stable over time")
        
        exec_trend = trends.get("execution_time_trend", {})
        if exec_trend.get("direction") == "increasing":
            insights.append("Execution times are increasing")
        
        success_trend = trends.get("success_rate_trend", {})
        if success_trend.get("direction") == "decreasing":
            insights.append("Success rate is declining")
        
        return insights
    
    def _generate_trend_recommendations(self, trends: Dict[str, Any], data: List[Dict[str, Any]]) -> List[str]:
        """Generate trend recommendations"""
        recommendations = []
        
        if trends.get("trend_direction") == "degrading":
            recommendations.append("Investigate recent changes that may have caused degradation")
        
        exec_trend = trends.get("execution_time_trend", {})
        if exec_trend.get("direction") == "increasing":
            recommendations.append("Monitor for performance regressions")
        
        success_trend = trends.get("success_rate_trend", {})
        if success_trend.get("direction") == "decreasing":
            recommendations.append("Review recent failures for patterns")
        
        return recommendations
    
    def _calculate_comparison_metrics(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate metrics for recipe comparison"""
        execution_times = [d.get("execution_time", 0) for d in data]
        success_count = sum(1 for d in data if d.get("success", False))
        
        return {
            "avg_execution_time": statistics.mean(execution_times),
            "success_rate": success_count / len(data),
            "reliability_score": success_count / len(data) * (1.0 - statistics.stdev(execution_times) / statistics.mean(execution_times)) if execution_times else 0.0,
            "efficiency_score": (success_count / len(data)) / statistics.mean(execution_times) if execution_times else 0.0
        }
    
    def _determine_winner(self, recipe_metrics: Dict[str, Dict[str, float]]) -> Optional[str]:
        """Determine winner based on composite score"""
        composite_scores = {}
        
        for recipe_id, metrics in recipe_metrics.items():
            # Weighted composite score
            score = (
                metrics.get("success_rate", 0) * 0.4 +
                metrics.get("reliability_score", 0) * 0.3 +
                metrics.get("efficiency_score", 0) * 0.3
            )
            composite_scores[recipe_id] = score
        
        if composite_scores:
            return max(composite_scores.items(), key=lambda x: x[1])[0]
        return None
    
    def _identify_significant_differences(self, recipe_metrics: Dict[str, Dict[str, float]]) -> List[str]:
        """Identify significant differences between recipes"""
        differences = []
        
        recipe_ids = list(recipe_metrics.keys())
        for i in range(len(recipe_ids)):
            for j in range(i + 1, len(recipe_ids)):
                recipe1, recipe2 = recipe_ids[i], recipe_ids[j]
                metrics1, metrics2 = recipe_metrics[recipe1], recipe_metrics[recipe2]
                
                for metric_name in metrics1.keys():
                    if metric_name in metrics2:
                        diff = abs(metrics1[metric_name] - metrics2[metric_name])
                        if diff > 0.1:  # 10% difference threshold
                            differences.append(
                                f"{metric_name}: {recipe1} ({metrics1[metric_name]:.3f}) vs "
                                f"{recipe2} ({metrics2[metric_name]:.3f})"
                            )
        
        return differences
    
    def _generate_comparison_recommendations(self, recipe_metrics: Dict[str, Dict[str, float]], 
                                           winner: Optional[str]) -> List[str]:
        """Generate comparison recommendations"""
        recommendations = []
        
        if winner:
            recommendations.append(f"Consider using {winner} as the primary recipe")
            
            winner_metrics = recipe_metrics[winner]
            for recipe_id, metrics in recipe_metrics.items():
                if recipe_id != winner:
                    if metrics.get("success_rate", 0) < winner_metrics.get("success_rate", 0) - 0.1:
                        recommendations.append(f"Improve success rate for {recipe_id}")
                    if metrics.get("avg_execution_time", float('inf')) > winner_metrics.get("avg_execution_time", 0) * 1.5:
                        recommendations.append(f"Optimize execution time for {recipe_id}")
        
        return recommendations
    
    def _metrics_to_dict(self, metrics) -> Dict[str, Any]:
        """Convert metrics dataclass to dictionary"""
        if hasattr(metrics, '__dict__'):
            return metrics.__dict__
        return {}
    
    def get_analysis_summary(self, recipe_id: str) -> Dict[str, Any]:
        """Get summary of all analyses for a recipe"""
        recipe_analyses = [a for a in self.analysis_history if a.recipe_id == recipe_id]
        
        if not recipe_analyses:
            return {"recipe_id": recipe_id, "analyses": 0}
        
        latest_analyses = {}
        for analysis in recipe_analyses:
            analysis_type = analysis.analysis_type.value
            if (analysis_type not in latest_analyses or 
                analysis.analysis_timestamp > latest_analyses[analysis_type].analysis_timestamp):
                latest_analyses[analysis_type] = analysis
        
        return {
            "recipe_id": recipe_id,
            "total_analyses": len(recipe_analyses),
            "latest_analyses": {
                analysis_type: {
                    "confidence": analysis.confidence_score,
                    "insights_count": len(analysis.insights),
                    "recommendations_count": len(analysis.recommendations),
                    "timestamp": analysis.analysis_timestamp.isoformat()
                }
                for analysis_type, analysis in latest_analyses.items()
            },
            "data_points": len(self.execution_data.get(recipe_id, []))
        }
