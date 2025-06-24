"""
Performance Tracker

Tracks and analyzes system performance metrics for optimization.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import statistics

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    name: str
    value: float
    timestamp: datetime
    unit: str = ""
    category: str = "general"


@dataclass
class PerformanceReport:
    """Performance analysis report"""
    timespan_hours: float
    total_metrics: int
    summary_stats: Dict[str, Any]
    recommendations: List[str]
    trends: Dict[str, str]


class PerformanceTracker:
    """Tracks system and application performance metrics."""
    
    def __init__(self, max_metrics: int = 10000):
        self.max_metrics = max_metrics
        self.metrics: List[PerformanceMetric] = []
        self.baseline_metrics: Dict[str, float] = {}
        
        logger.info("Performance Tracker initialized")
    
    def record_metric(self, name: str, value: float, unit: str = "", category: str = "general"):
        """Record a performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            unit=unit,
            category=category
        )
        
        self.metrics.append(metric)
        
        # Keep only recent metrics
        if len(self.metrics) > self.max_metrics:
            self.metrics = self.metrics[-self.max_metrics:]
    
    def get_recent_metrics(self, name: str, hours: float = 1.0) -> List[PerformanceMetric]:
        """Get recent metrics for a specific metric name."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            m for m in self.metrics 
            if m.name == name and m.timestamp > cutoff_time
        ]
    
    def get_metric_stats(self, name: str, hours: float = 1.0) -> Dict[str, float]:
        """Get statistics for a metric over time period."""
        recent_metrics = self.get_recent_metrics(name, hours)
        
        if not recent_metrics:
            return {}
        
        values = [m.value for m in recent_metrics]
        
        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'latest': values[-1]
        }
    
    def set_baseline(self, name: str, value: float):
        """Set baseline value for a metric."""
        self.baseline_metrics[name] = value
        logger.info(f"Set baseline for {name}: {value}")
    
    def get_performance_report(self, hours: float = 24.0) -> PerformanceReport:
        """Generate comprehensive performance report."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter metrics for time period
        period_metrics = [
            m for m in self.metrics 
            if m.timestamp > cutoff_time
        ]
        
        # Group by metric name
        metric_groups = {}
        for metric in period_metrics:
            if metric.name not in metric_groups:
                metric_groups[metric.name] = []
            metric_groups[metric.name].append(metric)
        
        # Calculate summary statistics
        summary_stats = {}
        trends = {}
        recommendations = []
        
        for name, metrics in metric_groups.items():
            values = [m.value for m in metrics]
            
            if values:
                stats = {
                    'count': len(values),
                    'mean': statistics.mean(values),
                    'min': min(values),
                    'max': max(values),
                    'latest': values[-1]
                }
                
                # Calculate trend
                if len(values) >= 2:
                    first_half = values[:len(values)//2]
                    second_half = values[len(values)//2:]
                    
                    first_avg = statistics.mean(first_half)
                    second_avg = statistics.mean(second_half)
                    
                    if second_avg > first_avg * 1.1:
                        trends[name] = "increasing"
                    elif second_avg < first_avg * 0.9:
                        trends[name] = "decreasing" 
                    else:
                        trends[name] = "stable"
                else:
                    trends[name] = "insufficient_data"
                
                summary_stats[name] = stats
                
                # Generate recommendations
                if name in self.baseline_metrics:
                    baseline = self.baseline_metrics[name]
                    if stats['latest'] > baseline * 1.5:
                        recommendations.append(f"{name} significantly above baseline ({stats['latest']:.2f} vs {baseline:.2f})")
        
        return PerformanceReport(
            timespan_hours=hours,
            total_metrics=len(period_metrics),
            summary_stats=summary_stats,
            recommendations=recommendations,
            trends=trends
        )
    
    def clear_old_metrics(self, hours: float = 48.0):
        """Clear metrics older than specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        original_count = len(self.metrics)
        self.metrics = [
            m for m in self.metrics 
            if m.timestamp > cutoff_time
        ]
        
        removed_count = original_count - len(self.metrics)
        if removed_count > 0:
            logger.info(f"Cleared {removed_count} old metrics")
