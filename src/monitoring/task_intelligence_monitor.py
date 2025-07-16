"""
Task Intelligence System Monitoring & Analytics

Real-time monitoring, analytics, and alerting for the Task Intelligence System.
Provides comprehensive insights into system performance, pattern effectiveness,
and optimization opportunities.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class SystemAlert:
    """System alert"""
    alert_id: str
    level: AlertLevel
    title: str
    description: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskIntelligenceMonitor:
    """
    Comprehensive monitoring and analytics system for Task Intelligence
    """
    
    def __init__(self, task_intelligence_integration):
        self.integration = task_intelligence_integration
        self.logger = logging.getLogger(f"{__name__}.TaskIntelligenceMonitor")
        
        # Monitoring state
        self.metrics_history: Dict[str, List[PerformanceMetric]] = {}
        self.active_alerts: Dict[str, SystemAlert] = {}
        self.alert_history: List[SystemAlert] = []
        
        # Performance thresholds
        self.thresholds = {
            "success_rate_critical": 0.5,
            "success_rate_warning": 0.7,
            "avg_execution_time_warning": 3600,  # 1 hour
            "avg_execution_time_critical": 7200,  # 2 hours
            "pattern_effectiveness_warning": 0.6,
            "stall_rate_warning": 0.2,
            "stall_rate_critical": 0.4
        }
        
        # Analytics configuration
        self.metrics_retention_hours = 168  # 1 week
        self.alert_retention_hours = 720    # 30 days
        self.monitoring_interval = 60       # 1 minute
        
        # Real-time dashboard data
        self.dashboard_data = {
            "current_metrics": {},
            "trend_data": {},
            "system_health": "unknown",
            "active_tasks": 0,
            "recent_alerts": []
        }
    
    async def start_monitoring(self):
        """Start the monitoring system"""
        try:
            self.logger.info("Starting Task Intelligence monitoring system...")
            
            # Start monitoring loops
            asyncio.create_task(self._metrics_collection_loop())
            asyncio.create_task(self._alerting_loop())
            asyncio.create_task(self._analytics_loop())
            asyncio.create_task(self._cleanup_loop())
            
            self.logger.info("Task Intelligence monitoring system started")
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring system: {e}")
            raise
    
    async def _metrics_collection_loop(self):
        """Main metrics collection loop"""
        
        while True:
            try:
                await asyncio.sleep(self.monitoring_interval)
                
                # Collect current metrics
                await self._collect_performance_metrics()
                await self._collect_system_metrics()
                await self._collect_pattern_metrics()
                await self._collect_integration_metrics()
                
                # Update dashboard data
                await self._update_dashboard_data()
                
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
    
    async def _collect_performance_metrics(self):
        """Collect performance metrics"""
        
        try:
            integration_status = self.integration.get_integration_status()
            metrics = integration_status["integration_metrics"]
            
            timestamp = datetime.utcnow()
            
            # Success rate metric
            self._record_metric(PerformanceMetric(
                name="task_success_rate",
                value=metrics["intelligence_success_rate"],
                timestamp=timestamp,
                unit="percentage"
            ))
            
            # Average improvement metric
            self._record_metric(PerformanceMetric(
                name="average_improvement",
                value=metrics["average_improvement"],
                timestamp=timestamp,
                unit="score"
            ))
            
            # Tasks processed metric
            self._record_metric(PerformanceMetric(
                name="tasks_processed_total",
                value=metrics["tasks_processed"],
                timestamp=timestamp,
                unit="count"
            ))
            
            # Pattern applications metric
            self._record_metric(PerformanceMetric(
                name="pattern_applications",
                value=metrics["pattern_applications"],
                timestamp=timestamp,
                unit="count"
            ))
            
            # Active tasks metric
            active_tasks = integration_status["active_intelligent_tasks"]
            self._record_metric(PerformanceMetric(
                name="active_tasks",
                value=active_tasks,
                timestamp=timestamp,
                unit="count"
            ))
            
        except Exception as e:
            self.logger.error(f"Failed to collect performance metrics: {e}")
    
    async def _collect_system_metrics(self):
        """Collect system health metrics"""
        
        try:
            # Task Intelligence System metrics
            task_intelligence = self.integration.task_intelligence
            
            timestamp = datetime.utcnow()
            
            # Pattern library size
            self._record_metric(PerformanceMetric(
                name="pattern_library_size",
                value=len(task_intelligence.workflow_patterns),
                timestamp=timestamp,
                unit="count"
            ))
            
            # Failure patterns count
            self._record_metric(PerformanceMetric(
                name="failure_patterns_count",
                value=len(task_intelligence.failure_patterns),
                timestamp=timestamp,
                unit="count"
            ))
            
            # Question generation stats
            question_stats = task_intelligence.get_question_generation_stats()
            
            self._record_metric(PerformanceMetric(
                name="questions_generated_total",
                value=question_stats["total_questions_generated"],
                timestamp=timestamp,
                unit="count"
            ))
            
            self._record_metric(PerformanceMetric(
                name="avg_questions_per_task",
                value=question_stats["average_questions_per_task"],
                timestamp=timestamp,
                unit="count"
            ))
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
    
    async def _collect_pattern_metrics(self):
        """Collect pattern learning metrics"""
        
        try:
            task_intelligence = self.integration.task_intelligence
            pattern_analytics = task_intelligence.get_pattern_analytics()
            
            timestamp = datetime.utcnow()
            
            # Pattern effectiveness
            workflow_patterns = pattern_analytics["workflow_patterns"]
            self._record_metric(PerformanceMetric(
                name="pattern_effectiveness_avg",
                value=workflow_patterns["average_effectiveness"],
                timestamp=timestamp,
                unit="score"
            ))
            
            # Pattern reuse rate
            learning_effectiveness = pattern_analytics["learning_effectiveness"]
            self._record_metric(PerformanceMetric(
                name="pattern_reuse_rate",
                value=learning_effectiveness["pattern_reuse_rate"],
                timestamp=timestamp,
                unit="rate"
            ))
            
            # Average pattern success rate
            self._record_metric(PerformanceMetric(
                name="pattern_success_rate_avg",
                value=learning_effectiveness["average_pattern_success_rate"],
                timestamp=timestamp,
                unit="percentage"
            ))
            
        except Exception as e:
            self.logger.error(f"Failed to collect pattern metrics: {e}")
    
    async def _collect_integration_metrics(self):
        """Collect integration-specific metrics"""
        
        try:
            # Meta-supervisor performance
            meta_performance = self.integration.meta_supervisor.get_supervisor_performance()
            
            timestamp = datetime.utcnow()
            
            if meta_performance:
                # Calculate average success rate across supervisors
                success_rates = [perf["success_rate"] for perf in meta_performance.values()]
                if success_rates:
                    avg_success_rate = sum(success_rates) / len(success_rates)
                    self._record_metric(PerformanceMetric(
                        name="meta_supervisor_success_rate",
                        value=avg_success_rate,
                        timestamp=timestamp,
                        unit="percentage"
                    ))
                
                # Active supervisors count
                self._record_metric(PerformanceMetric(
                    name="active_supervisors",
                    value=len(meta_performance),
                    timestamp=timestamp,
                    unit="count"
                ))
            
        except Exception as e:
            self.logger.error(f"Failed to collect integration metrics: {e}")
    
    def _record_metric(self, metric: PerformanceMetric):
        """Record a performance metric"""
        
        if metric.name not in self.metrics_history:
            self.metrics_history[metric.name] = []
        
        self.metrics_history[metric.name].append(metric)
        
        # Cleanup old metrics
        cutoff_time = datetime.utcnow() - timedelta(hours=self.metrics_retention_hours)
        self.metrics_history[metric.name] = [
            m for m in self.metrics_history[metric.name] 
            if m.timestamp > cutoff_time
        ]
    
    async def _alerting_loop(self):
        """Alerting and threshold monitoring loop"""
        
        while True:
            try:
                await asyncio.sleep(self.monitoring_interval)
                
                # Check performance thresholds
                await self._check_success_rate_thresholds()
                await self._check_execution_time_thresholds()
                await self._check_pattern_effectiveness_thresholds()
                await self._check_system_health_thresholds()
                
                # Auto-resolve alerts
                await self._auto_resolve_alerts()
                
            except Exception as e:
                self.logger.error(f"Alerting loop error: {e}")
    
    async def _check_success_rate_thresholds(self):
        """Check success rate thresholds"""
        
        try:
            if "task_success_rate" in self.metrics_history:
                recent_metrics = self._get_recent_metrics("task_success_rate", minutes=10)
                
                if recent_metrics:
                    avg_success_rate = sum(m.value for m in recent_metrics) / len(recent_metrics)
                    
                    if avg_success_rate < self.thresholds["success_rate_critical"]:
                        await self._create_alert(
                            AlertLevel.CRITICAL,
                            "Critical Success Rate",
                            f"Task success rate is critically low: {avg_success_rate:.2%}",
                            {"success_rate": avg_success_rate, "threshold": self.thresholds["success_rate_critical"]}
                        )
                    elif avg_success_rate < self.thresholds["success_rate_warning"]:
                        await self._create_alert(
                            AlertLevel.WARNING,
                            "Low Success Rate",
                            f"Task success rate is below warning threshold: {avg_success_rate:.2%}",
                            {"success_rate": avg_success_rate, "threshold": self.thresholds["success_rate_warning"]}
                        )
            
        except Exception as e:
            self.logger.error(f"Failed to check success rate thresholds: {e}")
    
    async def _check_execution_time_thresholds(self):
        """Check execution time thresholds"""
        
        try:
            # Calculate average execution time from recent task completions
            recent_completions = []
            
            for task_id, tracking in self.integration.task_performance_tracking.items():
                if tracking.get("completion_time") and tracking.get("success"):
                    completion_time = tracking["completion_time"]
                    recent_completions.append(completion_time)
            
            if len(recent_completions) >= 5:  # Need at least 5 samples
                avg_execution_time = sum(recent_completions) / len(recent_completions)
                
                if avg_execution_time > self.thresholds["avg_execution_time_critical"]:
                    await self._create_alert(
                        AlertLevel.CRITICAL,
                        "Critical Execution Time",
                        f"Average execution time is critically high: {avg_execution_time:.0f}s",
                        {"avg_execution_time": avg_execution_time, "threshold": self.thresholds["avg_execution_time_critical"]}
                    )
                elif avg_execution_time > self.thresholds["avg_execution_time_warning"]:
                    await self._create_alert(
                        AlertLevel.WARNING,
                        "High Execution Time",
                        f"Average execution time is above warning threshold: {avg_execution_time:.0f}s",
                        {"avg_execution_time": avg_execution_time, "threshold": self.thresholds["avg_execution_time_warning"]}
                    )
            
        except Exception as e:
            self.logger.error(f"Failed to check execution time thresholds: {e}")
    
    async def _check_pattern_effectiveness_thresholds(self):
        """Check pattern effectiveness thresholds"""
        
        try:
            if "pattern_effectiveness_avg" in self.metrics_history:
                recent_metrics = self._get_recent_metrics("pattern_effectiveness_avg", minutes=30)
                
                if recent_metrics:
                    latest_effectiveness = recent_metrics[-1].value
                    
                    if latest_effectiveness < self.thresholds["pattern_effectiveness_warning"]:
                        await self._create_alert(
                            AlertLevel.WARNING,
                            "Low Pattern Effectiveness",
                            f"Pattern effectiveness is below threshold: {latest_effectiveness:.2f}",
                            {"effectiveness": latest_effectiveness, "threshold": self.thresholds["pattern_effectiveness_warning"]}
                        )
            
        except Exception as e:
            self.logger.error(f"Failed to check pattern effectiveness thresholds: {e}")
    
    async def _check_system_health_thresholds(self):
        """Check overall system health"""
        
        try:
            # Check for system anomalies
            active_tasks = self._get_latest_metric_value("active_tasks")
            
            # Alert if too many tasks are stuck
            if active_tasks and active_tasks > 20:
                await self._create_alert(
                    AlertLevel.WARNING,
                    "High Active Task Count",
                    f"Unusually high number of active tasks: {active_tasks}",
                    {"active_tasks": active_tasks}
                )
            
            # Check pattern library growth
            pattern_count = self._get_latest_metric_value("pattern_library_size")
            failure_count = self._get_latest_metric_value("failure_patterns_count")
            
            if pattern_count and failure_count and failure_count > pattern_count * 0.5:
                await self._create_alert(
                    AlertLevel.WARNING,
                    "High Failure Pattern Ratio",
                    f"Failure patterns ({failure_count}) are high relative to success patterns ({pattern_count})",
                    {"failure_patterns": failure_count, "success_patterns": pattern_count}
                )
            
        except Exception as e:
            self.logger.error(f"Failed to check system health thresholds: {e}")
    
    def _get_recent_metrics(self, metric_name: str, minutes: int = 10) -> List[PerformanceMetric]:
        """Get recent metrics for a given metric name"""
        
        if metric_name not in self.metrics_history:
            return []
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        return [m for m in self.metrics_history[metric_name] if m.timestamp > cutoff_time]
    
    def _get_latest_metric_value(self, metric_name: str) -> Optional[float]:
        """Get the latest value for a metric"""
        
        if metric_name not in self.metrics_history or not self.metrics_history[metric_name]:
            return None
        
        return self.metrics_history[metric_name][-1].value
    
    async def _create_alert(self, level: AlertLevel, title: str, description: str, metadata: Dict[str, Any]):
        """Create a new alert"""
        
        alert_id = f"{level.value}_{title.replace(' ', '_').lower()}_{int(datetime.utcnow().timestamp())}"
        
        # Check if similar alert already exists
        for existing_alert in self.active_alerts.values():
            if existing_alert.title == title and not existing_alert.resolved:
                return  # Don't create duplicate alerts
        
        alert = SystemAlert(
            alert_id=alert_id,
            level=level,
            title=title,
            description=description,
            timestamp=datetime.utcnow(),
            metadata=metadata
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Log alert
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }[level]
        
        self.logger.log(log_level, f"ALERT [{level.value.upper()}] {title}: {description}")
    
    async def _auto_resolve_alerts(self):
        """Auto-resolve alerts when conditions improve"""
        
        try:
            for alert_id, alert in list(self.active_alerts.items()):
                if alert.resolved:
                    continue
                
                should_resolve = False
                
                # Auto-resolve success rate alerts
                if "Success Rate" in alert.title:
                    current_success_rate = self._get_latest_metric_value("task_success_rate")
                    if current_success_rate and current_success_rate > self.thresholds["success_rate_warning"]:
                        should_resolve = True
                
                # Auto-resolve execution time alerts
                elif "Execution Time" in alert.title:
                    # Check if recent execution times have improved
                    recent_completions = []
                    for tracking in self.integration.task_performance_tracking.values():
                        if tracking.get("completion_time") and tracking.get("success"):
                            recent_completions.append(tracking["completion_time"])
                    
                    if len(recent_completions) >= 3:
                        avg_time = sum(recent_completions[-3:]) / 3  # Last 3 completions
                        if avg_time < self.thresholds["avg_execution_time_warning"]:
                            should_resolve = True
                
                if should_resolve:
                    alert.resolved = True
                    alert.resolution_time = datetime.utcnow()
                    del self.active_alerts[alert_id]
                    self.logger.info(f"Auto-resolved alert: {alert.title}")
        
        except Exception as e:
            self.logger.error(f"Failed to auto-resolve alerts: {e}")
    
    async def _update_dashboard_data(self):
        """Update real-time dashboard data"""
        
        try:
            # Current metrics
            self.dashboard_data["current_metrics"] = {
                "success_rate": self._get_latest_metric_value("task_success_rate") or 0,
                "active_tasks": self._get_latest_metric_value("active_tasks") or 0,
                "pattern_effectiveness": self._get_latest_metric_value("pattern_effectiveness_avg") or 0,
                "tasks_processed": self._get_latest_metric_value("tasks_processed_total") or 0
            }
            
            # System health
            success_rate = self.dashboard_data["current_metrics"]["success_rate"]
            if success_rate >= 0.8:
                self.dashboard_data["system_health"] = "healthy"
            elif success_rate >= 0.6:
                self.dashboard_data["system_health"] = "warning"
            else:
                self.dashboard_data["system_health"] = "critical"
            
            # Recent alerts
            self.dashboard_data["recent_alerts"] = [
                {
                    "level": alert.level.value,
                    "title": alert.title,
                    "timestamp": alert.timestamp.isoformat(),
                    "resolved": alert.resolved
                }
                for alert in sorted(self.alert_history[-10:], key=lambda a: a.timestamp, reverse=True)
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to update dashboard data: {e}")
    
    async def _analytics_loop(self):
        """Analytics and reporting loop"""
        
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Generate trend analysis
                await self._analyze_trends()
                
                # Generate performance insights
                await self._generate_performance_insights()
                
            except Exception as e:
                self.logger.error(f"Analytics loop error: {e}")
    
    async def _analyze_trends(self):
        """Analyze performance trends"""
        
        try:
            # Analyze success rate trend
            success_rate_metrics = self._get_recent_metrics("task_success_rate", minutes=60)
            if len(success_rate_metrics) >= 10:
                recent_values = [m.value for m in success_rate_metrics[-10:]]
                trend = "improving" if recent_values[-1] > recent_values[0] else "declining"
                
                self.dashboard_data["trend_data"]["success_rate_trend"] = trend
            
            # Analyze pattern effectiveness trend
            pattern_metrics = self._get_recent_metrics("pattern_effectiveness_avg", minutes=60)
            if len(pattern_metrics) >= 5:
                recent_values = [m.value for m in pattern_metrics[-5:]]
                trend = "improving" if recent_values[-1] > recent_values[0] else "declining"
                
                self.dashboard_data["trend_data"]["pattern_effectiveness_trend"] = trend
            
        except Exception as e:
            self.logger.error(f"Failed to analyze trends: {e}")
    
    async def _generate_performance_insights(self):
        """Generate performance insights and recommendations"""
        
        try:
            insights = []
            
            # Pattern library insights
            pattern_count = self._get_latest_metric_value("pattern_library_size")
            reuse_rate = self._get_latest_metric_value("pattern_reuse_rate")
            
            if pattern_count and reuse_rate:
                if reuse_rate < 0.3:
                    insights.append("Low pattern reuse rate suggests need for better pattern matching")
                elif reuse_rate > 0.7:
                    insights.append("High pattern reuse rate indicates effective pattern learning")
            
            # Success rate insights
            success_rate = self._get_latest_metric_value("task_success_rate")
            if success_rate:
                if success_rate > 0.9:
                    insights.append("Excellent task success rate - system performing optimally")
                elif success_rate < 0.7:
                    insights.append("Low success rate - consider reviewing task complexity thresholds")
            
            self.dashboard_data["insights"] = insights
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance insights: {e}")
    
    async def _cleanup_loop(self):
        """Cleanup old data"""
        
        while True:
            try:
                await asyncio.sleep(3600)  # Every hour
                
                # Cleanup old alerts
                cutoff_time = datetime.utcnow() - timedelta(hours=self.alert_retention_hours)
                self.alert_history = [
                    alert for alert in self.alert_history 
                    if alert.timestamp > cutoff_time
                ]
                
                # Cleanup old performance tracking
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                for task_id in list(self.integration.task_performance_tracking.keys()):
                    tracking = self.integration.task_performance_tracking[task_id]
                    if tracking.get("start_time") and tracking["start_time"] < cutoff_time:
                        del self.integration.task_performance_tracking[task_id]
                
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        return self.dashboard_data.copy()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        
        summary = {}
        
        for metric_name, metrics in self.metrics_history.items():
            if metrics:
                latest = metrics[-1]
                summary[metric_name] = {
                    "current_value": latest.value,
                    "unit": latest.unit,
                    "last_updated": latest.timestamp.isoformat(),
                    "data_points": len(metrics)
                }
        
        return summary
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts"""
        
        return [
            {
                "alert_id": alert.alert_id,
                "level": alert.level.value,
                "title": alert.title,
                "description": alert.description,
                "timestamp": alert.timestamp.isoformat(),
                "metadata": alert.metadata
            }
            for alert in self.active_alerts.values()
        ]
