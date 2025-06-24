#!/usr/bin/env python3
"""
A2A System Monitoring and Metrics

Production-grade monitoring system for A2A multi-agent infrastructure.
"""

import time
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

try:
    from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricValue:
    """Metric value with metadata"""
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """System alert"""
    level: AlertLevel
    message: str
    metric: str
    value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False


class A2AMetrics:
    """A2A System Metrics Collector"""
    
    def __init__(self, enable_prometheus: bool = True):
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.metrics_history: Dict[str, list] = {}
        self.alerts: list = []
        self.alert_thresholds = self._default_thresholds()
        
        if self.enable_prometheus:
            self._setup_prometheus_metrics()
    
    def _default_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Default alert thresholds"""
        return {
            "response_time": {"warning": 1.0, "error": 5.0, "critical": 10.0},
            "error_rate": {"warning": 0.05, "error": 0.10, "critical": 0.25},
            "task_failure_rate": {"warning": 0.10, "error": 0.20, "critical": 0.50},
            "memory_usage": {"warning": 0.80, "error": 0.90, "critical": 0.95},
            "cpu_usage": {"warning": 0.80, "error": 0.90, "critical": 0.95},
            "active_tasks": {"warning": 100, "error": 200, "critical": 500},
            "agent_availability": {"warning": 0.90, "error": 0.80, "critical": 0.70}
        }
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        # Counters
        self.request_count = Counter(
            'a2a_requests_total',
            'Total A2A requests',
            ['method', 'status', 'agent_type']
        )
        
        self.task_count = Counter(
            'a2a_tasks_total',
            'Total A2A tasks',
            ['task_type', 'status', 'agent_id']
        )
        
        self.error_count = Counter(
            'a2a_errors_total',
            'Total A2A errors',
            ['error_type', 'component']
        )
        
        # Histograms
        self.request_duration = Histogram(
            'a2a_request_duration_seconds',
            'A2A request duration',
            ['method', 'agent_type']
        )
        
        self.task_duration = Histogram(
            'a2a_task_duration_seconds',
            'A2A task duration',
            ['task_type', 'agent_id']
        )
        
        # Gauges
        self.active_tasks = Gauge(
            'a2a_active_tasks',
            'Number of active tasks',
            ['agent_id']
        )
        
        self.agent_availability = Gauge(
            'a2a_agent_availability',
            'Agent availability ratio',
            ['agent_id', 'agent_type']
        )
        
        self.system_health = Gauge(
            'a2a_system_health',
            'Overall system health score (0-1)'
        )
        
        self.memory_usage = Gauge(
            'a2a_memory_usage_bytes',
            'Memory usage in bytes',
            ['component']
        )
        
        self.cpu_usage = Gauge(
            'a2a_cpu_usage_percent',
            'CPU usage percentage',
            ['component']
        )
        
        # Info
        self.system_info = Info(
            'a2a_system_info',
            'A2A system information'
        )
        
        logger.info("Prometheus metrics initialized")
    
    def record_request(self, method: str, status: str, duration: float, agent_type: str = "unknown"):
        """Record API request metrics"""
        if self.enable_prometheus:
            self.request_count.labels(method=method, status=status, agent_type=agent_type).inc()
            self.request_duration.labels(method=method, agent_type=agent_type).observe(duration)
        
        # Store in history
        self._store_metric("request_duration", duration, {"method": method, "status": status})
        
        # Check thresholds
        self._check_threshold("response_time", duration)
    
    def record_task(self, task_type: str, status: str, duration: float, agent_id: str):
        """Record task execution metrics"""
        if self.enable_prometheus:
            self.task_count.labels(task_type=task_type, status=status, agent_id=agent_id).inc()
            self.task_duration.labels(task_type=task_type, agent_id=agent_id).observe(duration)
            
            if status == "started":
                self.active_tasks.labels(agent_id=agent_id).inc()
            elif status in ["completed", "failed", "canceled"]:
                self.active_tasks.labels(agent_id=agent_id).dec()
        
        # Store in history
        self._store_metric("task_duration", duration, {"task_type": task_type, "status": status})
        
        # Check thresholds
        if status == "failed":
            self._check_task_failure_rate()
    
    def record_error(self, error_type: str, component: str, details: str = ""):
        """Record error occurrence"""
        if self.enable_prometheus:
            self.error_count.labels(error_type=error_type, component=component).inc()
        
        # Store in history
        self._store_metric("error_count", 1, {"error_type": error_type, "component": component})
        
        # Create alert
        self._create_alert(AlertLevel.ERROR, f"{error_type} in {component}: {details}", "error_count", 1, 0)
        
        logger.error(f"Error recorded: {error_type} in {component} - {details}")
    
    def update_agent_availability(self, agent_id: str, agent_type: str, availability: float):
        """Update agent availability metrics"""
        if self.enable_prometheus:
            self.agent_availability.labels(agent_id=agent_id, agent_type=agent_type).set(availability)
        
        # Store in history
        self._store_metric("agent_availability", availability, {"agent_id": agent_id})
        
        # Check thresholds
        self._check_threshold("agent_availability", availability)
    
    def update_system_resources(self, memory_bytes: int, cpu_percent: float, component: str = "a2a_server"):
        """Update system resource metrics"""
        if self.enable_prometheus:
            self.memory_usage.labels(component=component).set(memory_bytes)
            self.cpu_usage.labels(component=component).set(cpu_percent)
        
        # Store in history
        memory_ratio = memory_bytes / (8 * 1024 * 1024 * 1024)  # Assume 8GB max
        self._store_metric("memory_usage", memory_ratio, {"component": component})
        self._store_metric("cpu_usage", cpu_percent / 100, {"component": component})
        
        # Check thresholds
        self._check_threshold("memory_usage", memory_ratio)
        self._check_threshold("cpu_usage", cpu_percent / 100)
    
    def update_system_health(self, health_score: float):
        """Update overall system health score"""
        if self.enable_prometheus:
            self.system_health.set(health_score)
        
        # Store in history
        self._store_metric("system_health", health_score)
        
        # Check critical health
        if health_score < 0.5:
            self._create_alert(
                AlertLevel.CRITICAL,
                f"System health critically low: {health_score:.2f}",
                "system_health",
                health_score,
                0.5
            )
    
    def _store_metric(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """Store metric in history"""
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = []
        
        metric_value = MetricValue(value=value, labels=labels or {})
        self.metrics_history[metric_name].append(metric_value)
        
        # Keep only last 1000 entries
        if len(self.metrics_history[metric_name]) > 1000:
            self.metrics_history[metric_name] = self.metrics_history[metric_name][-1000:]
    
    def _check_threshold(self, metric_name: str, value: float):
        """Check if metric exceeds thresholds"""
        if metric_name not in self.alert_thresholds:
            return
        
        thresholds = self.alert_thresholds[metric_name]
        
        for level_name, threshold in thresholds.items():
            if (metric_name in ["agent_availability"] and value < threshold) or \
               (metric_name not in ["agent_availability"] and value > threshold):
                
                level = AlertLevel(level_name)
                self._create_alert(
                    level,
                    f"{metric_name} threshold exceeded: {value:.3f} > {threshold}",
                    metric_name,
                    value,
                    threshold
                )
                break
    
    def _check_task_failure_rate(self):
        """Check task failure rate over recent period"""
        if "task_duration" not in self.metrics_history:
            return
        
        # Get tasks from last 5 minutes
        cutoff = datetime.utcnow() - timedelta(minutes=5)
        recent_tasks = [
            m for m in self.metrics_history["task_duration"]
            if m.timestamp > cutoff
        ]
        
        if len(recent_tasks) < 10:  # Need minimum sample size
            return
        
        failed_tasks = [
            m for m in recent_tasks
            if m.labels.get("status") == "failed"
        ]
        
        failure_rate = len(failed_tasks) / len(recent_tasks)
        self._check_threshold("task_failure_rate", failure_rate)
    
    def _create_alert(self, level: AlertLevel, message: str, metric: str, value: float, threshold: float):
        """Create system alert"""
        alert = Alert(
            level=level,
            message=message,
            metric=metric,
            value=value,
            threshold=threshold
        )
        
        self.alerts.append(alert)
        
        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        
        # Log alert
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }[level]
        
        logger.log(log_level, f"ALERT [{level.value.upper()}]: {message}")
    
    def get_recent_alerts(self, minutes: int = 60) -> list:
        """Get alerts from recent period"""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        return [alert for alert in self.alerts if alert.timestamp > cutoff and not alert.resolved]
    
    def get_metric_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        summary = {}
        
        for metric_name, values in self.metrics_history.items():
            if not values:
                continue
            
            recent_values = [v.value for v in values[-10:]]  # Last 10 values
            
            summary[metric_name] = {
                "current": recent_values[-1] if recent_values else 0,
                "average": sum(recent_values) / len(recent_values),
                "min": min(recent_values),
                "max": max(recent_values),
                "count": len(values)
            }
        
        return summary
    
    def start_prometheus_server(self, port: int = 9090):
        """Start Prometheus metrics server"""
        if not self.enable_prometheus:
            logger.warning("Prometheus not available, metrics server not started")
            return
        
        try:
            start_http_server(port)
            logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")


# Global metrics instance
metrics = A2AMetrics()


def record_request_metric(method: str, status: str, duration: float, agent_type: str = "unknown"):
    """Convenience function to record request metrics"""
    metrics.record_request(method, status, duration, agent_type)


def record_task_metric(task_type: str, status: str, duration: float, agent_id: str):
    """Convenience function to record task metrics"""
    metrics.record_task(task_type, status, duration, agent_id)


def record_error_metric(error_type: str, component: str, details: str = ""):
    """Convenience function to record error metrics"""
    metrics.record_error(error_type, component, details)
