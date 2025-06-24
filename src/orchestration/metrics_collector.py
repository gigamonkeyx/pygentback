"""
Metrics Collector

Comprehensive performance tracking and analysis system for orchestration.
Provides real-time monitoring, historical analysis, and performance optimization insights.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import statistics
from dataclasses import asdict

from .coordination_models import (
    PerformanceMetrics, OrchestrationConfig, AgentID, ServerID, TaskID,
    PerformanceCallback
)
from .agent_registry import AgentRegistry
from .mcp_orchestrator import MCPOrchestrator
from .task_dispatcher import TaskDispatcher

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Comprehensive metrics collection and analysis system.
    
    Features:
    - Real-time performance monitoring
    - Historical data analysis
    - Trend detection and alerting
    - Performance optimization insights
    - Customizable metrics and dashboards
    """
    
    def __init__(self, 
                 config: OrchestrationConfig,
                 agent_registry: Optional[AgentRegistry] = None,
                 mcp_orchestrator: Optional[MCPOrchestrator] = None,
                 task_dispatcher: Optional[TaskDispatcher] = None):
        self.config = config
        self.agent_registry = agent_registry
        self.mcp_orchestrator = mcp_orchestrator
        self.task_dispatcher = task_dispatcher
        
        # Metrics storage
        self.system_metrics: PerformanceMetrics = PerformanceMetrics()
        self.historical_metrics: deque = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        self.agent_metrics_history: Dict[AgentID, deque] = defaultdict(lambda: deque(maxlen=288))  # 24 hours at 5-minute intervals
        self.server_metrics_history: Dict[ServerID, deque] = defaultdict(lambda: deque(maxlen=288))
        
        # Custom metrics
        self.custom_metrics: Dict[str, Any] = {}
        self.metric_calculators: Dict[str, Callable] = {}
        
        # Alerting
        self.alert_conditions: Dict[str, Dict[str, Any]] = {}
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_history: deque = deque(maxlen=1000)
        
        # Performance tracking
        self.performance_trends: Dict[str, List[float]] = defaultdict(list)
        self.baseline_metrics: Dict[str, float] = {}
        
        # Monitoring state
        self.is_running = False
        self.collection_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self.performance_callbacks: List[PerformanceCallback] = []
        
        # Initialize default alert conditions
        self._setup_default_alerts()
        
        logger.info("Metrics Collector initialized")
    
    async def start(self):
        """Start metrics collection."""
        self.is_running = True
        
        # Start collection loop
        self.collection_task = asyncio.create_task(self._collection_loop())
        
        logger.info("Metrics Collector started")
    
    async def stop(self):
        """Stop metrics collection."""
        self.is_running = False
        
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Metrics Collector stopped")
    
    async def collect_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics."""
        try:
            metrics = PerformanceMetrics()
            
            # Collect agent metrics
            if self.agent_registry:
                agent_metrics = await self.agent_registry.get_registry_metrics()
                metrics.active_agents = agent_metrics['available_agents']
                metrics.avg_agent_utilization = agent_metrics['avg_agent_utilization']
                metrics.agent_performance_score = agent_metrics['avg_agent_performance']
            
            # Collect MCP server metrics
            if self.mcp_orchestrator:
                server_metrics = await self.mcp_orchestrator.get_orchestration_metrics()
                metrics.healthy_servers = server_metrics['healthy_servers']
                metrics.total_servers = server_metrics['total_servers']
                metrics.avg_response_time = server_metrics['avg_response_time']
            
            # Collect task dispatcher metrics
            if self.task_dispatcher:
                dispatcher_metrics = await self.task_dispatcher.get_dispatcher_metrics()
                metrics.total_tasks = dispatcher_metrics['total_tasks']
                metrics.completed_tasks = dispatcher_metrics['completed_tasks']
                metrics.failed_tasks = dispatcher_metrics['failed_tasks']
                metrics.task_success_rate = dispatcher_metrics['success_rate']
                metrics.avg_task_execution_time = dispatcher_metrics['avg_execution_time']
                metrics.avg_task_wait_time = dispatcher_metrics['avg_wait_time']
            
            # Calculate derived metrics
            metrics.coordination_efficiency = self._calculate_coordination_efficiency(metrics)
            
            # Update system metrics
            self.system_metrics = metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return PerformanceMetrics()
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics as dictionary."""
        metrics = await self.collect_metrics()
        return asdict(metrics)
    
    async def get_historical_metrics(self, 
                                   hours: int = 24,
                                   resolution: str = "minute") -> List[Dict[str, Any]]:
        """Get historical metrics data."""
        if resolution == "minute":
            data_points = min(hours * 60, len(self.historical_metrics))
            return [asdict(metrics) for metrics in list(self.historical_metrics)[-data_points:]]
        elif resolution == "hour":
            # Aggregate by hour
            hourly_data = []
            current_data = list(self.historical_metrics)
            
            for i in range(0, len(current_data), 60):
                hour_data = current_data[i:i+60]
                if hour_data:
                    aggregated = self._aggregate_metrics(hour_data)
                    hourly_data.append(asdict(aggregated))
            
            return hourly_data[-hours:]
        else:
            return []
    
    async def get_agent_metrics(self, agent_id: Optional[AgentID] = None) -> Dict[str, Any]:
        """Get metrics for specific agent or all agents."""
        if agent_id:
            if agent_id in self.agent_metrics_history:
                recent_metrics = list(self.agent_metrics_history[agent_id])[-12:]  # Last hour
                return {
                    "agent_id": agent_id,
                    "recent_metrics": [asdict(m) for m in recent_metrics],
                    "current_performance": recent_metrics[-1] if recent_metrics else None
                }
            else:
                return {"error": f"No metrics found for agent {agent_id}"}
        else:
            # Return metrics for all agents
            all_metrics = {}
            for aid in self.agent_metrics_history:
                all_metrics[aid] = await self.get_agent_metrics(aid)
            return all_metrics
    
    async def get_server_metrics(self, server_id: Optional[ServerID] = None) -> Dict[str, Any]:
        """Get metrics for specific server or all servers."""
        if server_id:
            if server_id in self.server_metrics_history:
                recent_metrics = list(self.server_metrics_history[server_id])[-12:]  # Last hour
                return {
                    "server_id": server_id,
                    "recent_metrics": [asdict(m) for m in recent_metrics],
                    "current_performance": recent_metrics[-1] if recent_metrics else None
                }
            else:
                return {"error": f"No metrics found for server {server_id}"}
        else:
            # Return metrics for all servers
            all_metrics = {}
            for sid in self.server_metrics_history:
                all_metrics[sid] = await self.get_server_metrics(sid)
            return all_metrics
    
    async def get_performance_trends(self, metric_name: str, hours: int = 24) -> Dict[str, Any]:
        """Get performance trends for a specific metric."""
        if metric_name not in self.performance_trends:
            return {"error": f"No trend data for metric {metric_name}"}
        
        trend_data = self.performance_trends[metric_name]
        data_points = min(hours * 60, len(trend_data))  # Assuming minute-level data
        recent_data = trend_data[-data_points:]
        
        if len(recent_data) < 2:
            return {"error": "Insufficient data for trend analysis"}
        
        # Calculate trend statistics
        trend_analysis = {
            "metric_name": metric_name,
            "data_points": len(recent_data),
            "current_value": recent_data[-1],
            "min_value": min(recent_data),
            "max_value": max(recent_data),
            "avg_value": statistics.mean(recent_data),
            "median_value": statistics.median(recent_data),
            "std_deviation": statistics.stdev(recent_data) if len(recent_data) > 1 else 0.0
        }
        
        # Calculate trend direction
        if len(recent_data) >= 10:
            recent_avg = statistics.mean(recent_data[-10:])
            earlier_avg = statistics.mean(recent_data[-20:-10]) if len(recent_data) >= 20 else statistics.mean(recent_data[:-10])
            
            if recent_avg > earlier_avg * 1.05:
                trend_analysis["trend"] = "increasing"
            elif recent_avg < earlier_avg * 0.95:
                trend_analysis["trend"] = "decreasing"
            else:
                trend_analysis["trend"] = "stable"
        else:
            trend_analysis["trend"] = "insufficient_data"
        
        return trend_analysis
    
    async def get_alerts(self, active_only: bool = True) -> Dict[str, Any]:
        """Get current alerts."""
        if active_only:
            return dict(self.active_alerts)
        else:
            return {
                "active_alerts": dict(self.active_alerts),
                "alert_history": list(self.alert_history)[-50:]  # Last 50 alerts
            }
    
    def add_custom_metric(self, name: str, calculator: Callable[[], float]):
        """Add a custom metric calculator."""
        self.metric_calculators[name] = calculator
        logger.info(f"Added custom metric: {name}")
    
    def remove_custom_metric(self, name: str):
        """Remove a custom metric."""
        if name in self.metric_calculators:
            del self.metric_calculators[name]
            if name in self.custom_metrics:
                del self.custom_metrics[name]
            logger.info(f"Removed custom metric: {name}")
    
    def add_alert_condition(self, 
                          name: str,
                          metric_name: str,
                          condition: str,
                          threshold: float,
                          severity: str = "warning"):
        """Add an alert condition."""
        self.alert_conditions[name] = {
            "metric_name": metric_name,
            "condition": condition,  # "greater_than", "less_than", "equals"
            "threshold": threshold,
            "severity": severity,
            "enabled": True
        }
        logger.info(f"Added alert condition: {name}")
    
    def remove_alert_condition(self, name: str):
        """Remove an alert condition."""
        if name in self.alert_conditions:
            del self.alert_conditions[name]
            if name in self.active_alerts:
                del self.active_alerts[name]
            logger.info(f"Removed alert condition: {name}")
    
    def add_performance_callback(self, callback: PerformanceCallback):
        """Add a callback for performance updates."""
        self.performance_callbacks.append(callback)
    
    def remove_performance_callback(self, callback: PerformanceCallback):
        """Remove a performance callback."""
        if callback in self.performance_callbacks:
            self.performance_callbacks.remove(callback)
    
    async def _collection_loop(self):
        """Main metrics collection loop."""
        while self.is_running:
            try:
                # Collect current metrics
                metrics = await self.collect_metrics()
                
                # Store historical data
                self.historical_metrics.append(metrics)
                
                # Collect custom metrics
                await self._collect_custom_metrics()
                
                # Update performance trends
                self._update_performance_trends(metrics)
                
                # Check alert conditions
                await self._check_alerts(metrics)
                
                # Notify callbacks
                await self._notify_performance_callbacks(metrics)
                
                # Collect component-specific metrics
                await self._collect_component_metrics()
                
                await asyncio.sleep(self.config.metrics_collection_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(10.0)
    
    async def _collect_custom_metrics(self):
        """Collect custom metrics."""
        for name, calculator in self.metric_calculators.items():
            try:
                if asyncio.iscoroutinefunction(calculator):
                    value = await calculator()
                else:
                    value = calculator()
                self.custom_metrics[name] = value
            except Exception as e:
                logger.error(f"Failed to calculate custom metric {name}: {e}")
    
    async def _collect_component_metrics(self):
        """Collect metrics from individual components."""
        try:
            # Collect agent-specific metrics
            if self.agent_registry:
                agent_status = await self.agent_registry.get_agent_status()
                for agent_id, status in agent_status.items():
                    if isinstance(status, dict) and 'metrics' in status:
                        self.agent_metrics_history[agent_id].append(status['metrics'])
            
            # Collect server-specific metrics
            if self.mcp_orchestrator:
                server_status = await self.mcp_orchestrator.get_server_status()
                for server_id, status in server_status.items():
                    if isinstance(status, dict) and 'metrics' in status:
                        self.server_metrics_history[server_id].append(status['metrics'])
        
        except Exception as e:
            logger.error(f"Failed to collect component metrics: {e}")
    
    def _update_performance_trends(self, metrics: PerformanceMetrics):
        """Update performance trend data."""
        # Track key metrics
        trend_metrics = {
            'task_success_rate': metrics.task_success_rate,
            'avg_response_time': metrics.avg_response_time,
            'agent_utilization': metrics.avg_agent_utilization,
            'coordination_efficiency': metrics.coordination_efficiency,
            'system_health': metrics.system_health_score
        }
        
        for metric_name, value in trend_metrics.items():
            self.performance_trends[metric_name].append(value)
            
            # Keep only recent data
            if len(self.performance_trends[metric_name]) > 1440:  # 24 hours
                self.performance_trends[metric_name] = self.performance_trends[metric_name][-1440:]
    
    async def _check_alerts(self, metrics: PerformanceMetrics):
        """Check alert conditions and trigger alerts."""
        current_values = {
            'task_success_rate': metrics.task_success_rate,
            'avg_response_time': metrics.avg_response_time,
            'agent_utilization': metrics.avg_agent_utilization,
            'server_utilization': metrics.avg_server_utilization,
            'coordination_efficiency': metrics.coordination_efficiency,
            'system_health': metrics.system_health_score
        }
        
        # Add custom metrics
        current_values.update(self.custom_metrics)
        
        for alert_name, condition in self.alert_conditions.items():
            if not condition.get('enabled', True):
                continue
            
            metric_name = condition['metric_name']
            if metric_name not in current_values:
                continue
            
            current_value = current_values[metric_name]
            threshold = condition['threshold']
            condition_type = condition['condition']
            
            # Check condition
            alert_triggered = False
            if condition_type == "greater_than" and current_value > threshold:
                alert_triggered = True
            elif condition_type == "less_than" and current_value < threshold:
                alert_triggered = True
            elif condition_type == "equals" and abs(current_value - threshold) < 0.001:
                alert_triggered = True
            
            # Handle alert
            if alert_triggered:
                if alert_name not in self.active_alerts:
                    await self._trigger_alert(alert_name, condition, current_value)
            else:
                if alert_name in self.active_alerts:
                    await self._resolve_alert(alert_name)
    
    async def _trigger_alert(self, alert_name: str, condition: Dict[str, Any], current_value: float):
        """Trigger an alert."""
        alert = {
            "name": alert_name,
            "metric_name": condition['metric_name'],
            "condition": condition['condition'],
            "threshold": condition['threshold'],
            "current_value": current_value,
            "severity": condition['severity'],
            "triggered_at": datetime.utcnow(),
            "status": "active"
        }
        
        self.active_alerts[alert_name] = alert
        self.alert_history.append(alert.copy())
        
        logger.warning(f"Alert triggered: {alert_name} - {condition['metric_name']} "
                      f"{condition['condition']} {condition['threshold']} (current: {current_value})")
    
    async def _resolve_alert(self, alert_name: str):
        """Resolve an active alert."""
        if alert_name in self.active_alerts:
            alert = self.active_alerts[alert_name]
            alert["status"] = "resolved"
            alert["resolved_at"] = datetime.utcnow()
            
            self.alert_history.append(alert.copy())
            del self.active_alerts[alert_name]
            
            logger.info(f"Alert resolved: {alert_name}")
    
    async def _notify_performance_callbacks(self, metrics: PerformanceMetrics):
        """Notify all performance callbacks."""
        for callback in self.performance_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(metrics)
                else:
                    callback(metrics)
            except Exception as e:
                logger.error(f"Performance callback error: {e}")
    
    def _calculate_coordination_efficiency(self, metrics: PerformanceMetrics) -> float:
        """Calculate coordination efficiency score."""
        # Combine multiple factors into efficiency score
        factors = []
        
        # Task success rate
        factors.append(metrics.task_success_rate)
        
        # Response time efficiency (inverse of response time)
        if metrics.avg_response_time > 0:
            factors.append(min(1.0, 1.0 / metrics.avg_response_time))
        else:
            factors.append(1.0)
        
        # Agent utilization (optimal around 0.7-0.8)
        optimal_utilization = 0.75
        utilization_efficiency = 1.0 - abs(metrics.avg_agent_utilization - optimal_utilization)
        factors.append(max(0.0, utilization_efficiency))
        
        # Server health
        if metrics.total_servers > 0:
            server_health = metrics.healthy_servers / metrics.total_servers
            factors.append(server_health)
        
        # Calculate weighted average
        if factors:
            return sum(factors) / len(factors)
        else:
            return 0.0
    
    def _aggregate_metrics(self, metrics_list: List[PerformanceMetrics]) -> PerformanceMetrics:
        """Aggregate a list of metrics into a single metrics object."""
        if not metrics_list:
            return PerformanceMetrics()
        
        aggregated = PerformanceMetrics()
        
        # Sum counters
        aggregated.total_tasks = sum(m.total_tasks for m in metrics_list)
        aggregated.completed_tasks = sum(m.completed_tasks for m in metrics_list)
        aggregated.failed_tasks = sum(m.failed_tasks for m in metrics_list)
        aggregated.cancelled_tasks = sum(m.cancelled_tasks for m in metrics_list)
        
        # Average timing metrics
        valid_metrics = [m for m in metrics_list if m.avg_task_execution_time > 0]
        if valid_metrics:
            aggregated.avg_task_execution_time = statistics.mean(m.avg_task_execution_time for m in valid_metrics)
            aggregated.avg_task_wait_time = statistics.mean(m.avg_task_wait_time for m in valid_metrics)
            aggregated.avg_response_time = statistics.mean(m.avg_response_time for m in valid_metrics)
        
        # Average utilization metrics
        aggregated.avg_agent_utilization = statistics.mean(m.avg_agent_utilization for m in metrics_list)
        aggregated.avg_server_utilization = statistics.mean(m.avg_server_utilization for m in metrics_list)
        
        # Average quality metrics
        aggregated.task_success_rate = statistics.mean(m.task_success_rate for m in metrics_list)
        aggregated.agent_performance_score = statistics.mean(m.agent_performance_score for m in metrics_list)
        aggregated.coordination_efficiency = statistics.mean(m.coordination_efficiency for m in metrics_list)
        
        # Use latest values for counts
        latest = metrics_list[-1]
        aggregated.active_agents = latest.active_agents
        aggregated.healthy_servers = latest.healthy_servers
        aggregated.total_servers = latest.total_servers
        aggregated.peak_concurrent_tasks = max(m.peak_concurrent_tasks for m in metrics_list)
        
        # Use latest timestamps
        aggregated.last_updated = latest.last_updated
        
        return aggregated
    
    def _setup_default_alerts(self):
        """Setup default alert conditions."""
        if self.config.performance_alerts_enabled:
            thresholds = self.config.alert_thresholds
            
            # Task failure rate alert
            self.add_alert_condition(
                "high_task_failure_rate",
                "task_success_rate",
                "less_than",
                1.0 - thresholds.get('task_failure_rate', 0.1),
                "warning"
            )
            
            # Response time alert
            self.add_alert_condition(
                "high_response_time",
                "avg_response_time",
                "greater_than",
                thresholds.get('avg_response_time', 5.0),
                "warning"
            )
            
            # Agent utilization alert
            self.add_alert_condition(
                "high_agent_utilization",
                "agent_utilization",
                "greater_than",
                thresholds.get('agent_utilization', 0.9),
                "warning"
            )
            
            # Server utilization alert
            self.add_alert_condition(
                "high_server_utilization",
                "server_utilization",
                "greater_than",
                thresholds.get('server_utilization', 0.9),
                "critical"
            )
    
    async def export_metrics(self, 
                           format: str = "json",
                           include_history: bool = True,
                           hours: int = 24) -> str:
        """Export metrics data in specified format."""
        try:
            export_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "current_metrics": await self.get_current_metrics(),
                "custom_metrics": dict(self.custom_metrics),
                "active_alerts": dict(self.active_alerts)
            }
            
            if include_history:
                export_data["historical_metrics"] = await self.get_historical_metrics(hours)
                export_data["performance_trends"] = {
                    name: trend[-hours*60:] for name, trend in self.performance_trends.items()
                }
            
            if format.lower() == "json":
                return json.dumps(export_data, indent=2, default=str)
            else:
                return str(export_data)
                
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return "{}"