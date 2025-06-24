"""
Integration Monitoring System

Comprehensive monitoring for integration components, system health,
performance metrics, and alerting capabilities.
"""

import logging
import asyncio
import time
import psutil
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    network_io_bytes: Dict[str, int]
    process_count: int
    load_average: float
    uptime_seconds: float


@dataclass
class ComponentHealth:
    """Health status of a system component"""
    component_id: str
    component_type: str
    status: HealthStatus
    last_check: datetime
    response_time_ms: float
    error_count: int
    success_rate: float
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """System alert"""
    alert_id: str
    severity: AlertSeverity
    component_id: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntegrationMonitor:
    """
    Integration System Monitor.
    
    Monitors the health and performance of integration components,
    tracks system metrics, and manages alerting.
    """
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Component tracking
        self.monitored_components: Dict[str, ComponentHealth] = {}
        self.component_checkers: Dict[str, Callable] = {}
        
        # Metrics storage
        self.system_metrics_history: List[SystemMetrics] = []
        self.component_metrics_history: Dict[str, List[ComponentHealth]] = {}
        
        # Alert system
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self.alert_handlers: List[Callable] = []
        
        # Thresholds
        self.health_thresholds = {
            'cpu_usage_warning': 80.0,
            'cpu_usage_critical': 95.0,
            'memory_usage_warning': 85.0,
            'memory_usage_critical': 95.0,
            'disk_usage_warning': 90.0,
            'disk_usage_critical': 98.0,
            'response_time_warning': 5000.0,  # 5 seconds
            'response_time_critical': 10000.0,  # 10 seconds
            'error_rate_warning': 0.1,  # 10%
            'error_rate_critical': 0.25  # 25%
        }
        
        # Statistics
        self.monitoring_stats = {
            'total_checks': 0,
            'failed_checks': 0,
            'alerts_generated': 0,
            'components_monitored': 0
        }
    
    async def start_monitoring(self):
        """Start the monitoring system"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Integration monitoring started")
    
    async def stop_monitoring(self):
        """Stop the monitoring system"""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
        logger.info("Integration monitoring stopped")
    
    def register_component(self, component_id: str, component_type: str, 
                          health_checker: Optional[Callable] = None):
        """Register a component for monitoring"""
        self.monitored_components[component_id] = ComponentHealth(
            component_id=component_id,
            component_type=component_type,
            status=HealthStatus.UNKNOWN,
            last_check=datetime.utcnow(),
            response_time_ms=0.0,
            error_count=0,
            success_rate=1.0
        )
        
        if health_checker:
            self.component_checkers[component_id] = health_checker
        
        self.monitoring_stats['components_monitored'] += 1
        logger.info(f"Registered component for monitoring: {component_id}")
    
    def unregister_component(self, component_id: str):
        """Unregister a component from monitoring"""
        if component_id in self.monitored_components:
            del self.monitored_components[component_id]
        if component_id in self.component_checkers:
            del self.component_checkers[component_id]
        if component_id in self.component_metrics_history:
            del self.component_metrics_history[component_id]
        
        self.monitoring_stats['components_monitored'] -= 1
        logger.info(f"Unregistered component from monitoring: {component_id}")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Check component health
                await self._check_component_health()
                
                # Process alerts
                await self._process_alerts()
                
                # Update statistics
                self.monitoring_stats['total_checks'] += 1
                
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                self.monitoring_stats['failed_checks'] += 1
                await asyncio.sleep(self.check_interval)
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # Get system metrics with Windows compatibility
            cpu_percent = psutil.cpu_percent(interval=0.1)  # Shorter interval to prevent hanging
            memory = psutil.virtual_memory()

            # Windows-compatible disk usage
            import os
            disk_path = 'C:\\' if os.name == 'nt' else '/'
            disk = psutil.disk_usage(disk_path)

            network = psutil.net_io_counters()
            
            metrics = SystemMetrics(
                timestamp=datetime.utcnow(),
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory.percent,
                disk_usage_percent=disk.percent,
                network_io_bytes={
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv
                },
                process_count=len(psutil.pids()),
                # Load average is Unix-specific, use CPU percent on Windows
                load_average=psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else cpu_percent / 100.0,
                uptime_seconds=time.time() - psutil.boot_time()
            )
            
            # Store metrics
            self.system_metrics_history.append(metrics)
            
            # Keep only recent metrics (last 24 hours)
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            self.system_metrics_history = [
                m for m in self.system_metrics_history 
                if m.timestamp > cutoff_time
            ]
            
            # Check for system-level alerts
            await self._check_system_alerts(metrics)
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    async def _check_component_health(self):
        """Check health of all monitored components"""
        for component_id, component in self.monitored_components.items():
            try:
                start_time = time.time()
                
                # Use custom checker if available
                if component_id in self.component_checkers:
                    checker = self.component_checkers[component_id]
                    health_result = await self._call_health_checker(checker)
                else:
                    # Default health check
                    health_result = await self._default_health_check(component_id)
                
                response_time = (time.time() - start_time) * 1000  # Convert to ms
                
                # Update component health
                component.last_check = datetime.utcnow()
                component.response_time_ms = response_time
                
                if health_result.get('healthy', True):
                    component.status = HealthStatus.HEALTHY
                    # Update success rate
                    component.success_rate = min(1.0, component.success_rate + 0.01)
                else:
                    component.status = HealthStatus.CRITICAL
                    component.error_count += 1
                    component.success_rate = max(0.0, component.success_rate - 0.05)
                
                # Store metrics
                if component_id not in self.component_metrics_history:
                    self.component_metrics_history[component_id] = []
                
                self.component_metrics_history[component_id].append(
                    ComponentHealth(
                        component_id=component_id,
                        component_type=component.component_type,
                        status=component.status,
                        last_check=component.last_check,
                        response_time_ms=response_time,
                        error_count=component.error_count,
                        success_rate=component.success_rate,
                        metrics=health_result.get('metrics', {})
                    )
                )
                
                # Keep only recent history
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                self.component_metrics_history[component_id] = [
                    h for h in self.component_metrics_history[component_id]
                    if h.last_check > cutoff_time
                ]
                
                # Check for component alerts
                await self._check_component_alerts(component)
                
            except Exception as e:
                logger.error(f"Health check failed for {component_id}: {e}")
                component.status = HealthStatus.UNKNOWN
                component.error_count += 1
    
    async def _call_health_checker(self, checker: Callable) -> Dict[str, Any]:
        """Call a custom health checker"""
        try:
            if asyncio.iscoroutinefunction(checker):
                return await checker()
            else:
                return checker()
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    async def _default_health_check(self, component_id: str) -> Dict[str, Any]:
        """Default health check implementation"""
        # Simple ping-style check
        return {'healthy': True, 'metrics': {'check_type': 'default'}}
    
    async def _check_system_alerts(self, metrics: SystemMetrics):
        """Check for system-level alerts"""
        alerts = []
        
        # CPU usage alerts
        if metrics.cpu_usage_percent > self.health_thresholds['cpu_usage_critical']:
            alerts.append(self._create_alert(
                'system', AlertSeverity.CRITICAL,
                f"Critical CPU usage: {metrics.cpu_usage_percent:.1f}%"
            ))
        elif metrics.cpu_usage_percent > self.health_thresholds['cpu_usage_warning']:
            alerts.append(self._create_alert(
                'system', AlertSeverity.WARNING,
                f"High CPU usage: {metrics.cpu_usage_percent:.1f}%"
            ))
        
        # Memory usage alerts
        if metrics.memory_usage_percent > self.health_thresholds['memory_usage_critical']:
            alerts.append(self._create_alert(
                'system', AlertSeverity.CRITICAL,
                f"Critical memory usage: {metrics.memory_usage_percent:.1f}%"
            ))
        elif metrics.memory_usage_percent > self.health_thresholds['memory_usage_warning']:
            alerts.append(self._create_alert(
                'system', AlertSeverity.WARNING,
                f"High memory usage: {metrics.memory_usage_percent:.1f}%"
            ))
        
        # Disk usage alerts
        if metrics.disk_usage_percent > self.health_thresholds['disk_usage_critical']:
            alerts.append(self._create_alert(
                'system', AlertSeverity.CRITICAL,
                f"Critical disk usage: {metrics.disk_usage_percent:.1f}%"
            ))
        elif metrics.disk_usage_percent > self.health_thresholds['disk_usage_warning']:
            alerts.append(self._create_alert(
                'system', AlertSeverity.WARNING,
                f"High disk usage: {metrics.disk_usage_percent:.1f}%"
            ))
        
        # Add alerts
        for alert in alerts:
            await self._add_alert(alert)
    
    async def _check_component_alerts(self, component: ComponentHealth):
        """Check for component-specific alerts"""
        alerts = []
        
        # Response time alerts
        if component.response_time_ms > self.health_thresholds['response_time_critical']:
            alerts.append(self._create_alert(
                component.component_id, AlertSeverity.CRITICAL,
                f"Critical response time: {component.response_time_ms:.0f}ms"
            ))
        elif component.response_time_ms > self.health_thresholds['response_time_warning']:
            alerts.append(self._create_alert(
                component.component_id, AlertSeverity.WARNING,
                f"High response time: {component.response_time_ms:.0f}ms"
            ))
        
        # Error rate alerts
        error_rate = 1.0 - component.success_rate
        if error_rate > self.health_thresholds['error_rate_critical']:
            alerts.append(self._create_alert(
                component.component_id, AlertSeverity.CRITICAL,
                f"Critical error rate: {error_rate:.1%}"
            ))
        elif error_rate > self.health_thresholds['error_rate_warning']:
            alerts.append(self._create_alert(
                component.component_id, AlertSeverity.WARNING,
                f"High error rate: {error_rate:.1%}"
            ))
        
        # Component status alerts
        if component.status == HealthStatus.CRITICAL:
            alerts.append(self._create_alert(
                component.component_id, AlertSeverity.ERROR,
                f"Component health critical"
            ))
        elif component.status == HealthStatus.WARNING:
            alerts.append(self._create_alert(
                component.component_id, AlertSeverity.WARNING,
                f"Component health warning"
            ))
        
        # Add alerts
        for alert in alerts:
            await self._add_alert(alert)
    
    def _create_alert(self, component_id: str, severity: AlertSeverity, 
                     message: str) -> Alert:
        """Create a new alert"""
        alert_id = f"alert_{len(self.alert_history) + 1}_{int(time.time())}"
        return Alert(
            alert_id=alert_id,
            severity=severity,
            component_id=component_id,
            message=message,
            timestamp=datetime.utcnow()
        )
    
    async def _add_alert(self, alert: Alert):
        """Add an alert to the system"""
        # Check if similar alert already exists
        existing_alert = self._find_similar_alert(alert)
        if existing_alert:
            return  # Don't duplicate alerts
        
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        self.monitoring_stats['alerts_generated'] += 1
        
        # Trigger alert handlers
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
        
        logger.warning(f"Alert generated: {alert.severity.value} - {alert.message}")
    
    def _find_similar_alert(self, alert: Alert) -> Optional[Alert]:
        """Find similar active alert"""
        for active_alert in self.active_alerts:
            if (active_alert.component_id == alert.component_id and
                active_alert.severity == alert.severity and
                active_alert.message == alert.message and
                not active_alert.resolved):
                return active_alert
        return None
    
    async def _process_alerts(self):
        """Process and manage alerts"""
        # Auto-resolve old alerts
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        for alert in self.active_alerts:
            if alert.timestamp < cutoff_time and not alert.resolved:
                alert.resolved = True
                alert.resolution_time = datetime.utcnow()
        
        # Remove resolved alerts from active list
        self.active_alerts = [a for a in self.active_alerts if not a.resolved]
    
    def add_alert_handler(self, handler: Callable):
        """Add an alert handler"""
        self.alert_handlers.append(handler)
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        if not self.system_metrics_history:
            return {"status": "no_data"}
        
        latest_metrics = self.system_metrics_history[-1]
        
        # Determine overall health
        overall_status = HealthStatus.HEALTHY
        if (latest_metrics.cpu_usage_percent > self.health_thresholds['cpu_usage_critical'] or
            latest_metrics.memory_usage_percent > self.health_thresholds['memory_usage_critical'] or
            latest_metrics.disk_usage_percent > self.health_thresholds['disk_usage_critical']):
            overall_status = HealthStatus.CRITICAL
        elif (latest_metrics.cpu_usage_percent > self.health_thresholds['cpu_usage_warning'] or
              latest_metrics.memory_usage_percent > self.health_thresholds['memory_usage_warning'] or
              latest_metrics.disk_usage_percent > self.health_thresholds['disk_usage_warning']):
            overall_status = HealthStatus.WARNING
        
        return {
            "overall_status": overall_status.value,
            "system_metrics": {
                "cpu_usage": latest_metrics.cpu_usage_percent,
                "memory_usage": latest_metrics.memory_usage_percent,
                "disk_usage": latest_metrics.disk_usage_percent,
                "process_count": latest_metrics.process_count,
                "uptime_hours": latest_metrics.uptime_seconds / 3600
            },
            "component_health": {
                comp_id: comp.status.value 
                for comp_id, comp in self.monitored_components.items()
            },
            "active_alerts": len(self.active_alerts),
            "monitoring_stats": self.monitoring_stats
        }
    
    def get_component_metrics(self, component_id: str, 
                            hours: int = 24) -> List[ComponentHealth]:
        """Get metrics history for a component"""
        if component_id not in self.component_metrics_history:
            return []
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            metric for metric in self.component_metrics_history[component_id]
            if metric.last_check > cutoff_time
        ]


class SystemHealthMonitor:
    """
    System Health Monitor.
    
    Specialized monitor for overall system health and performance.
    """
    
    def __init__(self):
        self.integration_monitor = IntegrationMonitor()
        self.health_checks = {}
    
    async def start(self):
        """Start system health monitoring"""
        await self.integration_monitor.start_monitoring()
    
    async def stop(self):
        """Stop system health monitoring"""
        await self.integration_monitor.stop_monitoring()
    
    def register_health_check(self, component_id: str, health_check: Callable):
        """Register a health check for a component"""
        self.integration_monitor.register_component(
            component_id, "system_component", health_check
        )


class PerformanceMonitor:
    """
    Performance Monitor.
    
    Specialized monitor for performance metrics and optimization.
    """
    
    def __init__(self):
        self.performance_data = {}
        self.benchmarks = {}
    
    def record_performance(self, operation: str, duration: float, metadata: Dict[str, Any] = None):
        """Record performance data for an operation"""
        if operation not in self.performance_data:
            self.performance_data[operation] = []
        
        self.performance_data[operation].append({
            'timestamp': datetime.utcnow(),
            'duration': duration,
            'metadata': metadata or {}
        })
    
    def get_performance_summary(self, operation: str, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for an operation"""
        if operation not in self.performance_data:
            return {}
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_data = [
            d for d in self.performance_data[operation]
            if d['timestamp'] > cutoff_time
        ]
        
        if not recent_data:
            return {}
        
        durations = [d['duration'] for d in recent_data]
        
        return {
            'operation': operation,
            'sample_count': len(durations),
            'avg_duration': statistics.mean(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'median_duration': statistics.median(durations),
            'std_deviation': statistics.stdev(durations) if len(durations) > 1 else 0.0
        }
