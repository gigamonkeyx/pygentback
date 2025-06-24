"""
Server Monitor

Real-time monitoring and health assessment of MCP servers.
Tracks performance, availability, and resource usage for intelligent routing.
"""

import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    OFFLINE = "offline"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetrics:
    """Performance metrics for a server"""
    # Response time metrics
    avg_response_time_ms: float = 0.0
    min_response_time_ms: float = 0.0
    max_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    
    # Throughput metrics
    requests_per_second: float = 0.0
    successful_requests: int = 0
    failed_requests: int = 0
    total_requests: int = 0
    
    # Error metrics
    error_rate: float = 0.0
    timeout_rate: float = 0.0
    
    # Resource metrics
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    memory_usage_percent: float = 0.0
    disk_usage_mb: float = 0.0
    network_io_mbps: float = 0.0
    
    # Availability metrics
    uptime_seconds: float = 0.0
    uptime_percentage: float = 0.0
    last_seen: datetime = field(default_factory=datetime.utcnow)
    
    def calculate_success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'avg_response_time_ms': self.avg_response_time_ms,
            'p95_response_time_ms': self.p95_response_time_ms,
            'p99_response_time_ms': self.p99_response_time_ms,
            'requests_per_second': self.requests_per_second,
            'success_rate': self.calculate_success_rate(),
            'error_rate': self.error_rate,
            'cpu_usage_percent': self.cpu_usage_percent,
            'memory_usage_mb': self.memory_usage_mb,
            'uptime_percentage': self.uptime_percentage,
            'last_seen': self.last_seen.isoformat()
        }


@dataclass
class ServerHealth:
    """Health assessment for a server"""
    server_name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    overall_score: float = 0.0
    
    # Component health scores
    performance_score: float = 0.0
    availability_score: float = 0.0
    reliability_score: float = 0.0
    resource_score: float = 0.0
    
    # Health indicators
    response_time_health: float = 0.0
    error_rate_health: float = 0.0
    resource_health: float = 0.0
    uptime_health: float = 0.0
    
    # Issues and recommendations
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    last_assessment: datetime = field(default_factory=datetime.utcnow)
    assessment_confidence: float = 0.0
    
    def is_healthy(self) -> bool:
        """Check if server is healthy"""
        return self.status == HealthStatus.HEALTHY
    
    def needs_attention(self) -> bool:
        """Check if server needs attention"""
        return self.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]


@dataclass
class MonitoringAlert:
    """Monitoring alert"""
    server_name: str
    alert_type: str
    level: AlertLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metric_value: Optional[float] = None
    threshold_value: Optional[float] = None
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class ServerMonitor:
    """
    Real-time monitoring system for MCP servers.
    
    Continuously monitors server health, performance, and availability.
    Provides alerts, health assessments, and performance analytics.
    """
    
    def __init__(self, monitoring_interval: float = 30.0):
        self.monitoring_interval = monitoring_interval
        
        # Monitoring state
        self.is_monitoring = False
        self.monitored_servers: Dict[str, Dict[str, Any]] = {}
        
        # Performance data storage
        self.performance_history: Dict[str, List[PerformanceMetrics]] = {}
        self.health_history: Dict[str, List[ServerHealth]] = {}
        
        # Alert system
        self.active_alerts: List[MonitoringAlert] = []
        self.alert_history: List[MonitoringAlert] = []
        self.alert_callbacks: List[Callable] = []
        
        # Monitoring thresholds
        self.thresholds = self._initialize_thresholds()
        
        # Statistics
        self.monitoring_stats = {
            'total_checks': 0,
            'failed_checks': 0,
            'alerts_generated': 0,
            'servers_monitored': 0
        }
    
    def _initialize_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize monitoring thresholds"""
        return {
            'response_time': {
                'warning': 1000.0,  # ms
                'critical': 5000.0   # ms
            },
            'error_rate': {
                'warning': 0.05,     # 5%
                'critical': 0.15     # 15%
            },
            'cpu_usage': {
                'warning': 70.0,     # %
                'critical': 90.0     # %
            },
            'memory_usage': {
                'warning': 80.0,     # %
                'critical': 95.0     # %
            },
            'uptime': {
                'warning': 95.0,     # %
                'critical': 90.0     # %
            },
            'success_rate': {
                'warning': 95.0,     # %
                'critical': 85.0     # %
            }
        }
    
    def add_server(self, server_name: str, server_config: Dict[str, Any]):
        """Add server to monitoring"""
        self.monitored_servers[server_name] = {
            'config': server_config,
            'last_check': None,
            'consecutive_failures': 0,
            'start_time': datetime.utcnow()
        }
        
        # Initialize history
        self.performance_history[server_name] = []
        self.health_history[server_name] = []
        
        logger.info(f"Added server {server_name} to monitoring")
    
    def remove_server(self, server_name: str):
        """Remove server from monitoring"""
        if server_name in self.monitored_servers:
            del self.monitored_servers[server_name]
            
            # Clean up history
            if server_name in self.performance_history:
                del self.performance_history[server_name]
            if server_name in self.health_history:
                del self.health_history[server_name]
            
            logger.info(f"Removed server {server_name} from monitoring")
    
    async def start_monitoring(self):
        """Start the monitoring loop"""
        if self.is_monitoring:
            logger.warning("Monitoring is already running")
            return
        
        self.is_monitoring = True
        logger.info("Starting server monitoring")
        
        try:
            while self.is_monitoring:
                await self._monitoring_cycle()
                await asyncio.sleep(self.monitoring_interval)
        except Exception as e:
            logger.error(f"Monitoring loop failed: {e}")
        finally:
            self.is_monitoring = False
    
    def stop_monitoring(self):
        """Stop the monitoring loop"""
        self.is_monitoring = False
        logger.info("Stopping server monitoring")
    
    async def _monitoring_cycle(self):
        """Perform one monitoring cycle"""
        for server_name in list(self.monitored_servers.keys()):
            try:
                await self._check_server(server_name)
                self.monitoring_stats['total_checks'] += 1
            except Exception as e:
                logger.error(f"Failed to check server {server_name}: {e}")
                self.monitoring_stats['failed_checks'] += 1
        
        # Update statistics
        self.monitoring_stats['servers_monitored'] = len(self.monitored_servers)
        
        # Clean up old data
        await self._cleanup_old_data()
    
    async def _check_server(self, server_name: str):
        """Check a single server"""
        server_info = self.monitored_servers[server_name]
        
        # Collect performance metrics
        metrics = await self._collect_metrics(server_name, server_info)
        
        # Store metrics
        self.performance_history[server_name].append(metrics)
        
        # Assess health
        health = await self._assess_health(server_name, metrics)
        
        # Store health assessment
        self.health_history[server_name].append(health)
        
        # Check for alerts
        await self._check_alerts(server_name, metrics, health)
        
        # Update server info
        server_info['last_check'] = datetime.utcnow()
        
        # Reset consecutive failures if check succeeded
        if health.status != HealthStatus.OFFLINE:
            server_info['consecutive_failures'] = 0
        else:
            server_info['consecutive_failures'] += 1
    
    async def _collect_metrics(self, server_name: str, server_info: Dict[str, Any]) -> PerformanceMetrics:
        """Collect performance metrics for a server"""
        # This would normally make actual requests to the server
        # For now, we'll simulate metrics collection
        
        metrics = PerformanceMetrics()
        
        try:
            # Simulate server health check
            start_time = time.time()
            
            # Simulate network request (replace with actual MCP call)
            await asyncio.sleep(0.1)  # Simulate network delay
            
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Update metrics
            metrics.avg_response_time_ms = response_time
            metrics.min_response_time_ms = response_time
            metrics.max_response_time_ms = response_time
            metrics.p95_response_time_ms = response_time
            metrics.p99_response_time_ms = response_time
            
            metrics.successful_requests = 1
            metrics.total_requests = 1
            metrics.error_rate = 0.0
            
            # Simulate resource metrics
            metrics.cpu_usage_percent = 25.0 + (time.time() % 60) / 60 * 50  # Varying CPU
            metrics.memory_usage_mb = 512 + (time.time() % 100) * 5  # Varying memory
            metrics.memory_usage_percent = metrics.memory_usage_mb / 2048 * 100
            
            # Calculate uptime
            start_time_dt = server_info.get('start_time', datetime.utcnow())
            uptime_seconds = (datetime.utcnow() - start_time_dt).total_seconds()
            metrics.uptime_seconds = uptime_seconds
            metrics.uptime_percentage = 99.5  # Assume high uptime
            
            metrics.last_seen = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Failed to collect metrics for {server_name}: {e}")
            # Mark as failed
            metrics.failed_requests = 1
            metrics.total_requests = 1
            metrics.error_rate = 1.0
        
        return metrics
    
    async def _assess_health(self, server_name: str, metrics: PerformanceMetrics) -> ServerHealth:
        """Assess server health based on metrics"""
        health = ServerHealth(server_name=server_name)
        
        # Calculate component scores
        health.response_time_health = self._calculate_response_time_health(metrics)
        health.error_rate_health = self._calculate_error_rate_health(metrics)
        health.resource_health = self._calculate_resource_health(metrics)
        health.uptime_health = self._calculate_uptime_health(metrics)
        
        # Calculate overall component scores
        health.performance_score = (health.response_time_health + health.error_rate_health) / 2
        health.availability_score = health.uptime_health
        health.reliability_score = (health.error_rate_health + health.uptime_health) / 2
        health.resource_score = health.resource_health
        
        # Calculate overall health score
        health.overall_score = (
            health.performance_score * 0.3 +
            health.availability_score * 0.3 +
            health.reliability_score * 0.2 +
            health.resource_score * 0.2
        )
        
        # Determine health status
        if health.overall_score >= 0.8:
            health.status = HealthStatus.HEALTHY
        elif health.overall_score >= 0.6:
            health.status = HealthStatus.WARNING
        elif health.overall_score >= 0.3:
            health.status = HealthStatus.CRITICAL
        else:
            health.status = HealthStatus.OFFLINE
        
        # Generate issues and recommendations
        self._generate_health_insights(health, metrics)
        
        # Set assessment confidence
        health.assessment_confidence = min(1.0, len(self.performance_history[server_name]) / 10)
        
        return health
    
    def _calculate_response_time_health(self, metrics: PerformanceMetrics) -> float:
        """Calculate response time health score"""
        response_time = metrics.avg_response_time_ms
        
        if response_time <= self.thresholds['response_time']['warning']:
            return 1.0
        elif response_time <= self.thresholds['response_time']['critical']:
            # Linear degradation between warning and critical
            warning_threshold = self.thresholds['response_time']['warning']
            critical_threshold = self.thresholds['response_time']['critical']
            ratio = (response_time - warning_threshold) / (critical_threshold - warning_threshold)
            return 1.0 - (ratio * 0.5)  # Degrade from 1.0 to 0.5
        else:
            return 0.2  # Very poor performance
    
    def _calculate_error_rate_health(self, metrics: PerformanceMetrics) -> float:
        """Calculate error rate health score"""
        error_rate = metrics.error_rate
        
        if error_rate <= self.thresholds['error_rate']['warning']:
            return 1.0
        elif error_rate <= self.thresholds['error_rate']['critical']:
            # Linear degradation
            warning_threshold = self.thresholds['error_rate']['warning']
            critical_threshold = self.thresholds['error_rate']['critical']
            ratio = (error_rate - warning_threshold) / (critical_threshold - warning_threshold)
            return 1.0 - (ratio * 0.5)
        else:
            return 0.1  # Very high error rate
    
    def _calculate_resource_health(self, metrics: PerformanceMetrics) -> float:
        """Calculate resource usage health score"""
        cpu_health = 1.0
        memory_health = 1.0
        
        # CPU health
        if metrics.cpu_usage_percent > self.thresholds['cpu_usage']['critical']:
            cpu_health = 0.2
        elif metrics.cpu_usage_percent > self.thresholds['cpu_usage']['warning']:
            ratio = (metrics.cpu_usage_percent - self.thresholds['cpu_usage']['warning']) / \
                   (self.thresholds['cpu_usage']['critical'] - self.thresholds['cpu_usage']['warning'])
            cpu_health = 1.0 - (ratio * 0.6)
        
        # Memory health
        if metrics.memory_usage_percent > self.thresholds['memory_usage']['critical']:
            memory_health = 0.2
        elif metrics.memory_usage_percent > self.thresholds['memory_usage']['warning']:
            ratio = (metrics.memory_usage_percent - self.thresholds['memory_usage']['warning']) / \
                   (self.thresholds['memory_usage']['critical'] - self.thresholds['memory_usage']['warning'])
            memory_health = 1.0 - (ratio * 0.6)
        
        return (cpu_health + memory_health) / 2
    
    def _calculate_uptime_health(self, metrics: PerformanceMetrics) -> float:
        """Calculate uptime health score"""
        uptime = metrics.uptime_percentage
        
        if uptime >= self.thresholds['uptime']['warning']:
            return 1.0
        elif uptime >= self.thresholds['uptime']['critical']:
            # Linear degradation
            warning_threshold = self.thresholds['uptime']['warning']
            critical_threshold = self.thresholds['uptime']['critical']
            ratio = (warning_threshold - uptime) / (warning_threshold - critical_threshold)
            return 1.0 - (ratio * 0.5)
        else:
            return 0.3  # Poor uptime
    
    def _generate_health_insights(self, health: ServerHealth, metrics: PerformanceMetrics):
        """Generate health insights, issues, and recommendations"""
        # Check for issues
        if metrics.avg_response_time_ms > self.thresholds['response_time']['critical']:
            health.issues.append(f"High response time: {metrics.avg_response_time_ms:.1f}ms")
        
        if metrics.error_rate > self.thresholds['error_rate']['critical']:
            health.issues.append(f"High error rate: {metrics.error_rate:.1%}")
        
        if metrics.cpu_usage_percent > self.thresholds['cpu_usage']['critical']:
            health.issues.append(f"High CPU usage: {metrics.cpu_usage_percent:.1f}%")
        
        if metrics.memory_usage_percent > self.thresholds['memory_usage']['critical']:
            health.issues.append(f"High memory usage: {metrics.memory_usage_percent:.1f}%")
        
        # Check for warnings
        if metrics.avg_response_time_ms > self.thresholds['response_time']['warning']:
            health.warnings.append(f"Elevated response time: {metrics.avg_response_time_ms:.1f}ms")
        
        if metrics.error_rate > self.thresholds['error_rate']['warning']:
            health.warnings.append(f"Elevated error rate: {metrics.error_rate:.1%}")
        
        # Generate recommendations
        if health.resource_score < 0.7:
            health.recommendations.append("Consider scaling resources or optimizing performance")
        
        if health.performance_score < 0.6:
            health.recommendations.append("Investigate performance bottlenecks")
        
        if health.reliability_score < 0.8:
            health.recommendations.append("Review error handling and retry mechanisms")
    
    async def _check_alerts(self, server_name: str, metrics: PerformanceMetrics, health: ServerHealth):
        """Check for alert conditions"""
        # Response time alerts
        if metrics.avg_response_time_ms > self.thresholds['response_time']['critical']:
            await self._create_alert(
                server_name, "response_time", AlertLevel.CRITICAL,
                f"Response time {metrics.avg_response_time_ms:.1f}ms exceeds critical threshold",
                metrics.avg_response_time_ms, self.thresholds['response_time']['critical']
            )
        elif metrics.avg_response_time_ms > self.thresholds['response_time']['warning']:
            await self._create_alert(
                server_name, "response_time", AlertLevel.WARNING,
                f"Response time {metrics.avg_response_time_ms:.1f}ms exceeds warning threshold",
                metrics.avg_response_time_ms, self.thresholds['response_time']['warning']
            )
        
        # Error rate alerts
        if metrics.error_rate > self.thresholds['error_rate']['critical']:
            await self._create_alert(
                server_name, "error_rate", AlertLevel.CRITICAL,
                f"Error rate {metrics.error_rate:.1%} exceeds critical threshold",
                metrics.error_rate, self.thresholds['error_rate']['critical']
            )
        
        # Resource alerts
        if metrics.cpu_usage_percent > self.thresholds['cpu_usage']['critical']:
            await self._create_alert(
                server_name, "cpu_usage", AlertLevel.CRITICAL,
                f"CPU usage {metrics.cpu_usage_percent:.1f}% exceeds critical threshold",
                metrics.cpu_usage_percent, self.thresholds['cpu_usage']['critical']
            )
        
        # Health status alerts
        if health.status == HealthStatus.CRITICAL:
            await self._create_alert(
                server_name, "health_status", AlertLevel.ERROR,
                f"Server health is critical (score: {health.overall_score:.2f})",
                health.overall_score, 0.3
            )
    
    async def _create_alert(self, server_name: str, alert_type: str, level: AlertLevel,
                          message: str, metric_value: float, threshold_value: float):
        """Create and process an alert"""
        # Check if similar alert already exists
        existing_alert = None
        for alert in self.active_alerts:
            if (alert.server_name == server_name and 
                alert.alert_type == alert_type and 
                not alert.resolved):
                existing_alert = alert
                break
        
        if existing_alert:
            # Update existing alert
            existing_alert.message = message
            existing_alert.metric_value = metric_value
            existing_alert.timestamp = datetime.utcnow()
        else:
            # Create new alert
            alert = MonitoringAlert(
                server_name=server_name,
                alert_type=alert_type,
                level=level,
                message=message,
                metric_value=metric_value,
                threshold_value=threshold_value
            )
            
            self.active_alerts.append(alert)
            self.alert_history.append(alert)
            self.monitoring_stats['alerts_generated'] += 1
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert)
                    else:
                        callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        # Clean up performance history
        for server_name in self.performance_history:
            self.performance_history[server_name] = [
                metrics for metrics in self.performance_history[server_name]
                if metrics.last_seen > cutoff_time
            ]
        
        # Clean up health history
        for server_name in self.health_history:
            self.health_history[server_name] = [
                health for health in self.health_history[server_name]
                if health.last_assessment > cutoff_time
            ]
        
        # Clean up alert history
        self.alert_history = [
            alert for alert in self.alert_history
            if alert.timestamp > cutoff_time
        ]
    
    def get_server_health(self, server_name: str) -> Optional[ServerHealth]:
        """Get current health for a server"""
        if server_name not in self.health_history:
            return None
        
        health_list = self.health_history[server_name]
        return health_list[-1] if health_list else None
    
    def get_server_metrics(self, server_name: str) -> Optional[PerformanceMetrics]:
        """Get current metrics for a server"""
        if server_name not in self.performance_history:
            return None
        
        metrics_list = self.performance_history[server_name]
        return metrics_list[-1] if metrics_list else None
    
    def get_active_alerts(self, server_name: Optional[str] = None) -> List[MonitoringAlert]:
        """Get active alerts, optionally filtered by server"""
        alerts = [alert for alert in self.active_alerts if not alert.resolved]
        
        if server_name:
            alerts = [alert for alert in alerts if alert.server_name == server_name]
        
        return alerts
    
    def resolve_alert(self, alert: MonitoringAlert):
        """Resolve an alert"""
        alert.resolved = True
        alert.resolution_time = datetime.utcnow()
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary"""
        healthy_servers = 0
        warning_servers = 0
        critical_servers = 0
        offline_servers = 0
        
        for server_name in self.monitored_servers:
            health = self.get_server_health(server_name)
            if health:
                if health.status == HealthStatus.HEALTHY:
                    healthy_servers += 1
                elif health.status == HealthStatus.WARNING:
                    warning_servers += 1
                elif health.status == HealthStatus.CRITICAL:
                    critical_servers += 1
                elif health.status == HealthStatus.OFFLINE:
                    offline_servers += 1
        
        return {
            'monitoring_active': self.is_monitoring,
            'total_servers': len(self.monitored_servers),
            'server_status': {
                'healthy': healthy_servers,
                'warning': warning_servers,
                'critical': critical_servers,
                'offline': offline_servers
            },
            'active_alerts': len(self.get_active_alerts()),
            'monitoring_stats': self.monitoring_stats.copy(),
            'monitoring_interval': self.monitoring_interval
        }
