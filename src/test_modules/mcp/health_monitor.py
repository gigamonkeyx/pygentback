"""
MCP Health Monitor

Monitors the health and performance of MCP servers during testing.
Provides real-time health checks, performance metrics, and alerting.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    OFFLINE = "offline"


@dataclass
class HealthMetrics:
    """Health metrics for an MCP server"""
    server_id: str
    status: HealthStatus
    response_time_ms: float
    cpu_usage_percent: float
    memory_usage_mb: float
    active_connections: int
    total_requests: int
    failed_requests: int
    last_check: datetime = field(default_factory=datetime.utcnow)
    uptime_seconds: float = 0.0
    error_rate_percent: float = 0.0


@dataclass
class HealthAlert:
    """Health alert information"""
    server_id: str
    alert_type: str
    severity: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class HealthThresholds:
    """Health monitoring thresholds"""
    response_time_warning_ms: float = 1000.0
    response_time_critical_ms: float = 5000.0
    cpu_usage_warning_percent: float = 70.0
    cpu_usage_critical_percent: float = 90.0
    memory_usage_warning_mb: float = 500.0
    memory_usage_critical_mb: float = 1000.0
    error_rate_warning_percent: float = 5.0
    error_rate_critical_percent: float = 15.0
    max_failed_checks: int = 3


class MCPHealthMonitor:
    """
    MCP Server Health Monitoring System.
    
    Provides continuous health monitoring, performance tracking,
    and alerting for MCP servers during testing operations.
    """
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.thresholds = HealthThresholds()
        
        # Monitoring state
        self.monitored_servers: Dict[str, Any] = {}  # server_id -> server_client
        self.health_metrics: Dict[str, HealthMetrics] = {}
        self.health_history: Dict[str, List[HealthMetrics]] = {}
        self.active_alerts: Dict[str, List[HealthAlert]] = {}
        
        # Monitoring control
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.alert_callbacks: List[Callable[[HealthAlert], None]] = []
        
        # Performance tracking
        self.start_times: Dict[str, datetime] = {}
        self.request_counts: Dict[str, int] = {}
        self.failure_counts: Dict[str, int] = {}
        self.consecutive_failures: Dict[str, int] = {}
    
    def add_server(self, server_id: str, server_client: Any):
        """Add a server to monitoring"""
        self.monitored_servers[server_id] = server_client
        self.health_metrics[server_id] = HealthMetrics(
            server_id=server_id,
            status=HealthStatus.UNKNOWN,
            response_time_ms=0.0,
            cpu_usage_percent=0.0,
            memory_usage_mb=0.0,
            active_connections=0,
            total_requests=0,
            failed_requests=0
        )
        self.health_history[server_id] = []
        self.active_alerts[server_id] = []
        self.start_times[server_id] = datetime.utcnow()
        self.request_counts[server_id] = 0
        self.failure_counts[server_id] = 0
        self.consecutive_failures[server_id] = 0
        
        logger.info(f"Added server {server_id} to health monitoring")
    
    def remove_server(self, server_id: str):
        """Remove a server from monitoring"""
        if server_id in self.monitored_servers:
            del self.monitored_servers[server_id]
            del self.health_metrics[server_id]
            del self.health_history[server_id]
            del self.active_alerts[server_id]
            del self.start_times[server_id]
            del self.request_counts[server_id]
            del self.failure_counts[server_id]
            del self.consecutive_failures[server_id]
            
            logger.info(f"Removed server {server_id} from health monitoring")
    
    def add_alert_callback(self, callback: Callable[[HealthAlert], None]):
        """Add a callback for health alerts"""
        self.alert_callbacks.append(callback)
    
    async def start_monitoring(self):
        """Start continuous health monitoring"""
        if self.monitoring_active:
            logger.warning("Health monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started MCP health monitoring")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped MCP health monitoring")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Check health of all monitored servers
                for server_id in list(self.monitored_servers.keys()):
                    await self._check_server_health(server_id)
                
                # Clean up old history entries
                self._cleanup_history()
                
                # Wait for next check
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _check_server_health(self, server_id: str):
        """Check health of a specific server"""
        try:
            server_client = self.monitored_servers[server_id]
            start_time = time.time()
            
            # Perform health check (ping or simple operation)
            try:
                # Try to get server status or perform a simple operation
                if hasattr(server_client, 'get_server_status'):
                    await server_client.get_server_status(server_id)
                elif hasattr(server_client, 'list_tools'):
                    await server_client.list_tools(server_id)
                else:
                    # Fallback: assume healthy if client exists
                    pass
                
                response_time = (time.time() - start_time) * 1000
                self.consecutive_failures[server_id] = 0
                
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                self.consecutive_failures[server_id] += 1
                self.failure_counts[server_id] += 1
                logger.warning(f"Health check failed for {server_id}: {e}")
            
            # Update request count
            self.request_counts[server_id] += 1
            
            # Calculate metrics
            uptime = (datetime.utcnow() - self.start_times[server_id]).total_seconds()
            error_rate = (self.failure_counts[server_id] / max(self.request_counts[server_id], 1)) * 100
            
            # Determine health status
            status = self._determine_health_status(server_id, response_time, error_rate)
            
            # Create health metrics
            metrics = HealthMetrics(
                server_id=server_id,
                status=status,
                response_time_ms=response_time,
                cpu_usage_percent=self._get_cpu_usage(server_id),
                memory_usage_mb=self._get_memory_usage(server_id),
                active_connections=self._get_active_connections(server_id),
                total_requests=self.request_counts[server_id],
                failed_requests=self.failure_counts[server_id],
                uptime_seconds=uptime,
                error_rate_percent=error_rate
            )
            
            # Store metrics
            self.health_metrics[server_id] = metrics
            self.health_history[server_id].append(metrics)
            
            # Check for alerts
            await self._check_alerts(metrics)
            
        except Exception as e:
            logger.error(f"Failed to check health for server {server_id}: {e}")
            
            # Mark as offline
            metrics = HealthMetrics(
                server_id=server_id,
                status=HealthStatus.OFFLINE,
                response_time_ms=0.0,
                cpu_usage_percent=0.0,
                memory_usage_mb=0.0,
                active_connections=0,
                total_requests=self.request_counts.get(server_id, 0),
                failed_requests=self.failure_counts.get(server_id, 0)
            )
            
            self.health_metrics[server_id] = metrics
            self.health_history[server_id].append(metrics)
    
    def _determine_health_status(self, server_id: str, response_time: float, error_rate: float) -> HealthStatus:
        """Determine health status based on metrics"""
        # Check for consecutive failures
        if self.consecutive_failures[server_id] >= self.thresholds.max_failed_checks:
            return HealthStatus.CRITICAL
        
        # Check critical thresholds
        if (response_time > self.thresholds.response_time_critical_ms or
            error_rate > self.thresholds.error_rate_critical_percent):
            return HealthStatus.CRITICAL
        
        # Check warning thresholds
        if (response_time > self.thresholds.response_time_warning_ms or
            error_rate > self.thresholds.error_rate_warning_percent):
            return HealthStatus.WARNING
        
        return HealthStatus.HEALTHY
    
    def _get_cpu_usage(self, server_id: str) -> float:
        """Get CPU usage for server (mock implementation)"""
        # In a real implementation, this would query actual CPU usage
        return 25.0 + (hash(server_id) % 20)
    
    def _get_memory_usage(self, server_id: str) -> float:
        """Get memory usage for server (mock implementation)"""
        # In a real implementation, this would query actual memory usage
        return 100.0 + (hash(server_id) % 50)
    
    def _get_active_connections(self, server_id: str) -> int:
        """Get active connections for server (mock implementation)"""
        # In a real implementation, this would query actual connection count
        return 1 + (hash(server_id) % 5)
    
    async def _check_alerts(self, metrics: HealthMetrics):
        """Check for alert conditions and generate alerts"""
        server_id = metrics.server_id
        
        # Check for new alerts
        alerts_to_create = []
        
        if metrics.status == HealthStatus.CRITICAL:
            alerts_to_create.append(HealthAlert(
                server_id=server_id,
                alert_type="critical_status",
                severity=HealthStatus.CRITICAL,
                message=f"Server {server_id} is in critical state"
            ))
        
        if metrics.response_time_ms > self.thresholds.response_time_critical_ms:
            alerts_to_create.append(HealthAlert(
                server_id=server_id,
                alert_type="high_response_time",
                severity=HealthStatus.CRITICAL,
                message=f"High response time: {metrics.response_time_ms:.1f}ms"
            ))
        
        if metrics.error_rate_percent > self.thresholds.error_rate_critical_percent:
            alerts_to_create.append(HealthAlert(
                server_id=server_id,
                alert_type="high_error_rate",
                severity=HealthStatus.CRITICAL,
                message=f"High error rate: {metrics.error_rate_percent:.1f}%"
            ))
        
        # Create and notify alerts
        for alert in alerts_to_create:
            # Check if similar alert already exists
            existing_alert = None
            for existing in self.active_alerts[server_id]:
                if (existing.alert_type == alert.alert_type and 
                    not existing.resolved):
                    existing_alert = existing
                    break
            
            if not existing_alert:
                self.active_alerts[server_id].append(alert)
                logger.warning(f"Health alert for {server_id}: {alert.message}")
                
                # Notify callbacks
                for callback in self.alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"Alert callback failed: {e}")
    
    def _cleanup_history(self):
        """Clean up old history entries"""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        for server_id in self.health_history:
            self.health_history[server_id] = [
                metrics for metrics in self.health_history[server_id]
                if metrics.last_check > cutoff_time
            ]
    
    def get_server_health(self, server_id: str) -> Optional[HealthMetrics]:
        """Get current health metrics for a server"""
        return self.health_metrics.get(server_id)
    
    def get_server_history(self, server_id: str, hours: int = 1) -> List[HealthMetrics]:
        """Get health history for a server"""
        if server_id not in self.health_history:
            return []
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            metrics for metrics in self.health_history[server_id]
            if metrics.last_check > cutoff_time
        ]
    
    def get_active_alerts(self, server_id: Optional[str] = None) -> List[HealthAlert]:
        """Get active alerts for a server or all servers"""
        if server_id:
            return [alert for alert in self.active_alerts.get(server_id, []) if not alert.resolved]
        
        all_alerts = []
        for alerts in self.active_alerts.values():
            all_alerts.extend([alert for alert in alerts if not alert.resolved])
        return all_alerts
    
    def resolve_alert(self, server_id: str, alert_type: str):
        """Resolve an active alert"""
        for alert in self.active_alerts.get(server_id, []):
            if alert.alert_type == alert_type and not alert.resolved:
                alert.resolved = True
                alert.resolution_time = datetime.utcnow()
                logger.info(f"Resolved alert {alert_type} for server {server_id}")
                break
    
    def get_overall_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary for all monitored servers"""
        total_servers = len(self.monitored_servers)
        if total_servers == 0:
            return {"total_servers": 0, "status": "no_servers"}
        
        status_counts = {}
        total_response_time = 0
        total_error_rate = 0
        
        for metrics in self.health_metrics.values():
            status = metrics.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            total_response_time += metrics.response_time_ms
            total_error_rate += metrics.error_rate_percent
        
        avg_response_time = total_response_time / total_servers
        avg_error_rate = total_error_rate / total_servers
        
        # Determine overall status
        if status_counts.get("critical", 0) > 0:
            overall_status = "critical"
        elif status_counts.get("warning", 0) > 0:
            overall_status = "warning"
        elif status_counts.get("healthy", 0) == total_servers:
            overall_status = "healthy"
        else:
            overall_status = "mixed"
        
        return {
            "total_servers": total_servers,
            "status": overall_status,
            "status_breakdown": status_counts,
            "average_response_time_ms": avg_response_time,
            "average_error_rate_percent": avg_error_rate,
            "active_alerts": len(self.get_active_alerts())
        }
