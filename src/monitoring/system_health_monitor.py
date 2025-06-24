# src/monitoring/system_health_monitor.py

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import psutil
import os
import json
from dataclasses import dataclass
import aiohttp
import time

from ..core.gpu_config import gpu_manager
from ..core.ollama_manager import get_ollama_manager
from ..storage.vector.manager import VectorStoreManager
from ..utils.embedding import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System performance metrics snapshot"""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_available_gb: float
    gpu_usage_percent: float = 0.0
    gpu_memory_usage_percent: float = 0.0
    gpu_temperature: float = 0.0
    network_io_mbps: float = 0.0
    process_count: int = 0


@dataclass
class ServiceStatus:
    """Status of a system service"""
    service_name: str
    status: str  # "healthy", "degraded", "failed", "unknown"
    response_time_ms: float
    last_check: datetime
    error_message: str = ""
    uptime_seconds: float = 0.0
    success_rate: float = 1.0


@dataclass
class HealthAlert:
    """System health alert"""
    id: str
    severity: str  # "low", "medium", "high", "critical"
    component: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class PerformanceTrend:
    """Performance trend analysis"""
    metric_name: str
    current_value: float
    average_1h: float
    average_24h: float
    trend_direction: str  # "improving", "stable", "degrading"
    variance: float
    prediction_1h: float


class SystemHealthMonitor:
    """Comprehensive system health and performance monitoring."""
    
    def __init__(self, 
                 vector_manager: Optional[VectorStoreManager] = None,
                 embedding_service: Optional[EmbeddingService] = None,
                 monitoring_interval: int = 60):
        self.vector_manager = vector_manager
        self.embedding_service = embedding_service
        self.monitoring_interval = monitoring_interval
        
        # Core services
        self.ollama_manager = get_ollama_manager()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task = None
        
        # Data storage
        self.metrics_history: List[SystemMetrics] = []
        self.service_statuses: Dict[str, ServiceStatus] = {}
        self.active_alerts: Dict[str, HealthAlert] = {}
        self.performance_trends: Dict[str, PerformanceTrend] = {}
        
        # Configuration
        self.max_history_points = 1440  # 24 hours of minute-by-minute data
        self.alert_thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'gpu_usage': 90.0,
            'response_time': 5000.0  # 5 seconds
        }
        
        # Service endpoints to monitor
        self.monitored_services = {
            'ollama': {'url': f'{self.ollama_manager.ollama_url}/api/tags', 'timeout': 5},
            'vector_store': {'check_method': 'vector_health_check'},
            'embedding_service': {'check_method': 'embedding_health_check'},
            'gpu_manager': {'check_method': 'gpu_health_check'},
            'system': {'check_method': 'system_health_check'}
        }
        
        logger.info("System Health Monitor initialized")
    
    async def start_monitoring(self):
        """Start continuous system health monitoring."""
        if self.is_monitoring:
            logger.warning("Health monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("System health monitoring started")
    
    async def stop_monitoring(self):
        """Stop system health monitoring."""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("System health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                start_time = time.time()
                
                # Collect system metrics
                metrics = await self._collect_system_metrics()
                self._store_metrics(metrics)
                
                # Check service health
                await self._check_all_services()
                
                # Analyze trends and generate alerts
                await self._analyze_performance_trends()
                await self._check_alert_conditions()
                
                # Cleanup old data
                self._cleanup_old_data()
                
                # Calculate sleep time to maintain interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.monitoring_interval - elapsed)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system performance metrics."""
        try:
            # CPU metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_usage_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            disk_available_gb = disk.free / (1024**3)
            
            # Process count
            process_count = len(psutil.pids())
            
            # Network I/O (simplified)
            network_io = psutil.net_io_counters()
            network_io_mbps = (network_io.bytes_sent + network_io.bytes_recv) / (1024**2)
            
            # GPU metrics
            gpu_usage = 0.0
            gpu_memory_usage = 0.0
            gpu_temperature = 0.0
            
            if gpu_manager.is_available():
                gpu_metrics = await self._collect_gpu_metrics()
                gpu_usage = gpu_metrics.get('usage_percent', 0.0)
                gpu_memory_usage = gpu_metrics.get('memory_usage_percent', 0.0)
                gpu_temperature = gpu_metrics.get('temperature', 0.0)
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage_percent=cpu_usage,
                memory_usage_percent=memory_usage_percent,
                memory_available_gb=memory_available_gb,
                disk_usage_percent=disk_usage_percent,
                disk_available_gb=disk_available_gb,
                gpu_usage_percent=gpu_usage,
                gpu_memory_usage_percent=gpu_memory_usage,
                gpu_temperature=gpu_temperature,
                network_io_mbps=network_io_mbps,
                process_count=process_count
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage_percent=0.0,
                memory_usage_percent=0.0,
                memory_available_gb=0.0,
                disk_usage_percent=0.0,
                disk_available_gb=0.0
            )
    
    async def _collect_gpu_metrics(self) -> Dict[str, float]:
        """Collect GPU-specific metrics."""
        try:
            # This would integrate with GPU monitoring libraries
            # For now, return placeholder metrics
            return {
                'usage_percent': 0.0,
                'memory_usage_percent': 0.0,
                'temperature': 0.0
            }
        except Exception as e:
            logger.error(f"Error collecting GPU metrics: {str(e)}")
            return {}
    
    def _store_metrics(self, metrics: SystemMetrics):
        """Store metrics and maintain history size."""
        self.metrics_history.append(metrics)
        
        # Keep only recent history
        if len(self.metrics_history) > self.max_history_points:
            self.metrics_history = self.metrics_history[-self.max_history_points:]
    
    async def _check_all_services(self):
        """Check health of all monitored services."""
        for service_name, config in self.monitored_services.items():
            try:
                status = await self._check_service_health(service_name, config)
                self.service_statuses[service_name] = status
            except Exception as e:
                logger.error(f"Error checking service {service_name}: {str(e)}")
                self.service_statuses[service_name] = ServiceStatus(
                    service_name=service_name,
                    status="failed",
                    response_time_ms=0.0,
                    last_check=datetime.now(),
                    error_message=str(e)
                )
    
    async def _check_service_health(self, service_name: str, config: Dict[str, Any]) -> ServiceStatus:
        """Check health of a specific service."""
        start_time = time.time()
        
        try:
            if 'url' in config:
                # HTTP endpoint check
                status = await self._check_http_endpoint(config['url'], config.get('timeout', 5))
            elif 'check_method' in config:
                # Custom health check method
                status = await self._run_custom_health_check(config['check_method'])
            else:
                raise ValueError(f"No check method configured for service {service_name}")
            
            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            return ServiceStatus(
                service_name=service_name,
                status=status['status'],
                response_time_ms=response_time,
                last_check=datetime.now(),
                error_message=status.get('error', ''),
                uptime_seconds=status.get('uptime', 0.0),
                success_rate=status.get('success_rate', 1.0)
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return ServiceStatus(
                service_name=service_name,
                status="failed",
                response_time_ms=response_time,
                last_check=datetime.now(),
                error_message=str(e)
            )
    
    async def _check_http_endpoint(self, url: str, timeout: int) -> Dict[str, Any]:
        """Check HTTP endpoint health."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                    if response.status == 200:
                        return {'status': 'healthy'}
                    else:
                        return {
                            'status': 'degraded',
                            'error': f'HTTP {response.status}: {response.reason}'
                        }
        except asyncio.TimeoutError:
            return {'status': 'degraded', 'error': 'Request timeout'}
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
    
    async def _run_custom_health_check(self, method_name: str) -> Dict[str, Any]:
        """Run custom health check method."""
        if method_name == 'vector_health_check':
            return await self._vector_health_check()
        elif method_name == 'embedding_health_check':
            return await self._embedding_health_check()
        elif method_name == 'gpu_health_check':
            return await self._gpu_health_check()
        elif method_name == 'system_health_check':
            return await self._system_health_check()
        else:
            return {'status': 'unknown', 'error': f'Unknown check method: {method_name}'}
    
    async def _vector_health_check(self) -> Dict[str, Any]:
        """Check vector store health."""
        try:
            if not self.vector_manager:
                return {'status': 'unknown', 'error': 'Vector manager not available'}
            
            # Try a simple operation
            collections = await self.vector_manager.list_collections()
            return {'status': 'healthy', 'collections': len(collections)}
            
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
    
    async def _embedding_health_check(self) -> Dict[str, Any]:
        """Check embedding service health."""
        try:
            if not self.embedding_service:
                return {'status': 'unknown', 'error': 'Embedding service not available'}
            
            # Try a simple embedding operation
            test_text = "health check"
            embedding = await self.embedding_service.get_embedding(test_text)
            
            if embedding and len(embedding) > 0:
                return {'status': 'healthy', 'embedding_dim': len(embedding)}
            else:
                return {'status': 'degraded', 'error': 'Empty embedding returned'}
                
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
    
    async def _gpu_health_check(self) -> Dict[str, Any]:
        """Check GPU health."""
        try:
            if not gpu_manager.is_available():
                return {'status': 'healthy', 'note': 'GPU not required'}
            
            config = gpu_manager.get_config()
            if config:
                return {
                    'status': 'healthy',
                    'device_type': config.device_type,
                    'device_name': config.device_name
                }
            else:
                return {'status': 'degraded', 'error': 'GPU configuration unavailable'}
                
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
    
    async def _system_health_check(self) -> Dict[str, Any]:
        """Check overall system health."""
        try:
            # Get latest metrics
            if not self.metrics_history:
                return {'status': 'unknown', 'error': 'No metrics available'}
            
            latest = self.metrics_history[-1]
            
            # Check critical thresholds
            issues = []
            if latest.cpu_usage_percent > 90:
                issues.append(f"High CPU usage: {latest.cpu_usage_percent:.1f}%")
            if latest.memory_usage_percent > 90:
                issues.append(f"High memory usage: {latest.memory_usage_percent:.1f}%")
            if latest.disk_usage_percent > 95:
                issues.append(f"Critical disk usage: {latest.disk_usage_percent:.1f}%")
            
            if issues:
                return {'status': 'degraded', 'issues': issues}
            else:
                return {'status': 'healthy'}
                
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
    
    async def _analyze_performance_trends(self):
        """Analyze performance trends and generate insights."""
        if len(self.metrics_history) < 10:
            return  # Need more data points
        
        try:
            # Analyze key metrics
            metrics_to_analyze = [
                'cpu_usage_percent',
                'memory_usage_percent',
                'disk_usage_percent',
                'gpu_usage_percent'
            ]
            
            for metric_name in metrics_to_analyze:
                trend = self._calculate_metric_trend(metric_name)
                if trend:
                    self.performance_trends[metric_name] = trend
                    
        except Exception as e:
            logger.error(f"Error analyzing performance trends: {str(e)}")
    
    def _calculate_metric_trend(self, metric_name: str) -> Optional[PerformanceTrend]:
        """Calculate trend for a specific metric."""
        try:
            # Get metric values
            values = []
            for metrics in self.metrics_history:
                value = getattr(metrics, metric_name, 0.0)
                values.append(value)
            
            if len(values) < 5:
                return None
            
            current_value = values[-1]
            
            # Calculate averages
            one_hour_points = min(60, len(values))
            average_1h = sum(values[-one_hour_points:]) / one_hour_points
            
            twenty_four_hour_points = min(1440, len(values))
            average_24h = sum(values[-twenty_four_hour_points:]) / twenty_four_hour_points
            
            # Calculate trend direction
            recent_avg = sum(values[-5:]) / 5
            older_avg = sum(values[-10:-5]) / 5 if len(values) >= 10 else recent_avg
            
            if recent_avg > older_avg * 1.1:
                trend_direction = "degrading"
            elif recent_avg < older_avg * 0.9:
                trend_direction = "improving"
            else:
                trend_direction = "stable"
            
            # Calculate variance
            variance = sum((v - average_1h) ** 2 for v in values[-one_hour_points:]) / one_hour_points
            
            # Simple linear prediction (very basic)
            if len(values) >= 5:
                x_values = list(range(len(values[-5:])))
                y_values = values[-5:]
                
                # Simple linear regression
                n = len(x_values)
                sum_x = sum(x_values)
                sum_y = sum(y_values)
                sum_xy = sum(x * y for x, y in zip(x_values, y_values))
                sum_x2 = sum(x * x for x in x_values)
                
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                prediction_1h = current_value + slope * 60  # Predict 1 hour ahead
            else:
                prediction_1h = current_value
            
            return PerformanceTrend(
                metric_name=metric_name,
                current_value=current_value,
                average_1h=average_1h,
                average_24h=average_24h,
                trend_direction=trend_direction,
                variance=variance,
                prediction_1h=max(0, prediction_1h)  # Don't predict negative values
            )
            
        except Exception as e:
            logger.error(f"Error calculating trend for {metric_name}: {str(e)}")
            return None
    
    async def _check_alert_conditions(self):
        """Check for alert conditions and generate alerts."""
        try:
            if not self.metrics_history:
                return
            
            latest = self.metrics_history[-1]
            
            # Check system resource alerts
            self._check_resource_alert('cpu_usage', latest.cpu_usage_percent)
            self._check_resource_alert('memory_usage', latest.memory_usage_percent)
            self._check_resource_alert('disk_usage', latest.disk_usage_percent)
            self._check_resource_alert('gpu_usage', latest.gpu_usage_percent)
            
            # Check service response time alerts
            for service_name, status in self.service_statuses.items():
                self._check_response_time_alert(service_name, status.response_time_ms)
                self._check_service_status_alert(service_name, status.status)
            
        except Exception as e:
            logger.error(f"Error checking alert conditions: {str(e)}")
    
    def _check_resource_alert(self, metric_name: str, current_value: float):
        """Check if a resource metric should trigger an alert."""
        threshold = self.alert_thresholds.get(metric_name)
        if not threshold:
            return
        
        alert_id = f"{metric_name}_high"
        
        if current_value > threshold:
            if alert_id not in self.active_alerts:
                # Create new alert
                severity = "critical" if current_value > threshold * 1.1 else "high"
                alert = HealthAlert(
                    id=alert_id,
                    severity=severity,
                    component=metric_name,
                    message=f"{metric_name.replace('_', ' ').title()} is {current_value:.1f}% (threshold: {threshold}%)",
                    timestamp=datetime.now()
                )
                self.active_alerts[alert_id] = alert
                logger.warning(f"Health alert: {alert.message}")
        else:
            # Resolve alert if it exists
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].resolved = True
                self.active_alerts[alert_id].resolution_time = datetime.now()
                logger.info(f"Health alert resolved: {alert_id}")
    
    def _check_response_time_alert(self, service_name: str, response_time_ms: float):
        """Check if service response time should trigger an alert."""
        threshold = self.alert_thresholds.get('response_time', 5000.0)
        alert_id = f"{service_name}_slow_response"
        
        if response_time_ms > threshold:
            if alert_id not in self.active_alerts:
                alert = HealthAlert(
                    id=alert_id,
                    severity="medium",
                    component=service_name,
                    message=f"{service_name} response time is {response_time_ms:.0f}ms (threshold: {threshold}ms)",
                    timestamp=datetime.now()
                )
                self.active_alerts[alert_id] = alert
                logger.warning(f"Response time alert: {alert.message}")
        else:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].resolved = True
                self.active_alerts[alert_id].resolution_time = datetime.now()
    
    def _check_service_status_alert(self, service_name: str, status: str):
        """Check if service status should trigger an alert."""
        alert_id = f"{service_name}_status"
        
        if status in ["failed", "degraded"]:
            if alert_id not in self.active_alerts:
                severity = "critical" if status == "failed" else "medium"
                alert = HealthAlert(
                    id=alert_id,
                    severity=severity,
                    component=service_name,
                    message=f"{service_name} service status is {status}",
                    timestamp=datetime.now()
                )
                self.active_alerts[alert_id] = alert
                logger.warning(f"Service status alert: {alert.message}")
        else:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].resolved = True
                self.active_alerts[alert_id].resolution_time = datetime.now()
    
    def _cleanup_old_data(self):
        """Clean up old monitoring data."""
        # Remove old metrics
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.metrics_history = [
            m for m in self.metrics_history 
            if m.timestamp > cutoff_time
        ]
        
        # Remove resolved alerts older than 1 hour
        alert_cutoff = datetime.now() - timedelta(hours=1)
        resolved_alerts = [
            alert_id for alert_id, alert in self.active_alerts.items()
            if alert.resolved and alert.resolution_time and alert.resolution_time < alert_cutoff
        ]
        
        for alert_id in resolved_alerts:
            del self.active_alerts[alert_id]
    
    # Public API methods
    
    async def get_system_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive system health summary."""
        try:
            # Overall system status
            active_alerts_count = len([a for a in self.active_alerts.values() if not a.resolved])
            critical_alerts = len([a for a in self.active_alerts.values() if a.severity == "critical" and not a.resolved])
            
            if critical_alerts > 0:
                overall_status = "critical"
            elif active_alerts_count > 0:
                overall_status = "degraded"
            else:
                overall_status = "healthy"
            
            # Latest metrics
            latest_metrics = None
            if self.metrics_history:
                latest = self.metrics_history[-1]
                latest_metrics = {
                    'timestamp': latest.timestamp.isoformat(),
                    'cpu_usage_percent': latest.cpu_usage_percent,
                    'memory_usage_percent': latest.memory_usage_percent,
                    'memory_available_gb': latest.memory_available_gb,
                    'disk_usage_percent': latest.disk_usage_percent,
                    'disk_available_gb': latest.disk_available_gb,
                    'gpu_usage_percent': latest.gpu_usage_percent,
                    'process_count': latest.process_count
                }
            
            # Service statuses
            service_summary = {}
            for service_name, status in self.service_statuses.items():
                service_summary[service_name] = {
                    'status': status.status,
                    'response_time_ms': status.response_time_ms,
                    'last_check': status.last_check.isoformat(),
                    'error_message': status.error_message
                }
            
            # Performance trends summary
            trends_summary = {}
            for metric_name, trend in self.performance_trends.items():
                trends_summary[metric_name] = {
                    'current_value': trend.current_value,
                    'trend_direction': trend.trend_direction,
                    'prediction_1h': trend.prediction_1h
                }
            
            return {
                'overall_status': overall_status,
                'monitoring_active': self.is_monitoring,
                'latest_metrics': latest_metrics,
                'service_statuses': service_summary,
                'performance_trends': trends_summary,
                'alerts': {
                    'active_count': active_alerts_count,
                    'critical_count': critical_alerts,
                    'recent_alerts': [
                        {
                            'id': alert.id,
                            'severity': alert.severity,
                            'component': alert.component,
                            'message': alert.message,
                            'timestamp': alert.timestamp.isoformat(),
                            'resolved': alert.resolved
                        }
                        for alert in sorted(self.active_alerts.values(), 
                                          key=lambda a: a.timestamp, reverse=True)[:10]
                    ]
                },
                'system_info': {
                    'monitoring_interval': self.monitoring_interval,
                    'metrics_history_points': len(self.metrics_history),
                    'uptime_hours': self._get_system_uptime_hours()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system health summary: {str(e)}")
            return {'error': f"Failed to get health summary: {str(e)}"}
    
    async def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Get detailed performance report for specified time period."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Filter metrics for time period
            period_metrics = [
                m for m in self.metrics_history 
                if m.timestamp > cutoff_time
            ]
            
            if not period_metrics:
                return {'error': 'No metrics available for the specified period'}
            
            # Calculate statistics
            cpu_values = [m.cpu_usage_percent for m in period_metrics]
            memory_values = [m.memory_usage_percent for m in period_metrics]
            disk_values = [m.disk_usage_percent for m in period_metrics]
            
            stats = {
                'cpu_usage': {
                    'average': sum(cpu_values) / len(cpu_values),
                    'maximum': max(cpu_values),
                    'minimum': min(cpu_values),
                    'current': cpu_values[-1] if cpu_values else 0
                },
                'memory_usage': {
                    'average': sum(memory_values) / len(memory_values),
                    'maximum': max(memory_values),
                    'minimum': min(memory_values),
                    'current': memory_values[-1] if memory_values else 0
                },
                'disk_usage': {
                    'average': sum(disk_values) / len(disk_values),
                    'maximum': max(disk_values),
                    'minimum': min(disk_values),
                    'current': disk_values[-1] if disk_values else 0
                }
            }
            
            return {
                'period_hours': hours,
                'data_points': len(period_metrics),
                'statistics': stats,
                'performance_trends': self.performance_trends,
                'recommendations': self._generate_performance_recommendations(stats)
            }
            
        except Exception as e:
            logger.error(f"Error generating performance report: {str(e)}")
            return {'error': f"Failed to generate performance report: {str(e)}"}
    
    def _get_system_uptime_hours(self) -> float:
        """Get system uptime in hours."""
        try:
            uptime_seconds = time.time() - psutil.boot_time()
            return uptime_seconds / 3600
        except Exception:
            return 0.0
    
    def _generate_performance_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # CPU recommendations
        if stats['cpu_usage']['average'] > 70:
            recommendations.append("High average CPU usage detected. Consider optimizing CPU-intensive processes.")
        
        # Memory recommendations
        if stats['memory_usage']['average'] > 80:
            recommendations.append("High memory usage detected. Consider increasing system memory or optimizing memory usage.")
        
        # Disk recommendations
        if stats['disk_usage']['average'] > 85:
            recommendations.append("High disk usage detected. Consider freeing up disk space or adding storage capacity.")
        
        # Performance trends recommendations
        for metric_name, trend in self.performance_trends.items():
            if trend.trend_direction == "degrading":
                recommendations.append(f"{metric_name.replace('_', ' ').title()} showing degrading trend. Monitor closely.")
        
        if not recommendations:
            recommendations.append("System performance is within normal parameters.")
        
        return recommendations
    
    async def export_monitoring_data(self, filepath: str) -> Dict[str, Any]:
        """Export monitoring data to JSON file."""
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'monitoring_config': {
                    'interval': self.monitoring_interval,
                    'thresholds': self.alert_thresholds
                },
                'metrics_history': [
                    {
                        'timestamp': m.timestamp.isoformat(),
                        'cpu_usage_percent': m.cpu_usage_percent,
                        'memory_usage_percent': m.memory_usage_percent,
                        'memory_available_gb': m.memory_available_gb,
                        'disk_usage_percent': m.disk_usage_percent,
                        'disk_available_gb': m.disk_available_gb,
                        'gpu_usage_percent': m.gpu_usage_percent,
                        'process_count': m.process_count
                    }
                    for m in self.metrics_history
                ],
                'service_statuses': {
                    name: {
                        'status': status.status,
                        'response_time_ms': status.response_time_ms,
                        'last_check': status.last_check.isoformat(),
                        'error_message': status.error_message
                    }
                    for name, status in self.service_statuses.items()
                },
                'alerts_history': [
                    {
                        'id': alert.id,
                        'severity': alert.severity,
                        'component': alert.component,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat(),
                        'resolved': alert.resolved,
                        'resolution_time': alert.resolution_time.isoformat() if alert.resolution_time else None
                    }
                    for alert in self.active_alerts.values()
                ]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            return {
                'success': True,
                'filepath': filepath,
                'metrics_exported': len(self.metrics_history),
                'file_size_kb': os.path.getsize(filepath) // 1024
            }
            
        except Exception as e:
            logger.error(f"Error exporting monitoring data: {str(e)}")
            return {'error': f"Export failed: {str(e)}"}
