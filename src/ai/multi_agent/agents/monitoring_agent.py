"""
Monitoring Agent

Specialized agent for system monitoring, performance tracking,
and health assessment in multi-agent environments.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricData:
    """Metric data point"""
    metric_name: str
    value: float
    timestamp: datetime
    unit: str = ""
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class Alert:
    """System alert"""
    alert_id: str
    level: AlertLevel
    message: str
    source: str
    timestamp: datetime
    resolved: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MonitoringAgent:
    """
    Agent specialized in system monitoring and health assessment.
    
    Capabilities:
    - Performance metric collection
    - Health status monitoring
    - Alert generation and management
    - Trend analysis and reporting
    - Resource utilization tracking
    """
    
    def __init__(self, agent_id: str = "monitoring_agent"):
        self.agent_id = agent_id
        self.agent_type = "monitoring"
        self.status = "initialized"
        self.capabilities = [
            "metric_collection",
            "health_monitoring",
            "alert_management",
            "trend_analysis",
            "resource_tracking"
        ]
        
        # Monitoring state
        self.metrics: Dict[str, List[MetricData]] = {}
        self.alerts: Dict[str, Alert] = {}
        self.monitored_components: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.config = {
            'metric_retention_hours': 24,
            'alert_retention_hours': 72,
            'collection_interval_seconds': 30,
            'health_check_interval_seconds': 60,
            'max_metrics_per_type': 1000,
            'alert_thresholds': {
                'cpu_usage': 80.0,
                'memory_usage': 85.0,
                'error_rate': 5.0,
                'response_time_ms': 5000.0
            }
        }
        
        # Statistics
        self.stats = {
            'metrics_collected': 0,
            'alerts_generated': 0,
            'components_monitored': 0,
            'health_checks_performed': 0,
            'uptime_seconds': 0,
            'last_collection_time': None
        }
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_time: Optional[datetime] = None
        
        logger.info(f"MonitoringAgent {agent_id} initialized")
    
    async def start(self) -> bool:
        """Start the monitoring agent"""
        try:
            self.status = "active"
            self._start_time = datetime.utcnow()
            
            # Start background monitoring tasks
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info(f"MonitoringAgent {self.agent_id} started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start MonitoringAgent {self.agent_id}: {e}")
            self.status = "error"
            return False
    
    async def stop(self) -> bool:
        """Stop the monitoring agent"""
        try:
            self.status = "stopping"
            
            # Cancel background tasks
            if self._monitoring_task:
                self._monitoring_task.cancel()
            if self._cleanup_task:
                self._cleanup_task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(
                self._monitoring_task, self._cleanup_task,
                return_exceptions=True
            )
            
            self.status = "stopped"
            logger.info(f"MonitoringAgent {self.agent_id} stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop MonitoringAgent {self.agent_id}: {e}")
            return False
    
    async def monitor_system(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Monitor system performance and health.
        
        Args:
            system_data: System metrics and status data
            
        Returns:
            Monitoring results with health assessment
        """
        start_time = datetime.utcnow()
        
        try:
            # Collect metrics from system data
            metrics_collected = await self._collect_metrics(system_data)
            
            # Perform health assessment
            health_assessment = await self._assess_system_health(system_data)
            
            # Check for alerts
            alerts_generated = await self._check_alert_conditions(system_data)
            
            # Update statistics
            self.stats['metrics_collected'] += metrics_collected
            self.stats['alerts_generated'] += len(alerts_generated)
            self.stats['health_checks_performed'] += 1
            self.stats['last_collection_time'] = start_time.isoformat()
            
            monitoring_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            result = {
                'monitoring_id': f"monitor_{int(start_time.timestamp())}",
                'success': True,
                'metrics_collected': metrics_collected,
                'health_status': health_assessment['overall_health'],
                'health_score': health_assessment['health_score'],
                'alerts_generated': len(alerts_generated),
                'new_alerts': [alert.alert_id for alert in alerts_generated],
                'monitoring_time_ms': monitoring_time,
                'timestamp': start_time.isoformat()
            }
            
            logger.debug(f"System monitoring completed in {monitoring_time:.2f}ms")
            return result
            
        except Exception as e:
            monitoring_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.error(f"System monitoring failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'monitoring_time_ms': monitoring_time,
                'timestamp': start_time.isoformat()
            }
    
    async def collect_metrics(self, metrics_data: Dict[str, float]) -> bool:
        """Collect and store metrics"""
        try:
            timestamp = datetime.utcnow()
            
            for metric_name, value in metrics_data.items():
                metric = MetricData(
                    metric_name=metric_name,
                    value=value,
                    timestamp=timestamp,
                    unit=self._get_metric_unit(metric_name)
                )
                
                if metric_name not in self.metrics:
                    self.metrics[metric_name] = []
                
                self.metrics[metric_name].append(metric)
                
                # Limit metric history
                max_metrics = self.config['max_metrics_per_type']
                if len(self.metrics[metric_name]) > max_metrics:
                    self.metrics[metric_name] = self.metrics[metric_name][-max_metrics:]
            
            self.stats['metrics_collected'] += len(metrics_data)
            return True
            
        except Exception as e:
            logger.error(f"Metric collection failed: {e}")
            return False
    
    async def generate_alert(self, level: AlertLevel, message: str, source: str, 
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate a system alert"""
        try:
            alert_id = f"alert_{int(datetime.utcnow().timestamp())}_{len(self.alerts)}"
            
            alert = Alert(
                alert_id=alert_id,
                level=level,
                message=message,
                source=source,
                timestamp=datetime.utcnow(),
                metadata=metadata or {}
            )
            
            self.alerts[alert_id] = alert
            self.stats['alerts_generated'] += 1
            
            logger.info(f"Generated {level.value} alert: {message}")
            return alert_id
            
        except Exception as e:
            logger.error(f"Alert generation failed: {e}")
            return ""
    
    def get_metrics(self, metric_name: Optional[str] = None, 
                   hours_back: int = 1) -> Dict[str, List[Dict]]:
        """Get collected metrics"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
            result = {}
            
            metrics_to_check = [metric_name] if metric_name else self.metrics.keys()
            
            for name in metrics_to_check:
                if name in self.metrics:
                    recent_metrics = [
                        {
                            'value': m.value,
                            'timestamp': m.timestamp.isoformat(),
                            'unit': m.unit,
                            'tags': m.tags
                        }
                        for m in self.metrics[name]
                        if m.timestamp >= cutoff_time
                    ]
                    result[name] = recent_metrics
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {}
    
    def get_alerts(self, level: Optional[AlertLevel] = None, 
                  unresolved_only: bool = True) -> List[Dict]:
        """Get system alerts"""
        try:
            alerts = []
            
            for alert in self.alerts.values():
                if unresolved_only and alert.resolved:
                    continue
                
                if level and alert.level != level:
                    continue
                
                alerts.append({
                    'alert_id': alert.alert_id,
                    'level': alert.level.value,
                    'message': alert.message,
                    'source': alert.source,
                    'timestamp': alert.timestamp.isoformat(),
                    'resolved': alert.resolved,
                    'metadata': alert.metadata
                })
            
            # Sort by timestamp (newest first)
            alerts.sort(key=lambda x: x['timestamp'], reverse=True)
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            return []
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.status == "active":
            try:
                # Update uptime
                if self._start_time:
                    self.stats['uptime_seconds'] = (datetime.utcnow() - self._start_time).total_seconds()
                
                # Perform periodic health checks
                await self._perform_health_checks()
                
                # Wait for next iteration
                await asyncio.sleep(self.config['collection_interval_seconds'])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(10.0)
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while self.status == "active":
            try:
                # Clean up old metrics
                await self._cleanup_old_metrics()
                
                # Clean up old alerts
                await self._cleanup_old_alerts()
                
                # Wait for next cleanup
                await asyncio.sleep(3600)  # Cleanup every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _collect_metrics(self, system_data: Dict[str, Any]) -> int:
        """Collect metrics from system data"""
        metrics_count = 0
        
        # Extract common system metrics
        metrics_to_collect = {
            'cpu_usage': system_data.get('cpu_usage', 0.0),
            'memory_usage': system_data.get('memory_usage', 0.0),
            'disk_usage': system_data.get('disk_usage', 0.0),
            'network_usage': system_data.get('network_usage', 0.0),
            'response_time_ms': system_data.get('response_time_ms', 0.0),
            'error_rate': system_data.get('error_rate', 0.0),
            'throughput': system_data.get('throughput', 0.0)
        }
        
        # Filter out None values
        valid_metrics = {k: v for k, v in metrics_to_collect.items() if v is not None}
        
        if valid_metrics:
            await self.collect_metrics(valid_metrics)
            metrics_count = len(valid_metrics)
        
        return metrics_count
    
    async def _assess_system_health(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall system health"""
        health_factors = []
        
        # CPU health
        cpu_usage = system_data.get('cpu_usage', 0.0)
        cpu_health = max(0.0, 1.0 - (cpu_usage / 100.0))
        health_factors.append(cpu_health)
        
        # Memory health
        memory_usage = system_data.get('memory_usage', 0.0)
        memory_health = max(0.0, 1.0 - (memory_usage / 100.0))
        health_factors.append(memory_health)
        
        # Response time health
        response_time = system_data.get('response_time_ms', 0.0)
        response_health = max(0.0, 1.0 - (response_time / 5000.0))  # 5 seconds max
        health_factors.append(response_health)
        
        # Error rate health
        error_rate = system_data.get('error_rate', 0.0)
        error_health = max(0.0, 1.0 - (error_rate / 10.0))  # 10% max
        health_factors.append(error_health)
        
        # Calculate overall health score
        overall_score = sum(health_factors) / len(health_factors) if health_factors else 0.5
        
        # Determine health status
        if overall_score >= 0.8:
            health_status = "healthy"
        elif overall_score >= 0.6:
            health_status = "warning"
        elif overall_score >= 0.4:
            health_status = "degraded"
        else:
            health_status = "unhealthy"
        
        return {
            'overall_health': health_status,
            'health_score': overall_score,
            'health_factors': {
                'cpu_health': cpu_health,
                'memory_health': memory_health,
                'response_health': response_health,
                'error_health': error_health
            }
        }
    
    async def _check_alert_conditions(self, system_data: Dict[str, Any]) -> List[Alert]:
        """Check for alert conditions"""
        alerts_generated = []
        thresholds = self.config['alert_thresholds']
        
        # Check CPU usage
        cpu_usage = system_data.get('cpu_usage', 0.0)
        if cpu_usage > thresholds['cpu_usage']:
            alert_id = await self.generate_alert(
                AlertLevel.WARNING,
                f"High CPU usage: {cpu_usage:.1f}%",
                "system_monitor",
                {'metric': 'cpu_usage', 'value': cpu_usage}
            )
            if alert_id:
                alerts_generated.append(self.alerts[alert_id])
        
        # Check memory usage
        memory_usage = system_data.get('memory_usage', 0.0)
        if memory_usage > thresholds['memory_usage']:
            alert_id = await self.generate_alert(
                AlertLevel.WARNING,
                f"High memory usage: {memory_usage:.1f}%",
                "system_monitor",
                {'metric': 'memory_usage', 'value': memory_usage}
            )
            if alert_id:
                alerts_generated.append(self.alerts[alert_id])
        
        # Check error rate
        error_rate = system_data.get('error_rate', 0.0)
        if error_rate > thresholds['error_rate']:
            level = AlertLevel.CRITICAL if error_rate > 10.0 else AlertLevel.ERROR
            alert_id = await self.generate_alert(
                level,
                f"High error rate: {error_rate:.1f}%",
                "system_monitor",
                {'metric': 'error_rate', 'value': error_rate}
            )
            if alert_id:
                alerts_generated.append(self.alerts[alert_id])
        
        # Check response time
        response_time = system_data.get('response_time_ms', 0.0)
        if response_time > thresholds['response_time_ms']:
            alert_id = await self.generate_alert(
                AlertLevel.WARNING,
                f"High response time: {response_time:.0f}ms",
                "system_monitor",
                {'metric': 'response_time_ms', 'value': response_time}
            )
            if alert_id:
                alerts_generated.append(self.alerts[alert_id])
        
        return alerts_generated
    
    async def _perform_health_checks(self):
        """Perform periodic health checks"""
        try:
            # Check monitored components
            for component_id, component_info in self.monitored_components.items():
                # Simulate health check
                health_score = 0.9  # Placeholder
                component_info['last_health_check'] = datetime.utcnow()
                component_info['health_score'] = health_score
            
            self.stats['health_checks_performed'] += 1
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    async def _cleanup_old_metrics(self):
        """Clean up old metrics"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=self.config['metric_retention_hours'])
            
            for metric_name in list(self.metrics.keys()):
                self.metrics[metric_name] = [
                    m for m in self.metrics[metric_name]
                    if m.timestamp >= cutoff_time
                ]
                
                # Remove empty metric lists
                if not self.metrics[metric_name]:
                    del self.metrics[metric_name]
            
        except Exception as e:
            logger.error(f"Metric cleanup failed: {e}")
    
    async def _cleanup_old_alerts(self):
        """Clean up old resolved alerts"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=self.config['alert_retention_hours'])
            
            old_alert_ids = [
                alert_id for alert_id, alert in self.alerts.items()
                if alert.resolved and alert.timestamp < cutoff_time
            ]
            
            for alert_id in old_alert_ids:
                del self.alerts[alert_id]
            
        except Exception as e:
            logger.error(f"Alert cleanup failed: {e}")
    
    def _get_metric_unit(self, metric_name: str) -> str:
        """Get unit for metric"""
        unit_map = {
            'cpu_usage': '%',
            'memory_usage': '%',
            'disk_usage': '%',
            'network_usage': 'Mbps',
            'response_time_ms': 'ms',
            'error_rate': '%',
            'throughput': 'req/s'
        }
        return unit_map.get(metric_name, '')
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitoring agent status"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'status': self.status,
            'capabilities': self.capabilities,
            'metrics_types': len(self.metrics),
            'total_metrics': sum(len(metrics) for metrics in self.metrics.values()),
            'active_alerts': len([a for a in self.alerts.values() if not a.resolved]),
            'total_alerts': len(self.alerts),
            'monitored_components': len(self.monitored_components),
            'statistics': self.stats.copy(),
            'config': self.config.copy()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            'agent_id': self.agent_id,
            'status': self.status,
            'is_healthy': self.status == "active",
            'uptime_seconds': self.stats['uptime_seconds'],
            'metrics_collected': self.stats['metrics_collected'],
            'alerts_generated': self.stats['alerts_generated'],
            'health_checks_performed': self.stats['health_checks_performed'],
            'last_collection': self.stats['last_collection_time'],
            'last_check': datetime.utcnow().isoformat()
        }
