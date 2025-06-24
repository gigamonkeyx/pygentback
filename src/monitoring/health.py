#!/usr/bin/env python3
"""
A2A System Health Monitoring

Comprehensive health monitoring for A2A multi-agent system components.
"""

import asyncio
import logging
import psutil
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Individual health check result"""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    duration: float


class A2AHealthMonitor:
    """A2A System Health Monitor"""
    
    def __init__(self, db_manager=None, redis_manager=None, a2a_manager=None):
        self.db_manager = db_manager
        self.redis_manager = redis_manager
        self.a2a_manager = a2a_manager
        self.health_history: List[Dict[str, Any]] = []
        self.last_health_check = None
        
    async def run_comprehensive_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check of all system components"""
        
        start_time = datetime.utcnow()
        checks = []
        
        # System resource checks
        checks.append(await self._check_system_resources())
        checks.append(await self._check_memory_usage())
        checks.append(await self._check_disk_space())
        
        # Infrastructure checks
        if self.db_manager:
            checks.append(await self._check_database_health())
        if self.redis_manager:
            checks.append(await self._check_redis_health())
        
        # A2A system checks
        if self.a2a_manager:
            checks.append(await self._check_a2a_system())
            checks.append(await self._check_agent_health())
            checks.append(await self._check_task_processing())
        
        # Network connectivity checks
        checks.append(await self._check_network_connectivity())
        
        # Calculate overall health
        overall_status = self._calculate_overall_health(checks)
        
        health_report = {
            "overall_status": overall_status.value,
            "timestamp": start_time.isoformat(),
            "duration": (datetime.utcnow() - start_time).total_seconds(),
            "checks": [
                {
                    "name": check.name,
                    "status": check.status.value,
                    "message": check.message,
                    "details": check.details,
                    "duration": check.duration
                }
                for check in checks
            ],
            "summary": self._generate_health_summary(checks)
        }
        
        # Store in history
        self.health_history.append(health_report)
        if len(self.health_history) > 100:  # Keep last 100 checks
            self.health_history = self.health_history[-100:]
        
        self.last_health_check = health_report
        
        return health_report
    
    async def _check_system_resources(self) -> HealthCheck:
        """Check system CPU and load"""
        start_time = datetime.utcnow()
        
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
            cpu_count = psutil.cpu_count()
            
            details = {
                "cpu_percent": cpu_percent,
                "load_1min": load_avg[0],
                "load_5min": load_avg[1],
                "load_15min": load_avg[2],
                "cpu_count": cpu_count
            }
            
            # Determine status
            if cpu_percent > 90 or load_avg[0] > cpu_count * 2:
                status = HealthStatus.CRITICAL
                message = f"High CPU usage: {cpu_percent:.1f}%, load: {load_avg[0]:.2f}"
            elif cpu_percent > 80 or load_avg[0] > cpu_count * 1.5:
                status = HealthStatus.UNHEALTHY
                message = f"Elevated CPU usage: {cpu_percent:.1f}%, load: {load_avg[0]:.2f}"
            elif cpu_percent > 70 or load_avg[0] > cpu_count:
                status = HealthStatus.DEGRADED
                message = f"Moderate CPU usage: {cpu_percent:.1f}%, load: {load_avg[0]:.2f}"
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU usage normal: {cpu_percent:.1f}%, load: {load_avg[0]:.2f}"
            
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Failed to check system resources: {e}"
            details = {"error": str(e)}
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        return HealthCheck("system_resources", status, message, details, start_time, duration)
    
    async def _check_memory_usage(self) -> HealthCheck:
        """Check system memory usage"""
        start_time = datetime.utcnow()
        
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            details = {
                "memory_total": memory.total,
                "memory_available": memory.available,
                "memory_percent": memory.percent,
                "swap_total": swap.total,
                "swap_used": swap.used,
                "swap_percent": swap.percent
            }
            
            # Determine status
            if memory.percent > 95 or swap.percent > 80:
                status = HealthStatus.CRITICAL
                message = f"Critical memory usage: {memory.percent:.1f}%, swap: {swap.percent:.1f}%"
            elif memory.percent > 90 or swap.percent > 60:
                status = HealthStatus.UNHEALTHY
                message = f"High memory usage: {memory.percent:.1f}%, swap: {swap.percent:.1f}%"
            elif memory.percent > 80 or swap.percent > 40:
                status = HealthStatus.DEGRADED
                message = f"Elevated memory usage: {memory.percent:.1f}%, swap: {swap.percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory.percent:.1f}%, swap: {swap.percent:.1f}%"
            
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Failed to check memory usage: {e}"
            details = {"error": str(e)}
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        return HealthCheck("memory_usage", status, message, details, start_time, duration)
    
    async def _check_disk_space(self) -> HealthCheck:
        """Check disk space usage"""
        start_time = datetime.utcnow()
        
        try:
            disk_usage = psutil.disk_usage('/')
            
            details = {
                "disk_total": disk_usage.total,
                "disk_used": disk_usage.used,
                "disk_free": disk_usage.free,
                "disk_percent": (disk_usage.used / disk_usage.total) * 100
            }
            
            disk_percent = details["disk_percent"]
            
            # Determine status
            if disk_percent > 95:
                status = HealthStatus.CRITICAL
                message = f"Critical disk usage: {disk_percent:.1f}%"
            elif disk_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"High disk usage: {disk_percent:.1f}%"
            elif disk_percent > 80:
                status = HealthStatus.DEGRADED
                message = f"Elevated disk usage: {disk_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage normal: {disk_percent:.1f}%"
            
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Failed to check disk space: {e}"
            details = {"error": str(e)}
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        return HealthCheck("disk_space", status, message, details, start_time, duration)
    
    async def _check_database_health(self) -> HealthCheck:
        """Check database connectivity and performance"""
        start_time = datetime.utcnow()
        
        try:
            # Test basic connectivity
            health = await self.db_manager.health_check()
            
            if health:
                # Test query performance
                query_start = datetime.utcnow()
                await self.db_manager.fetch_all("SELECT 1")
                query_duration = (datetime.utcnow() - query_start).total_seconds()
                
                details = {
                    "connected": True,
                    "query_duration": query_duration,
                    "pool_size": getattr(self.db_manager, 'pool_size', 'unknown')
                }
                
                if query_duration > 1.0:
                    status = HealthStatus.DEGRADED
                    message = f"Database slow: {query_duration:.3f}s query time"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Database healthy: {query_duration:.3f}s query time"
            else:
                status = HealthStatus.UNHEALTHY
                message = "Database health check failed"
                details = {"connected": False}
            
        except Exception as e:
            status = HealthStatus.CRITICAL
            message = f"Database connection failed: {e}"
            details = {"error": str(e), "connected": False}
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        return HealthCheck("database", status, message, details, start_time, duration)
    
    async def _check_redis_health(self) -> HealthCheck:
        """Check Redis connectivity and performance"""
        start_time = datetime.utcnow()
        
        try:
            # Test basic connectivity
            health = await self.redis_manager.health_check()
            
            if health:
                # Test operation performance
                op_start = datetime.utcnow()
                await self.redis_manager.set("health_check", "test", expire=10)
                value = await self.redis_manager.get("health_check")
                op_duration = (datetime.utcnow() - op_start).total_seconds()
                
                details = {
                    "connected": True,
                    "operation_duration": op_duration,
                    "test_value_match": value == "test"
                }
                
                if op_duration > 0.1:
                    status = HealthStatus.DEGRADED
                    message = f"Redis slow: {op_duration:.3f}s operation time"
                elif not details["test_value_match"]:
                    status = HealthStatus.UNHEALTHY
                    message = "Redis data integrity issue"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Redis healthy: {op_duration:.3f}s operation time"
            else:
                status = HealthStatus.UNHEALTHY
                message = "Redis health check failed"
                details = {"connected": False}
            
        except Exception as e:
            status = HealthStatus.CRITICAL
            message = f"Redis connection failed: {e}"
            details = {"error": str(e), "connected": False}
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        return HealthCheck("redis", status, message, details, start_time, duration)
    
    async def _check_a2a_system(self) -> HealthCheck:
        """Check A2A system health"""
        start_time = datetime.utcnow()
        
        try:
            if not self.a2a_manager.initialized:
                status = HealthStatus.CRITICAL
                message = "A2A manager not initialized"
                details = {"initialized": False}
            else:
                agent_status = await self.a2a_manager.get_agent_status()
                
                details = {
                    "initialized": True,
                    "total_agents": agent_status.get("total_agents", 0),
                    "active_tasks": agent_status.get("active_tasks", 0)
                }
                
                if details["total_agents"] == 0:
                    status = HealthStatus.UNHEALTHY
                    message = "No agents registered"
                elif details["active_tasks"] > 100:
                    status = HealthStatus.DEGRADED
                    message = f"High task load: {details['active_tasks']} active tasks"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"A2A system healthy: {details['total_agents']} agents, {details['active_tasks']} tasks"
            
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"A2A system check failed: {e}"
            details = {"error": str(e)}
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        return HealthCheck("a2a_system", status, message, details, start_time, duration)
    
    async def _check_agent_health(self) -> HealthCheck:
        """Check individual agent health"""
        start_time = datetime.utcnow()
        
        try:
            agents = await self.a2a_manager.a2a_agent_registry.list_agents()
            
            agent_details = []
            healthy_agents = 0
            
            for agent_wrapper in agents:
                agent = agent_wrapper.agent
                agent_health = {
                    "agent_id": agent.agent_id,
                    "name": agent.name,
                    "type": str(agent.agent_type),
                    "status": str(agent.status),
                    "capabilities": len(agent.capabilities)
                }
                
                if "IDLE" in str(agent.status) or "WORKING" in str(agent.status):
                    healthy_agents += 1
                    agent_health["healthy"] = True
                else:
                    agent_health["healthy"] = False
                
                agent_details.append(agent_health)
            
            details = {
                "total_agents": len(agents),
                "healthy_agents": healthy_agents,
                "agents": agent_details
            }
            
            if len(agents) == 0:
                status = HealthStatus.CRITICAL
                message = "No agents available"
            elif healthy_agents == 0:
                status = HealthStatus.CRITICAL
                message = "No healthy agents"
            elif healthy_agents < len(agents) * 0.5:
                status = HealthStatus.UNHEALTHY
                message = f"Only {healthy_agents}/{len(agents)} agents healthy"
            elif healthy_agents < len(agents):
                status = HealthStatus.DEGRADED
                message = f"{healthy_agents}/{len(agents)} agents healthy"
            else:
                status = HealthStatus.HEALTHY
                message = f"All {healthy_agents} agents healthy"
            
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Agent health check failed: {e}"
            details = {"error": str(e)}
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        return HealthCheck("agent_health", status, message, details, start_time, duration)
    
    async def _check_task_processing(self) -> HealthCheck:
        """Check task processing performance"""
        start_time = datetime.utcnow()
        
        try:
            # Get recent task metrics from A2A protocol
            task_count = len(self.a2a_manager.a2a_protocol.tasks)
            
            # Check for stuck tasks (tasks older than 5 minutes)
            stuck_tasks = 0
            cutoff = datetime.utcnow() - timedelta(minutes=5)
            
            for task in self.a2a_manager.a2a_protocol.tasks.values():
                if task.status.timestamp:
                    task_time = datetime.fromisoformat(task.status.timestamp.replace('Z', '+00:00'))
                    if task_time < cutoff and task.status.state in ["submitted", "working"]:
                        stuck_tasks += 1
            
            details = {
                "total_tasks": task_count,
                "stuck_tasks": stuck_tasks,
                "task_processing_rate": "unknown"  # Would need historical data
            }
            
            if stuck_tasks > 10:
                status = HealthStatus.UNHEALTHY
                message = f"Many stuck tasks: {stuck_tasks}"
            elif stuck_tasks > 5:
                status = HealthStatus.DEGRADED
                message = f"Some stuck tasks: {stuck_tasks}"
            elif task_count > 200:
                status = HealthStatus.DEGRADED
                message = f"High task backlog: {task_count}"
            else:
                status = HealthStatus.HEALTHY
                message = f"Task processing normal: {task_count} total, {stuck_tasks} stuck"
            
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Task processing check failed: {e}"
            details = {"error": str(e)}
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        return HealthCheck("task_processing", status, message, details, start_time, duration)
    
    async def _check_network_connectivity(self) -> HealthCheck:
        """Check network connectivity"""
        start_time = datetime.utcnow()
        
        try:
            # Test local connectivity (simplified)
            import socket
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(('127.0.0.1', 8080))
            sock.close()
            
            details = {
                "local_port_8080": result == 0,
                "test_timestamp": start_time.isoformat()
            }
            
            if result == 0:
                status = HealthStatus.HEALTHY
                message = "Network connectivity normal"
            else:
                status = HealthStatus.UNHEALTHY
                message = "Local port 8080 not accessible"
            
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Network connectivity check failed: {e}"
            details = {"error": str(e)}
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        return HealthCheck("network_connectivity", status, message, details, start_time, duration)
    
    def _calculate_overall_health(self, checks: List[HealthCheck]) -> HealthStatus:
        """Calculate overall system health from individual checks"""
        if not checks:
            return HealthStatus.UNHEALTHY
        
        # Count status levels
        status_counts = {status: 0 for status in HealthStatus}
        for check in checks:
            status_counts[check.status] += 1
        
        total_checks = len(checks)
        
        # Determine overall status
        if status_counts[HealthStatus.CRITICAL] > 0:
            return HealthStatus.CRITICAL
        elif status_counts[HealthStatus.UNHEALTHY] > total_checks * 0.3:
            return HealthStatus.UNHEALTHY
        elif status_counts[HealthStatus.DEGRADED] > total_checks * 0.5:
            return HealthStatus.DEGRADED
        elif status_counts[HealthStatus.UNHEALTHY] > 0 or status_counts[HealthStatus.DEGRADED] > 0:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def _generate_health_summary(self, checks: List[HealthCheck]) -> Dict[str, Any]:
        """Generate health summary statistics"""
        status_counts = {status.value: 0 for status in HealthStatus}
        total_duration = 0
        
        for check in checks:
            status_counts[check.status.value] += 1
            total_duration += check.duration
        
        return {
            "total_checks": len(checks),
            "status_distribution": status_counts,
            "average_check_duration": total_duration / len(checks) if checks else 0,
            "total_check_duration": total_duration
        }
    
    def get_health_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get health check history for specified hours"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        return [
            report for report in self.health_history
            if datetime.fromisoformat(report["timestamp"]) > cutoff
        ]


# Global health monitor instance
health_monitor = A2AHealthMonitor()
