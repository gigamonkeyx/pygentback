"""
Health Monitor

Monitors the health and performance of the documentation system
with intelligent alerting and self-healing capabilities.
"""

import asyncio
import logging
import time
import psutil
import aiohttp
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from .documentation_models import (
    DocumentationTask, DocumentationConfig, HealthCheckResult,
    DocumentationEventType
)
# Handle integration events gracefully
try:
    from ..integration.events import EventBus, Event
    INTEGRATION_EVENTS_AVAILABLE = True
except ImportError:
    EventBus = None
    Event = None
    INTEGRATION_EVENTS_AVAILABLE = False


logger = logging.getLogger(__name__)

class HealthMonitor:
    """
    Monitors documentation system health with intelligent alerting.
    
    Provides continuous monitoring of build processes, file integrity,
    server availability, and performance metrics.
    """
    
    def __init__(self, config: DocumentationConfig, event_bus):
        self.config = config
        self.event_bus = event_bus
        if self.event_bus is None:
            raise RuntimeError("EventBus is required for HealthMonitor")
        self.is_running = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.health_history: List[HealthCheckResult] = []
        self.alerts_sent: List[Dict[str, Any]] = []
        
        # Health thresholds
        self.thresholds = {
            "max_build_time": 300,  # 5 minutes
            "max_sync_time": 60,    # 1 minute
            "min_disk_space_gb": 1.0,
            "max_memory_usage_percent": 80,
            "max_response_time_ms": 5000
        }
        
        # Health check intervals
        self.check_intervals = {
            "system_resources": 30,    # seconds
            "file_integrity": 300,     # 5 minutes
            "server_availability": 60, # 1 minute
            "performance_metrics": 120 # 2 minutes
        }
    
    async def start(self):
        """Start the health monitor"""
        self.is_running = True
        
        # Start monitoring task
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Health Monitor started")
    
    async def stop(self):
        """Stop the health monitor"""
        self.is_running = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Health Monitor stopped")
    
    async def start_monitoring(self, task: DocumentationTask) -> Dict[str, Any]:
        """Start monitoring for a specific task"""
        monitoring_result = {
            "monitoring_started": True,
            "checks_enabled": list(self.check_intervals.keys()),
            "thresholds": self.thresholds.copy()
        }
        
        # Perform initial health check
        health_result = await self.perform_health_check()
        monitoring_result["initial_health"] = health_result.__dict__
        
        return monitoring_result
    
    async def perform_health_check(self) -> HealthCheckResult:
        """Perform comprehensive health check"""
        checks_performed = []
        issues_found = []
        recommendations = []
        
        try:
            # Check system resources
            resource_issues = await self._check_system_resources()
            checks_performed.append("system_resources")
            issues_found.extend(resource_issues)
            
            # Check file integrity
            file_issues = await self._check_file_integrity()
            checks_performed.append("file_integrity")
            issues_found.extend(file_issues)
            
            # Check server availability
            server_issues = await self._check_server_availability()
            checks_performed.append("server_availability")
            issues_found.extend(server_issues)
            
            # Check performance metrics
            performance_issues = await self._check_performance_metrics()
            checks_performed.append("performance_metrics")
            issues_found.extend(performance_issues)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(issues_found)
            
            # Determine overall health
            overall_health = self._calculate_overall_health(issues_found)
            
            health_result = HealthCheckResult(
                overall_health=overall_health,
                checks_performed=checks_performed,
                issues_found=issues_found,
                recommendations=recommendations
            )
            
            # Store in history
            self.health_history.append(health_result)
            
            # Keep only last 100 health checks
            if len(self.health_history) > 100:
                self.health_history = self.health_history[-100:]
            
            # Emit health check event
            await self.event_bus.publish_event(
                event_type="health_check_completed",
                source="health_monitor",
                data={
                    "overall_health": overall_health,
                    "issues_count": len(issues_found),
                    "checks_performed": len(checks_performed)
                }
            )
            
            return health_result
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthCheckResult(
                overall_health="critical",
                checks_performed=checks_performed,
                issues_found=[f"Health check error: {str(e)}"],
                recommendations=["Investigate health monitoring system"]
            )
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        last_checks = {check: 0 for check in self.check_intervals}
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Check if any health checks are due
                for check_type, interval in self.check_intervals.items():
                    if current_time - last_checks[check_type] >= interval:
                        await self._perform_specific_check(check_type)
                        last_checks[check_type] = current_time
                
                # Sleep for a short interval
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(10)
    
    async def _perform_specific_check(self, check_type: str):
        """Perform a specific type of health check"""
        try:
            if check_type == "system_resources":
                issues = await self._check_system_resources()
            elif check_type == "file_integrity":
                issues = await self._check_file_integrity()
            elif check_type == "server_availability":
                issues = await self._check_server_availability()
            elif check_type == "performance_metrics":
                issues = await self._check_performance_metrics()
            else:
                return
            
            # Send alerts for critical issues
            for issue in issues:
                if "critical" in issue.lower() or "error" in issue.lower():
                    await self._send_alert(check_type, issue)
                    
        except Exception as e:
            logger.error(f"Specific health check failed ({check_type}): {e}")
    
    async def _check_system_resources(self) -> List[str]:
        """Check system resource usage"""
        issues = []
        
        try:
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > self.thresholds["max_memory_usage_percent"]:
                issues.append(f"High memory usage: {memory.percent:.1f}%")
            
            # Check disk space
            disk = psutil.disk_usage(str(self.config.docs_source_path))
            free_gb = disk.free / (1024**3)
            if free_gb < self.thresholds["min_disk_space_gb"]:
                issues.append(f"Low disk space: {free_gb:.1f}GB remaining")
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
                
        except Exception as e:
            issues.append(f"System resource check error: {str(e)}")
        
        return issues
    
    async def _check_file_integrity(self) -> List[str]:
        """Check file integrity and structure"""
        issues = []
        
        try:
            # Check if source files exist
            required_files = [
                self.config.docs_source_path / "package.json",
                self.config.docs_source_path / ".vitepress" / "config.ts",
                self.config.docs_source_path / "index.md"
            ]
            
            for file_path in required_files:
                if not file_path.exists():
                    issues.append(f"Missing required file: {file_path}")
            
            # Check if build output exists and is recent
            if self.config.docs_build_path.exists():
                index_file = self.config.docs_build_path / "index.html"
                if index_file.exists():
                    # Check if build is older than 24 hours
                    file_age = time.time() - index_file.stat().st_mtime
                    if file_age > 86400:  # 24 hours
                        issues.append("Build output is older than 24 hours")
                else:
                    issues.append("Build output missing index.html")
            else:
                issues.append("Build output directory does not exist")
            
            # Check frontend sync integrity
            if self.config.frontend_docs_path.exists():
                manifest_file = self.config.frontend_docs_path / "manifest.json"
                if not manifest_file.exists():
                    issues.append("Frontend sync missing manifest.json")
            else:
                issues.append("Frontend documentation directory does not exist")
                
        except Exception as e:
            issues.append(f"File integrity check error: {str(e)}")
        
        return issues
    
    async def _check_server_availability(self) -> List[str]:
        """Check server availability and response times"""
        issues = []
        
        try:
            # Check VitePress dev server if it should be running
            dev_server_url = f"http://localhost:{self.config.vitepress_port}"
            
            async with aiohttp.ClientSession() as session:
                try:
                    start_time = time.time()
                    async with session.get(dev_server_url, timeout=5) as response:
                        response_time = (time.time() - start_time) * 1000
                        
                        if response.status != 200:
                            issues.append(f"Dev server returned status {response.status}")
                        
                        if response_time > self.thresholds["max_response_time_ms"]:
                            issues.append(f"Slow dev server response: {response_time:.0f}ms")
                            
                except asyncio.TimeoutError:
                    issues.append("Dev server timeout")
                except aiohttp.ClientConnectorError:
                    # Dev server not running - this might be expected
                    pass
            
            # Check frontend documentation availability
            frontend_url = "http://localhost:3000/docs/"
            
            async with aiohttp.ClientSession() as session:
                try:
                    start_time = time.time()
                    async with session.get(frontend_url, timeout=5) as response:
                        response_time = (time.time() - start_time) * 1000
                        
                        if response.status != 200:
                            issues.append(f"Frontend docs returned status {response.status}")
                        
                        if response_time > self.thresholds["max_response_time_ms"]:
                            issues.append(f"Slow frontend docs response: {response_time:.0f}ms")
                            
                except (asyncio.TimeoutError, aiohttp.ClientConnectorError):
                    issues.append("Frontend documentation not accessible")
                    
        except Exception as e:
            issues.append(f"Server availability check error: {str(e)}")
        
        return issues
    
    async def _check_performance_metrics(self) -> List[str]:
        """Check performance metrics"""
        issues = []
        
        try:
            # Check recent build times from history
            if len(self.health_history) >= 5:
                recent_checks = self.health_history[-5:]
                build_issues = [issue for check in recent_checks for issue in check.issues_found if "build" in issue.lower()]
                
                if len(build_issues) > 2:
                    issues.append("Frequent build issues detected")
            
            # Check file sizes
            if self.config.docs_build_path.exists():
                total_size = sum(f.stat().st_size for f in self.config.docs_build_path.rglob("*") if f.is_file())
                size_mb = total_size / (1024**2)
                
                if size_mb > 100:  # 100MB threshold
                    issues.append(f"Large build output size: {size_mb:.1f}MB")
                    
        except Exception as e:
            issues.append(f"Performance metrics check error: {str(e)}")
        
        return issues
    
    async def _generate_recommendations(self, issues: List[str]) -> List[str]:
        """Generate recommendations based on issues found"""
        recommendations = []
        
        for issue in issues:
            if "memory" in issue.lower():
                recommendations.append("Consider increasing system memory or optimizing build process")
            elif "disk space" in issue.lower():
                recommendations.append("Clean up old build artifacts and temporary files")
            elif "build" in issue.lower():
                recommendations.append("Review build configuration and dependencies")
            elif "server" in issue.lower():
                recommendations.append("Check server configuration and network connectivity")
            elif "missing" in issue.lower():
                recommendations.append("Restore missing files from backup or repository")
        
        # Add general recommendations
        if len(issues) > 5:
            recommendations.append("Consider running full system diagnostics")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _calculate_overall_health(self, issues: List[str]) -> str:
        """Calculate overall health status"""
        if not issues:
            return "healthy"
        
        critical_keywords = ["critical", "error", "missing", "failed"]
        warning_keywords = ["slow", "high", "low", "old"]
        
        critical_count = sum(1 for issue in issues if any(keyword in issue.lower() for keyword in critical_keywords))
        warning_count = sum(1 for issue in issues if any(keyword in issue.lower() for keyword in warning_keywords))
        
        if critical_count > 0:
            return "critical"
        elif warning_count > 2:
            return "warning"
        elif len(issues) > 0:
            return "warning"
        else:
            return "healthy"
    
    async def _send_alert(self, check_type: str, issue: str):
        """Send alert for critical issues"""
        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "check_type": check_type,
            "issue": issue,
            "severity": "critical" if "critical" in issue.lower() else "warning"
        }
        
        self.alerts_sent.append(alert)
        
        # Emit alert event
        await self.event_bus.publish_event(
            event_type="health_alert",
            source="health_monitor",
            data={"alert": alert}
        )
        
        logger.warning(f"Health alert ({check_type}): {issue}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get health monitor status"""
        latest_health = self.health_history[-1] if self.health_history else None
        
        return {
            "running": self.is_running,
            "checks_performed": len(self.health_history),
            "alerts_sent": len(self.alerts_sent),
            "latest_health": latest_health.__dict__ if latest_health else None,
            "thresholds": self.thresholds.copy()
        }
    
    async def get_health_history(self, limit: int = 10) -> List[HealthCheckResult]:
        """Get recent health check history"""
        return self.health_history[-limit:] if self.health_history else []
