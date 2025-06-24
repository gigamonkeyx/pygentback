"""
System Utilities

This module provides system-level utilities including system information,
resource monitoring, health checking, process management, and file watching.
"""

import os
import sys
import platform
import psutil
import asyncio
import subprocess
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
# Optional watchdog import for file watching
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    # Fallback classes when watchdog is not available
    class Observer:
        def __init__(self): pass
        def schedule(self, *args, **kwargs): pass
        def start(self): pass
        def stop(self): pass
        def join(self): pass

    class FileSystemEventHandler:
        def on_modified(self, event): pass
        def on_created(self, event): pass
        def on_deleted(self, event): pass
        def on_moved(self, event): pass

    class FileSystemEvent:
        def __init__(self, src_path):
            self.src_path = src_path
            self.is_directory = False

    WATCHDOG_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SystemInfo:
    """System information"""
    platform: str
    platform_version: str
    architecture: str
    processor: str
    python_version: str
    hostname: str
    username: str
    boot_time: datetime
    cpu_count: int
    memory_total_gb: float
    disk_total_gb: float
    
    @classmethod
    def get_current(cls) -> 'SystemInfo':
        """Get current system information"""
        memory_total = psutil.virtual_memory().total
        disk_total = psutil.disk_usage('/').total
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        
        return cls(
            platform=platform.system(),
            platform_version=platform.version(),
            architecture=platform.architecture()[0],
            processor=platform.processor(),
            python_version=platform.python_version(),
            hostname=platform.node(),
            username=os.getenv('USER', os.getenv('USERNAME', 'unknown')),
            boot_time=boot_time,
            cpu_count=psutil.cpu_count(),
            memory_total_gb=memory_total / (1024**3),
            disk_total_gb=disk_total / (1024**3)
        )


@dataclass
class ResourceUsage:
    """Resource usage snapshot"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    load_average: Optional[List[float]] = None  # Unix only
    
    @classmethod
    def get_current(cls) -> 'ResourceUsage':
        """Get current resource usage"""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        # Load average (Unix only)
        load_avg = None
        if hasattr(os, 'getloadavg'):
            try:
                load_avg = list(os.getloadavg())
            except OSError:
                pass
        
        return cls(
            timestamp=datetime.utcnow(),
            cpu_percent=psutil.cpu_percent(interval=0.1),
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_available_gb=memory.available / (1024**3),
            disk_usage_percent=disk.percent,
            disk_free_gb=disk.free / (1024**3),
            network_bytes_sent=network.bytes_sent,
            network_bytes_recv=network.bytes_recv,
            process_count=len(psutil.pids()),
            load_average=load_avg
        )


class ResourceMonitor:
    """Continuous resource monitoring"""
    
    def __init__(self, interval: float = 5.0, max_history: int = 1000):
        self.interval = interval
        self.max_history = max_history
        self.history: List[ResourceUsage] = []
        self.callbacks: List[Callable[[ResourceUsage], None]] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    def add_callback(self, callback: Callable[[ResourceUsage], None]):
        """Add callback for resource updates"""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[ResourceUsage], None]):
        """Remove callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    async def start(self):
        """Start monitoring"""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("Resource monitoring started")
    
    async def stop(self):
        """Stop monitoring"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Resource monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                # Get current usage
                usage = ResourceUsage.get_current()
                
                # Add to history
                self.history.append(usage)
                
                # Trim history if needed
                if len(self.history) > self.max_history:
                    self.history = self.history[-self.max_history:]
                
                # Call callbacks
                for callback in self.callbacks:
                    try:
                        callback(usage)
                    except Exception as e:
                        logger.error(f"Resource monitor callback failed: {e}")
                
                # Wait for next interval
                await asyncio.sleep(self.interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(self.interval)
    
    def get_average_usage(self, minutes: int = 5) -> Optional[ResourceUsage]:
        """Get average usage over specified minutes"""
        if not self.history:
            return None
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        recent_usage = [u for u in self.history if u.timestamp >= cutoff_time]
        
        if not recent_usage:
            return None
        
        # Calculate averages
        avg_cpu = sum(u.cpu_percent for u in recent_usage) / len(recent_usage)
        avg_memory = sum(u.memory_percent for u in recent_usage) / len(recent_usage)
        avg_disk = sum(u.disk_usage_percent for u in recent_usage) / len(recent_usage)
        
        # Use latest values for other fields
        latest = recent_usage[-1]
        
        return ResourceUsage(
            timestamp=datetime.utcnow(),
            cpu_percent=avg_cpu,
            memory_percent=avg_memory,
            memory_used_gb=latest.memory_used_gb,
            memory_available_gb=latest.memory_available_gb,
            disk_usage_percent=avg_disk,
            disk_free_gb=latest.disk_free_gb,
            network_bytes_sent=latest.network_bytes_sent,
            network_bytes_recv=latest.network_bytes_recv,
            process_count=latest.process_count,
            load_average=latest.load_average
        )


@dataclass
class HealthCheck:
    """Health check result"""
    name: str
    status: str  # healthy, warning, critical
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HealthChecker:
    """System health checker"""
    
    def __init__(self):
        self.checks: Dict[str, Callable[[], HealthCheck]] = {}
        self.thresholds = {
            'cpu_warning': 80.0,
            'cpu_critical': 95.0,
            'memory_warning': 80.0,
            'memory_critical': 95.0,
            'disk_warning': 80.0,
            'disk_critical': 95.0
        }
    
    def add_check(self, name: str, check_func: Callable[[], HealthCheck]):
        """Add custom health check"""
        self.checks[name] = check_func
    
    def remove_check(self, name: str):
        """Remove health check"""
        if name in self.checks:
            del self.checks[name]
    
    async def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all health checks"""
        results = {}
        
        # System resource checks
        results.update(await self._check_system_resources())
        
        # Custom checks
        for name, check_func in self.checks.items():
            try:
                results[name] = check_func()
            except Exception as e:
                results[name] = HealthCheck(
                    name=name,
                    status="critical",
                    message=f"Health check failed: {str(e)}"
                )
        
        return results
    
    async def _check_system_resources(self) -> Dict[str, HealthCheck]:
        """Check system resource health"""
        usage = ResourceUsage.get_current()
        checks = {}
        
        # CPU check
        if usage.cpu_percent >= self.thresholds['cpu_critical']:
            status = "critical"
            message = f"CPU usage critical: {usage.cpu_percent:.1f}%"
        elif usage.cpu_percent >= self.thresholds['cpu_warning']:
            status = "warning"
            message = f"CPU usage high: {usage.cpu_percent:.1f}%"
        else:
            status = "healthy"
            message = f"CPU usage normal: {usage.cpu_percent:.1f}%"
        
        checks['cpu'] = HealthCheck(
            name="cpu",
            status=status,
            message=message,
            metadata={'cpu_percent': usage.cpu_percent}
        )
        
        # Memory check
        if usage.memory_percent >= self.thresholds['memory_critical']:
            status = "critical"
            message = f"Memory usage critical: {usage.memory_percent:.1f}%"
        elif usage.memory_percent >= self.thresholds['memory_warning']:
            status = "warning"
            message = f"Memory usage high: {usage.memory_percent:.1f}%"
        else:
            status = "healthy"
            message = f"Memory usage normal: {usage.memory_percent:.1f}%"
        
        checks['memory'] = HealthCheck(
            name="memory",
            status=status,
            message=message,
            metadata={
                'memory_percent': usage.memory_percent,
                'memory_used_gb': usage.memory_used_gb,
                'memory_available_gb': usage.memory_available_gb
            }
        )
        
        # Disk check
        if usage.disk_usage_percent >= self.thresholds['disk_critical']:
            status = "critical"
            message = f"Disk usage critical: {usage.disk_usage_percent:.1f}%"
        elif usage.disk_usage_percent >= self.thresholds['disk_warning']:
            status = "warning"
            message = f"Disk usage high: {usage.disk_usage_percent:.1f}%"
        else:
            status = "healthy"
            message = f"Disk usage normal: {usage.disk_usage_percent:.1f}%"
        
        checks['disk'] = HealthCheck(
            name="disk",
            status=status,
            message=message,
            metadata={
                'disk_usage_percent': usage.disk_usage_percent,
                'disk_free_gb': usage.disk_free_gb
            }
        )
        
        return checks
    
    def get_overall_status(self, checks: Dict[str, HealthCheck]) -> str:
        """Get overall health status"""
        if any(check.status == "critical" for check in checks.values()):
            return "critical"
        elif any(check.status == "warning" for check in checks.values()):
            return "warning"
        else:
            return "healthy"


class ProcessManager:
    """Process management utility"""
    
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
    
    async def start_process(self, 
                           name: str, 
                           command: List[str],
                           cwd: Optional[str] = None,
                           env: Optional[Dict[str, str]] = None) -> bool:
        """Start a new process"""
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes[name] = process
            logger.info(f"Started process '{name}' with PID {process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start process '{name}': {e}")
            return False
    
    def stop_process(self, name: str, timeout: float = 10.0) -> bool:
        """Stop a process"""
        if name not in self.processes:
            return False
        
        process = self.processes[name]
        
        try:
            # Try graceful termination first
            process.terminate()
            
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Force kill if graceful termination fails
                process.kill()
                process.wait()
            
            del self.processes[name]
            logger.info(f"Stopped process '{name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop process '{name}': {e}")
            return False
    
    def get_process_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get process status"""
        if name not in self.processes:
            return None
        
        process = self.processes[name]
        
        try:
            # Check if process is still running
            return_code = process.poll()
            
            if return_code is None:
                # Process is still running
                ps_process = psutil.Process(process.pid)
                return {
                    'name': name,
                    'pid': process.pid,
                    'status': 'running',
                    'cpu_percent': ps_process.cpu_percent(),
                    'memory_percent': ps_process.memory_percent(),
                    'create_time': datetime.fromtimestamp(ps_process.create_time())
                }
            else:
                # Process has terminated
                return {
                    'name': name,
                    'pid': process.pid,
                    'status': 'terminated',
                    'return_code': return_code
                }
                
        except psutil.NoSuchProcess:
            return {
                'name': name,
                'pid': process.pid,
                'status': 'not_found'
            }
        except Exception as e:
            return {
                'name': name,
                'pid': process.pid,
                'status': 'error',
                'error': str(e)
            }
    
    def list_processes(self) -> Dict[str, Dict[str, Any]]:
        """List all managed processes"""
        return {name: self.get_process_status(name) for name in self.processes}
    
    def cleanup(self):
        """Stop all processes"""
        for name in list(self.processes.keys()):
            self.stop_process(name)


class FileWatcher:
    """File system watcher"""
    
    def __init__(self, path: str):
        self.path = Path(path)
        self.observer = Observer()
        self.handlers: List[FileSystemEventHandler] = []
        self._running = False
    
    def add_handler(self, handler: FileSystemEventHandler):
        """Add file system event handler"""
        self.handlers.append(handler)
        if self._running:
            self.observer.schedule(handler, str(self.path), recursive=True)
    
    def remove_handler(self, handler: FileSystemEventHandler):
        """Remove file system event handler"""
        if handler in self.handlers:
            self.handlers.remove(handler)
    
    def start(self):
        """Start watching"""
        if self._running:
            return

        if not WATCHDOG_AVAILABLE:
            logger.warning("Watchdog not available, file watching disabled")
            return

        for handler in self.handlers:
            self.observer.schedule(handler, str(self.path), recursive=True)

        self.observer.start()
        self._running = True
        logger.info(f"Started watching {self.path}")

    def stop(self):
        """Stop watching"""
        if not self._running:
            return

        if not WATCHDOG_AVAILABLE:
            return

        self.observer.stop()
        self.observer.join()
        self._running = False
        logger.info(f"Stopped watching {self.path}")


class SimpleFileHandler(FileSystemEventHandler):
    """Simple file event handler"""
    
    def __init__(self, callback: Callable[[FileSystemEvent], None]):
        self.callback = callback
    
    def on_any_event(self, event):
        """Handle any file system event"""
        try:
            self.callback(event)
        except Exception as e:
            logger.error(f"File event handler error: {e}")


def get_system_info() -> SystemInfo:
    """Get current system information"""
    return SystemInfo.get_current()


def get_resource_usage() -> ResourceUsage:
    """Get current resource usage"""
    return ResourceUsage.get_current()
