"""
Performance Monitoring Utilities

This module provides utilities for monitoring and measuring performance
including timers, profilers, memory tracking, and system metrics.
"""

import time
import asyncio
import psutil
import gc
import threading
from typing import Dict, Any, Optional, Callable, List, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import contextmanager, asynccontextmanager
from collections import defaultdict, deque
import functools
import cProfile
import pstats
import io


@dataclass
class PerformanceMetric:
    """Represents a performance metric"""
    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System resource metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


class Timer:
    """High-precision timer for measuring execution time"""
    
    def __init__(self, name: str = "timer"):
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed_time: Optional[float] = None
    
    def start(self):
        """Start the timer"""
        self.start_time = time.perf_counter()
        return self
    
    def stop(self) -> float:
        """Stop the timer and return elapsed time in seconds"""
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time
        return self.elapsed_time
    
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds"""
        if self.elapsed_time is None:
            raise RuntimeError("Timer not stopped")
        return self.elapsed_time * 1000
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


class AsyncTimer:
    """Async version of Timer"""
    
    def __init__(self, name: str = "async_timer"):
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed_time: Optional[float] = None
    
    async def start(self):
        """Start the timer"""
        self.start_time = time.perf_counter()
        return self
    
    async def stop(self) -> float:
        """Stop the timer and return elapsed time"""
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time
        return self.elapsed_time
    
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds"""
        if self.elapsed_time is None:
            raise RuntimeError("Timer not stopped")
        return self.elapsed_time * 1000
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()


class PerformanceMonitor:
    """Comprehensive performance monitoring system"""
    
    def __init__(self, max_metrics: int = 10000):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_metrics))
        self.counters: Dict[str, int] = defaultdict(int)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def record_metric(self, name: str, value: float, unit: str = "", **metadata):
        """Record a performance metric"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            metadata=metadata
        )
        
        with self._lock:
            self.metrics[name].append(metric)
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter"""
        with self._lock:
            self.counters[name] += value
    
    def record_timing(self, name: str, duration_ms: float):
        """Record timing information"""
        with self._lock:
            self.timers[name].append(duration_ms)
            
            # Keep only last 1000 timings
            if len(self.timers[name]) > 1000:
                self.timers[name] = self.timers[name][-1000:]
    
    def get_metric_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        with self._lock:
            if name not in self.metrics:
                return {}
            
            values = [m.value for m in self.metrics[name]]
            if not values:
                return {}
            
            return {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values),
                'latest': values[-1]
            }
    
    def get_timing_stats(self, name: str) -> Dict[str, float]:
        """Get timing statistics"""
        with self._lock:
            if name not in self.timers:
                return {}
            
            timings = self.timers[name]
            if not timings:
                return {}
            
            sorted_timings = sorted(timings)
            count = len(sorted_timings)
            
            return {
                'count': count,
                'min_ms': min(sorted_timings),
                'max_ms': max(sorted_timings),
                'avg_ms': sum(sorted_timings) / count,
                'p50_ms': sorted_timings[count // 2],
                'p95_ms': sorted_timings[int(count * 0.95)],
                'p99_ms': sorted_timings[int(count * 0.99)]
            }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get all performance statistics"""
        stats = {
            'metrics': {},
            'counters': dict(self.counters),
            'timings': {}
        }
        
        # Get metric stats
        for name in self.metrics:
            stats['metrics'][name] = self.get_metric_stats(name)
        
        # Get timing stats
        for name in self.timers:
            stats['timings'][name] = self.get_timing_stats(name)
        
        return stats
    
    def clear(self):
        """Clear all metrics"""
        with self._lock:
            self.metrics.clear()
            self.counters.clear()
            self.timers.clear()


class MemoryTracker:
    """Memory usage tracking utility"""
    
    def __init__(self):
        self.snapshots: List[Dict[str, Any]] = []
        self.baseline: Optional[Dict[str, Any]] = None
    
    def take_snapshot(self, label: str = "") -> Dict[str, Any]:
        """Take a memory usage snapshot"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        snapshot = {
            'label': label,
            'timestamp': datetime.utcnow(),
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def set_baseline(self, label: str = "baseline"):
        """Set baseline memory usage"""
        self.baseline = self.take_snapshot(label)
    
    def get_memory_delta(self) -> Optional[Dict[str, float]]:
        """Get memory usage delta from baseline"""
        if not self.baseline or not self.snapshots:
            return None
        
        current = self.snapshots[-1]
        return {
            'rss_delta_mb': current['rss_mb'] - self.baseline['rss_mb'],
            'vms_delta_mb': current['vms_mb'] - self.baseline['vms_mb'],
            'percent_delta': current['percent'] - self.baseline['percent']
        }
    
    def get_peak_usage(self) -> Optional[Dict[str, Any]]:
        """Get peak memory usage"""
        if not self.snapshots:
            return None
        
        peak_snapshot = max(self.snapshots, key=lambda s: s['rss_mb'])
        return peak_snapshot


class ProfilerContext:
    """Context manager for code profiling"""
    
    def __init__(self, name: str = "profile"):
        self.name = name
        self.profiler = cProfile.Profile()
        self.stats: Optional[pstats.Stats] = None
    
    def __enter__(self):
        """Start profiling"""
        self.profiler.enable()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop profiling"""
        self.profiler.disable()
        
        # Create stats
        s = io.StringIO()
        self.stats = pstats.Stats(self.profiler, stream=s)
    
    def get_stats(self, sort_by: str = 'cumulative', limit: int = 20) -> str:
        """Get profiling statistics"""
        if self.stats is None:
            return "No profiling data available"
        
        s = io.StringIO()
        stats = pstats.Stats(self.profiler, stream=s)
        stats.sort_stats(sort_by)
        stats.print_stats(limit)
        
        return s.getvalue()
    
    def save_stats(self, filename: str):
        """Save profiling stats to file"""
        if self.stats is not None:
            self.profiler.dump_stats(filename)


def timed(func: Optional[Callable] = None, *, monitor: Optional[PerformanceMonitor] = None):
    """Decorator to time function execution"""
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            timer = Timer(f.__name__)
            timer.start()
            
            try:
                result = f(*args, **kwargs)
                return result
            finally:
                duration_ms = timer.stop() * 1000
                
                if monitor:
                    monitor.record_timing(f.__name__, duration_ms)
        
        @functools.wraps(f)
        async def async_wrapper(*args, **kwargs):
            timer = Timer(f.__name__)
            timer.start()
            
            try:
                result = await f(*args, **kwargs)
                return result
            finally:
                duration_ms = timer.stop() * 1000
                
                if monitor:
                    monitor.record_timing(f.__name__, duration_ms)
        
        return async_wrapper if asyncio.iscoroutinefunction(f) else wrapper
    
    return decorator if func is None else decorator(func)


def get_system_metrics() -> SystemMetrics:
    """Get current system metrics"""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    network = psutil.net_io_counters()
    
    return SystemMetrics(
        cpu_percent=cpu_percent,
        memory_percent=memory.percent,
        memory_used_mb=memory.used / 1024 / 1024,
        memory_available_mb=memory.available / 1024 / 1024,
        disk_usage_percent=disk.percent,
        network_bytes_sent=network.bytes_sent,
        network_bytes_recv=network.bytes_recv
    )


# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_global_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor
