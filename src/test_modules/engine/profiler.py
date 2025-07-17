"""
Performance Profiler

This module provides comprehensive performance profiling for Agent + MCP recipe
execution, including timing, memory usage, resource utilization, and bottleneck analysis.
"""

import asyncio
import logging
import time
import traceback
from typing import Dict, List, Any, Optional, Callable, ContextManager, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import threading
from contextlib import contextmanager
import json

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None


logger = logging.getLogger(__name__)


@dataclass
class ProfilePoint:
    """A single profiling measurement point"""
    timestamp: float
    name: str
    category: str
    value: float
    unit: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    execution_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    peak_memory_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    io_read_bytes: int = 0
    io_write_bytes: int = 0
    network_sent_bytes: int = 0
    network_recv_bytes: int = 0
    function_calls: int = 0
    async_operations: int = 0
    bottlenecks: List[str] = field(default_factory=list)
    profile_points: List[ProfilePoint] = field(default_factory=list)


@dataclass
class ProfilingSession:
    """A profiling session"""
    session_id: str
    name: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    nested_sessions: List['ProfilingSession'] = field(default_factory=list)
    parent_session: Optional['ProfilingSession'] = None


class PerformanceProfiler:
    """
    Comprehensive performance profiler for recipe execution.
    
    Provides detailed timing, memory, CPU, I/O, and network profiling
    with bottleneck detection and optimization recommendations.
    """
    
    def __init__(self, enable_detailed_profiling: bool = True):
        self.enable_detailed_profiling = enable_detailed_profiling
        
        # Profiling state
        self.active_sessions: Dict[str, ProfilingSession] = {}
        self.completed_sessions: List[ProfilingSession] = []
        self.current_session: Optional[ProfilingSession] = None
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_interval = 0.1  # 100ms
        
        # Performance thresholds for bottleneck detection
        self.thresholds = {
            "slow_execution_ms": 5000,
            "high_memory_mb": 1024,
            "high_cpu_percent": 80,
            "high_io_bytes": 100 * 1024 * 1024,  # 100MB
            "many_function_calls": 10000
        }
        
        # Function call tracking
        self.function_call_counts: Dict[str, int] = {}
        self.function_timings: Dict[str, List[float]] = {}
        
        # System baseline
        self.baseline_metrics: Optional[Dict[str, float]] = None
        if PSUTIL_AVAILABLE:
            self._establish_baseline()
    
    def start_session(self, name: str, session_id: Optional[str] = None) -> str:
        """
        Start a new profiling session.
        
        Args:
            name: Session name
            session_id: Optional session ID
            
        Returns:
            Session ID
        """
        if session_id is None:
            session_id = f"session_{int(time.time() * 1000)}"
        
        session = ProfilingSession(
            session_id=session_id,
            name=name,
            started_at=datetime.utcnow(),
            parent_session=self.current_session
        )
        
        # Add as nested session if we have a parent
        if self.current_session:
            self.current_session.nested_sessions.append(session)
        
        self.active_sessions[session_id] = session
        self.current_session = session
        
        # Start monitoring if this is the first session
        if len(self.active_sessions) == 1:
            self._start_monitoring()
        
        logger.debug(f"Started profiling session: {name} ({session_id})")
        return session_id
    
    def end_session(self, session_id: str) -> PerformanceMetrics:
        """
        End a profiling session.
        
        Args:
            session_id: Session ID to end
            
        Returns:
            Performance metrics for the session
        """
        if session_id not in self.active_sessions:
            logger.warning(f"Session {session_id} not found")
            return PerformanceMetrics()
        
        session = self.active_sessions[session_id]
        session.ended_at = datetime.utcnow()
        
        # Calculate execution time
        if session.started_at and session.ended_at:
            execution_time = (session.ended_at - session.started_at).total_seconds() * 1000
            session.metrics.execution_time_ms = execution_time
        
        # Detect bottlenecks
        self._detect_bottlenecks(session)
        
        # Move to completed sessions
        self.active_sessions.pop(session_id)
        self.completed_sessions.append(session)
        
        # Update current session to parent
        if session.parent_session and session.parent_session.session_id in self.active_sessions:
            self.current_session = session.parent_session
        else:
            self.current_session = None
        
        # Stop monitoring if no active sessions
        if not self.active_sessions:
            self._stop_monitoring()
        
        logger.debug(f"Ended profiling session: {session.name} ({session_id})")
        return session.metrics
    
    @contextmanager
    def profile(self, name: str) -> ContextManager[str]:
        """
        Context manager for profiling a code block.
        
        Args:
            name: Profile name
            
        Yields:
            Session ID
        """
        session_id = self.start_session(name)
        try:
            yield session_id
        finally:
            self.end_session(session_id)
    
    def add_profile_point(self, 
                         name: str,
                         category: str,
                         value: float,
                         unit: str,
                         metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a custom profile point"""
        if not self.current_session:
            return
        
        point = ProfilePoint(
            timestamp=time.time(),
            name=name,
            category=category,
            value=value,
            unit=unit,
            metadata=metadata or {}
        )
        
        self.current_session.metrics.profile_points.append(point)
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile function execution"""
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = (time.time() - start_time) * 1000
                
                # Track function calls
                self.function_call_counts[func_name] = self.function_call_counts.get(func_name, 0) + 1
                if func_name not in self.function_timings:
                    self.function_timings[func_name] = []
                self.function_timings[func_name].append(execution_time)
                
                # Update current session metrics
                if self.current_session:
                    self.current_session.metrics.function_calls += 1
                    
                    # Add profile point for slow functions
                    if execution_time > 100:  # 100ms threshold
                        self.add_profile_point(
                            name=f"function_{func_name}",
                            category="function_timing",
                            value=execution_time,
                            unit="ms",
                            metadata={"function": func_name, "args_count": len(args)}
                        )
        
        return wrapper
    
    def profile_async_function(self, func: Callable) -> Callable:
        """Decorator to profile async function execution"""
        async def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"
            
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                execution_time = (time.time() - start_time) * 1000
                
                # Track function calls
                self.function_call_counts[func_name] = self.function_call_counts.get(func_name, 0) + 1
                if func_name not in self.function_timings:
                    self.function_timings[func_name] = []
                self.function_timings[func_name].append(execution_time)
                
                # Update current session metrics
                if self.current_session:
                    self.current_session.metrics.function_calls += 1
                    self.current_session.metrics.async_operations += 1
                    
                    # Add profile point for slow functions
                    if execution_time > 100:  # 100ms threshold
                        self.add_profile_point(
                            name=f"async_function_{func_name}",
                            category="async_timing",
                            value=execution_time,
                            unit="ms",
                            metadata={"function": func_name, "args_count": len(args)}
                        )
        
        return wrapper
    
    def get_session_metrics(self, session_id: str) -> Optional[PerformanceMetrics]:
        """Get metrics for a specific session"""
        # Check active sessions
        if session_id in self.active_sessions:
            return self.active_sessions[session_id].metrics
        
        # Check completed sessions
        for session in self.completed_sessions:
            if session.session_id == session_id:
                return session.metrics
        
        return None
    
    def get_function_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get function call statistics"""
        stats = {}
        
        for func_name, call_count in self.function_call_counts.items():
            timings = self.function_timings.get(func_name, [])
            
            if timings:
                avg_time = sum(timings) / len(timings)
                max_time = max(timings)
                min_time = min(timings)
                total_time = sum(timings)
            else:
                avg_time = max_time = min_time = total_time = 0.0
            
            stats[func_name] = {
                "call_count": call_count,
                "average_time_ms": avg_time,
                "max_time_ms": max_time,
                "min_time_ms": min_time,
                "total_time_ms": total_time
            }
        
        return stats
    
    def generate_performance_report(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if session_id:
            sessions = [s for s in self.completed_sessions if s.session_id == session_id]
        else:
            sessions = self.completed_sessions
        
        if not sessions:
            return {"error": "No sessions found"}
        
        # Aggregate metrics
        total_execution_time = sum(s.metrics.execution_time_ms for s in sessions)
        avg_execution_time = total_execution_time / len(sessions)
        max_memory = max(s.metrics.peak_memory_mb for s in sessions)
        avg_memory = sum(s.metrics.memory_usage_mb for s in sessions) / len(sessions)
        
        # Collect all bottlenecks
        all_bottlenecks = []
        for session in sessions:
            all_bottlenecks.extend(session.metrics.bottlenecks)
        
        # Function statistics
        function_stats = self.get_function_statistics()
        
        # Top slow functions
        slow_functions = sorted(
            function_stats.items(),
            key=lambda x: x[1]["average_time_ms"],
            reverse=True
        )[:10]
        
        # Most called functions
        frequent_functions = sorted(
            function_stats.items(),
            key=lambda x: x[1]["call_count"],
            reverse=True
        )[:10]
        
        report = {
            "summary": {
                "total_sessions": len(sessions),
                "total_execution_time_ms": total_execution_time,
                "average_execution_time_ms": avg_execution_time,
                "max_memory_usage_mb": max_memory,
                "average_memory_usage_mb": avg_memory,
                "total_function_calls": sum(self.function_call_counts.values()),
                "unique_functions_called": len(self.function_call_counts)
            },
            "bottlenecks": {
                "total_detected": len(all_bottlenecks),
                "unique_bottlenecks": list(set(all_bottlenecks)),
                "most_common": self._get_most_common_bottlenecks(all_bottlenecks)
            },
            "function_analysis": {
                "slowest_functions": slow_functions,
                "most_frequent_functions": frequent_functions
            },
            "recommendations": self._generate_optimization_recommendations(sessions)
        }
        
        return report
    
    def _establish_baseline(self) -> None:
        """Establish system performance baseline"""
        if not PSUTIL_AVAILABLE:
            return
        
        try:
            process = psutil.Process()
            
            self.baseline_metrics = {
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "cpu_percent": process.cpu_percent(),
                "io_read_bytes": process.io_counters().read_bytes,
                "io_write_bytes": process.io_counters().write_bytes
            }
            
            logger.debug("Established performance baseline")
        except Exception as e:
            logger.warning(f"Failed to establish baseline: {e}")
    
    def _start_monitoring(self) -> None:
        """Start background monitoring thread"""
        if self.monitoring_active or not PSUTIL_AVAILABLE:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.debug("Started performance monitoring")
    
    def _stop_monitoring(self) -> None:
        """Stop background monitoring thread"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        
        logger.debug("Stopped performance monitoring")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop"""
        try:
            process = psutil.Process()
            
            while self.monitoring_active:
                if self.current_session and PSUTIL_AVAILABLE:
                    try:
                        # Memory usage
                        memory_info = process.memory_info()
                        current_memory = memory_info.rss / 1024 / 1024  # MB
                        
                        self.current_session.metrics.memory_usage_mb = current_memory
                        self.current_session.metrics.peak_memory_mb = max(
                            self.current_session.metrics.peak_memory_mb,
                            current_memory
                        )
                        
                        # CPU usage
                        cpu_percent = process.cpu_percent()
                        self.current_session.metrics.cpu_usage_percent = cpu_percent
                        
                        # I/O counters
                        io_counters = process.io_counters()
                        if self.baseline_metrics:
                            self.current_session.metrics.io_read_bytes = (
                                io_counters.read_bytes - self.baseline_metrics["io_read_bytes"]
                            )
                            self.current_session.metrics.io_write_bytes = (
                                io_counters.write_bytes - self.baseline_metrics["io_write_bytes"]
                            )
                        
                        # Network I/O (if available)
                        try:
                            net_io = psutil.net_io_counters()
                            self.current_session.metrics.network_sent_bytes = net_io.bytes_sent
                            self.current_session.metrics.network_recv_bytes = net_io.bytes_recv
                        except AttributeError:
                            pass  # Network counters not available
                        
                    except Exception as e:
                        logger.warning(f"Monitoring error: {e}")
                
                time.sleep(self.monitoring_interval)
                
        except Exception as e:
            logger.error(f"Monitoring loop failed: {e}")
    
    def _detect_bottlenecks(self, session: ProfilingSession) -> None:
        """Detect performance bottlenecks in a session"""
        metrics = session.metrics
        bottlenecks = []
        
        # Slow execution
        if metrics.execution_time_ms > self.thresholds["slow_execution_ms"]:
            bottlenecks.append(f"slow_execution_{metrics.execution_time_ms:.0f}ms")
        
        # High memory usage
        if metrics.peak_memory_mb > self.thresholds["high_memory_mb"]:
            bottlenecks.append(f"high_memory_{metrics.peak_memory_mb:.0f}mb")
        
        # High CPU usage
        if metrics.cpu_usage_percent > self.thresholds["high_cpu_percent"]:
            bottlenecks.append(f"high_cpu_{metrics.cpu_usage_percent:.0f}percent")
        
        # High I/O
        total_io = metrics.io_read_bytes + metrics.io_write_bytes
        if total_io > self.thresholds["high_io_bytes"]:
            bottlenecks.append(f"high_io_{total_io // (1024*1024)}mb")
        
        # Many function calls
        if metrics.function_calls > self.thresholds["many_function_calls"]:
            bottlenecks.append(f"many_function_calls_{metrics.function_calls}")
        
        # Analyze profile points for specific bottlenecks
        for point in metrics.profile_points:
            if point.category == "function_timing" and point.value > 1000:  # 1 second
                bottlenecks.append(f"slow_function_{point.metadata.get('function', 'unknown')}")
        
        metrics.bottlenecks = bottlenecks
    
    def _get_most_common_bottlenecks(self, bottlenecks: List[str]) -> List[Tuple[str, int]]:
        """Get most common bottlenecks"""
        bottleneck_counts = {}
        for bottleneck in bottlenecks:
            # Extract bottleneck type (before the first underscore)
            bottleneck_type = bottleneck.split('_')[0] if '_' in bottleneck else bottleneck
            bottleneck_counts[bottleneck_type] = bottleneck_counts.get(bottleneck_type, 0) + 1
        
        return sorted(bottleneck_counts.items(), key=lambda x: x[1], reverse=True)
    
    def _generate_optimization_recommendations(self, sessions: List[ProfilingSession]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Analyze common bottlenecks
        all_bottlenecks = []
        for session in sessions:
            all_bottlenecks.extend(session.metrics.bottlenecks)
        
        bottleneck_counts = {}
        for bottleneck in all_bottlenecks:
            bottleneck_type = bottleneck.split('_')[0]
            bottleneck_counts[bottleneck_type] = bottleneck_counts.get(bottleneck_type, 0) + 1
        
        # Generate recommendations based on bottlenecks
        if bottleneck_counts.get("slow", 0) > len(sessions) * 0.5:
            recommendations.append("Consider optimizing algorithm complexity or adding caching")
        
        if bottleneck_counts.get("high", 0) > len(sessions) * 0.3:
            recommendations.append("Monitor resource usage and consider scaling or optimization")
        
        if bottleneck_counts.get("many", 0) > len(sessions) * 0.3:
            recommendations.append("Reduce function call overhead or batch operations")
        
        # Analyze function statistics
        function_stats = self.get_function_statistics()
        if function_stats:
            # Find functions with high total time
            high_time_functions = [
                func for func, stats in function_stats.items()
                if stats["total_time_ms"] > 5000  # 5 seconds total
            ]
            
            if high_time_functions:
                recommendations.append(f"Optimize high-impact functions: {', '.join(high_time_functions[:3])}")
        
        # Memory recommendations
        avg_memory = sum(s.metrics.peak_memory_mb for s in sessions) / len(sessions)
        if avg_memory > 512:  # 512MB
            recommendations.append("Consider memory optimization or garbage collection tuning")
        
        if not recommendations:
            recommendations.append("Performance looks good! No major optimizations needed.")
        
        return recommendations
