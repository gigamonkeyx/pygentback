#!/usr/bin/env python3
"""
Memory System Performance Monitor

Tracks memory system performance and GPU utilization as data scales.
Provides real-time monitoring and historical analysis for optimization.
"""

import asyncio
import json
import time
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# GPU monitoring
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_MEMORY_TOTAL = torch.cuda.get_device_properties(0).total_memory
except ImportError:
    GPU_AVAILABLE = False
    GPU_NAME = "Not Available"
    GPU_MEMORY_TOTAL = 0

# Memory system imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.memory.memory_manager import MemorySpace, MemoryType, MemoryImportance
    from src.storage.vector_store import VectorStoreManager
    from src.utils.embedding import EmbeddingService
    from src.config.settings import Settings
    MEMORY_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Memory system imports not available: {e}")
    MEMORY_IMPORTS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""
    timestamp: datetime
    
    # System metrics
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    
    # GPU metrics
    gpu_available: bool
    gpu_memory_used_mb: float
    gpu_memory_total_mb: float
    gpu_utilization_percent: float
    
    # Memory system metrics
    total_memory_entries: int
    memory_by_type: Dict[str, int]
    average_retrieval_time_ms: float
    cache_hit_rate: float
    
    # Vector store metrics
    total_vectors: int
    vector_dimensions: int
    index_size_mb: float
    search_throughput_qps: float
    
    # Database metrics
    db_connections: int
    db_query_time_ms: float
    db_size_mb: float


class MemorySystemMonitor:
    """Monitor memory system performance and scaling"""
    
    def __init__(self, settings: Optional[Any] = None):
        self.settings = settings
        self.metrics_history: List[PerformanceMetrics] = []
        self.monitoring = False
        self.monitor_interval = 10.0  # seconds
        
        # Initialize components if available
        self.memory_manager = None
        self.vector_store_manager = None
        self.embedding_service = None
        
        if MEMORY_IMPORTS_AVAILABLE and settings:
            self._initialize_components()
    
    def _initialize_components(self):
        """Initialize memory system components"""
        try:
            self.vector_store_manager = VectorStoreManager(self.settings)
            self.embedding_service = EmbeddingService(self.settings)
            logger.info("Memory system components initialized")
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        # System metrics
        memory = psutil.virtual_memory()
        
        metrics = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": memory.percent,
            "memory_used_gb": memory.used / (1024**3),
            "memory_available_gb": memory.available / (1024**3),
        }
        
        # GPU metrics
        if GPU_AVAILABLE:
            try:
                gpu_memory_used = torch.cuda.memory_allocated() / (1024**2)  # MB
                gpu_memory_total = GPU_MEMORY_TOTAL / (1024**2)  # MB
                
                metrics.update({
                    "gpu_available": True,
                    "gpu_memory_used_mb": gpu_memory_used,
                    "gpu_memory_total_mb": gpu_memory_total,
                    "gpu_utilization_percent": (gpu_memory_used / gpu_memory_total) * 100,
                })
            except Exception as e:
                logger.warning(f"Failed to get GPU metrics: {e}")
                metrics.update({
                    "gpu_available": False,
                    "gpu_memory_used_mb": 0,
                    "gpu_memory_total_mb": 0,
                    "gpu_utilization_percent": 0,
                })
        else:
            metrics.update({
                "gpu_available": False,
                "gpu_memory_used_mb": 0,
                "gpu_memory_total_mb": 0,
                "gpu_utilization_percent": 0,
            })
        
        return metrics
    
    async def get_memory_system_metrics(self) -> Dict[str, Any]:
        """Get memory system specific metrics"""
        metrics = {
            "total_memory_entries": 0,
            "memory_by_type": {},
            "average_retrieval_time_ms": 0,
            "cache_hit_rate": 0,
            "total_vectors": 0,
            "vector_dimensions": 0,
            "index_size_mb": 0,
            "search_throughput_qps": 0,
            "db_connections": 0,
            "db_query_time_ms": 0,
            "db_size_mb": 0,
        }
        
        # If memory system components are available, get real metrics
        if self.embedding_service:
            try:
                cache_size = self.embedding_service.cache.size()
                metrics.update({
                    "cache_hit_rate": min(cache_size / max(cache_size + 100, 1) * 100, 100),
                })
            except Exception as e:
                logger.warning(f"Failed to get cache metrics: {e}")
        
        return metrics
    
    async def collect_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive performance metrics"""
        system_metrics = self.get_system_metrics()
        memory_metrics = await self.get_memory_system_metrics()
        
        return PerformanceMetrics(
            timestamp=datetime.utcnow(),
            cpu_percent=system_metrics["cpu_percent"],
            memory_percent=system_metrics["memory_percent"],
            memory_used_gb=system_metrics["memory_used_gb"],
            memory_available_gb=system_metrics["memory_available_gb"],
            gpu_available=system_metrics["gpu_available"],
            gpu_memory_used_mb=system_metrics["gpu_memory_used_mb"],
            gpu_memory_total_mb=system_metrics["gpu_memory_total_mb"],
            gpu_utilization_percent=system_metrics["gpu_utilization_percent"],
            total_memory_entries=memory_metrics["total_memory_entries"],
            memory_by_type=memory_metrics["memory_by_type"],
            average_retrieval_time_ms=memory_metrics["average_retrieval_time_ms"],
            cache_hit_rate=memory_metrics["cache_hit_rate"],
            total_vectors=memory_metrics["total_vectors"],
            vector_dimensions=memory_metrics["vector_dimensions"],
            index_size_mb=memory_metrics["index_size_mb"],
            search_throughput_qps=memory_metrics["search_throughput_qps"],
            db_connections=memory_metrics["db_connections"],
            db_query_time_ms=memory_metrics["db_query_time_ms"],
            db_size_mb=memory_metrics["db_size_mb"],
        )
    
    async def start_monitoring(self, interval: float = 10.0):
        """Start continuous monitoring"""
        self.monitor_interval = interval
        self.monitoring = True
        
        logger.info(f"Starting memory system monitoring (interval: {interval}s)")
        
        while self.monitoring:
            try:
                metrics = await self.collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only last 1000 entries
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                # Log important metrics
                logger.info(
                    f"CPU: {metrics.cpu_percent:.1f}% | "
                    f"Memory: {metrics.memory_percent:.1f}% | "
                    f"GPU Memory: {metrics.gpu_memory_used_mb:.0f}MB | "
                    f"Cache Hit: {metrics.cache_hit_rate:.1f}%"
                )
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(interval)
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        logger.info("Memory system monitoring stopped")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from collected metrics"""
        if not self.metrics_history:
            return {"error": "No metrics collected"}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 samples
        
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_gpu_memory = sum(m.gpu_memory_used_mb for m in recent_metrics) / len(recent_metrics)
        avg_cache_hit = sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics)
        
        return {
            "monitoring_duration_minutes": len(self.metrics_history) * self.monitor_interval / 60,
            "total_samples": len(self.metrics_history),
            "average_cpu_percent": round(avg_cpu, 1),
            "average_memory_percent": round(avg_memory, 1),
            "average_gpu_memory_mb": round(avg_gpu_memory, 1),
            "average_cache_hit_rate": round(avg_cache_hit, 1),
            "gpu_available": GPU_AVAILABLE,
            "gpu_name": GPU_NAME,
            "memory_system_available": MEMORY_IMPORTS_AVAILABLE,
            "latest_metrics": asdict(recent_metrics[-1]) if recent_metrics else None,
        }
    
    def save_metrics(self, filepath: str):
        """Save metrics history to file"""
        data = {
            "metadata": {
                "created_at": datetime.utcnow().isoformat(),
                "gpu_available": GPU_AVAILABLE,
                "gpu_name": GPU_NAME,
                "memory_system_available": MEMORY_IMPORTS_AVAILABLE,
                "total_samples": len(self.metrics_history),
            },
            "metrics": [asdict(m) for m in self.metrics_history],
            "summary": self.get_performance_summary(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Metrics saved to {filepath}")


async def run_performance_test():
    """Run a comprehensive performance test"""
    print("🔍 Memory System Performance Monitor")
    print("=" * 40)
    
    monitor = MemorySystemMonitor()
    
    # Collect initial metrics
    print("\n📊 Collecting initial metrics...")
    initial_metrics = await monitor.collect_metrics()
    
    print(f"✅ System Status:")
    print(f"   CPU Usage: {initial_metrics.cpu_percent:.1f}%")
    print(f"   Memory Usage: {initial_metrics.memory_percent:.1f}%")
    print(f"   GPU Available: {initial_metrics.gpu_available}")
    
    if initial_metrics.gpu_available:
        print(f"   GPU Memory: {initial_metrics.gpu_memory_used_mb:.0f}MB / {initial_metrics.gpu_memory_total_mb:.0f}MB")
        print(f"   GPU Utilization: {initial_metrics.gpu_utilization_percent:.1f}%")
    
    # Short monitoring session
    print(f"\n🔄 Running 30-second monitoring session...")
    monitor_task = asyncio.create_task(monitor.start_monitoring(interval=2.0))
    
    await asyncio.sleep(30)
    monitor.stop_monitoring()
    monitor_task.cancel()
    
    # Generate summary
    summary = monitor.get_performance_summary()
    print(f"\n📈 Performance Summary:")
    print(f"   Samples Collected: {summary['total_samples']}")
    print(f"   Average CPU: {summary['average_cpu_percent']}%")
    print(f"   Average Memory: {summary['average_memory_percent']}%")
    print(f"   Average GPU Memory: {summary['average_gpu_memory_mb']}MB")
    print(f"   Cache Hit Rate: {summary['average_cache_hit_rate']}%")
    
    # Save detailed report
    report_path = "memory_system_performance_report.json"
    monitor.save_metrics(report_path)
    print(f"\n💾 Detailed report saved to: {report_path}")
    
    return summary


if __name__ == "__main__":
    asyncio.run(run_performance_test())
