"""
System Monitoring Implementation

Provides real-time system metrics collection and monitoring.
"""

import asyncio
import logging
import platform
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available, using mock system metrics")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    logging.warning("GPUtil not available, using mock GPU metrics")

logger = logging.getLogger(__name__)


@dataclass
class CPUMetrics:
    usage_percent: float
    cores: int
    frequency: float
    temperature: Optional[float] = None


@dataclass
class MemoryMetrics:
    total_gb: float
    used_gb: float
    available_gb: float
    usage_percent: float


@dataclass
class GPUMetrics:
    name: str
    usage_percent: float
    memory_total_gb: float
    memory_used_gb: float
    memory_free_gb: float
    temperature: float
    power_usage: float
    fan_speed: Optional[float] = None


@dataclass
class NetworkMetrics:
    bytes_sent: int
    bytes_received: int
    packets_sent: int
    packets_received: int
    connections: int


@dataclass
class AIComponentMetrics:
    reasoning_requests: int
    evolution_requests: int
    search_requests: int
    average_response_time: float
    success_rate: float
    error_count: int


@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu: CPUMetrics
    memory: MemoryMetrics
    gpu: Optional[GPUMetrics]
    network: NetworkMetrics
    ai_components: AIComponentMetrics


class SystemMonitor:
    """Real-time system monitoring service"""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.is_running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._metrics_history: List[SystemMetrics] = []
        self._max_history = 1000  # Keep last 1000 metrics
        
        # AI component tracking
        self._ai_metrics = {
            'reasoning_requests': 0,
            'evolution_requests': 0,
            'search_requests': 0,
            'total_response_time': 0.0,
            'success_count': 0,
            'error_count': 0
        }
    
    async def start(self):
        """Start the monitoring service"""
        if self.is_running:
            return
        
        self.is_running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("System monitor started")
    
    async def stop(self):
        """Stop the monitoring service"""
        if not self.is_running:
            return
        
        self.is_running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("System monitor stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                metrics = await self.collect_metrics()
                self._add_to_history(metrics)
                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.update_interval)
    
    def _add_to_history(self, metrics: SystemMetrics):
        """Add metrics to history with size limit"""
        self._metrics_history.append(metrics)
        if len(self._metrics_history) > self._max_history:
            self._metrics_history.pop(0)
    
    async def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        timestamp = datetime.now()
        
        # Collect metrics in parallel
        cpu_task = asyncio.create_task(self._get_cpu_metrics())
        memory_task = asyncio.create_task(self._get_memory_metrics())
        gpu_task = asyncio.create_task(self._get_gpu_metrics())
        network_task = asyncio.create_task(self._get_network_metrics())
        ai_task = asyncio.create_task(self._get_ai_metrics())
        
        cpu_metrics = await cpu_task
        memory_metrics = await memory_task
        gpu_metrics = await gpu_task
        network_metrics = await network_task
        ai_metrics = await ai_task
        
        return SystemMetrics(
            timestamp=timestamp,
            cpu=cpu_metrics,
            memory=memory_metrics,
            gpu=gpu_metrics,
            network=network_metrics,
            ai_components=ai_metrics
        )
    
    async def _get_cpu_metrics(self) -> CPUMetrics:
        """Get CPU metrics"""
        if not PSUTIL_AVAILABLE:
            return CPUMetrics(
                usage_percent=50.0,
                cores=8,
                frequency=2400.0,
                temperature=None
            )
        
        # Run CPU-intensive operations in thread pool
        loop = asyncio.get_event_loop()
        
        def get_cpu_info():
            usage = psutil.cpu_percent(interval=0.1)
            cores = psutil.cpu_count()
            freq = psutil.cpu_freq()
            frequency = freq.current if freq else 0.0
            
            # Try to get temperature (may not be available on all systems)
            temperature = None
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    # Get first available temperature sensor
                    for name, entries in temps.items():
                        if entries:
                            temperature = entries[0].current
                            break
            except (AttributeError, OSError):
                pass
            
            return usage, cores, frequency, temperature
        
        usage, cores, frequency, temperature = await loop.run_in_executor(
            None, get_cpu_info
        )
        
        return CPUMetrics(
            usage_percent=usage,
            cores=cores,
            frequency=frequency,
            temperature=temperature
        )
    
    async def _get_memory_metrics(self) -> MemoryMetrics:
        """Get memory metrics"""
        if not PSUTIL_AVAILABLE:
            return MemoryMetrics(
                total_gb=16.0,
                used_gb=8.0,
                available_gb=8.0,
                usage_percent=50.0
            )
        
        memory = psutil.virtual_memory()
        
        return MemoryMetrics(
            total_gb=memory.total / (1024**3),
            used_gb=memory.used / (1024**3),
            available_gb=memory.available / (1024**3),
            usage_percent=memory.percent
        )
    
    async def _get_gpu_metrics(self) -> Optional[GPUMetrics]:
        """Get real GPU metrics - no mock data allowed"""
        if not GPUTIL_AVAILABLE:
            # Try alternative GPU monitoring methods
            try:
                # Try nvidia-ml-py for NVIDIA GPUs
                import pynvml
                pynvml.nvmlInit()

                device_count = pynvml.nvmlDeviceGetCount()
                if device_count == 0:
                    logger.warning("No NVIDIA GPUs detected")
                    return None

                # Get first GPU
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)

                # Get GPU info
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')

                # Get utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)

                # Get memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                # Get temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

                # Get power usage
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
                except:
                    power = 0.0

                # Get fan speed
                try:
                    fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
                except:
                    fan_speed = None

                return GPUMetrics(
                    name=name,
                    usage_percent=float(util.gpu),
                    memory_total_gb=mem_info.total / (1024**3),
                    memory_used_gb=mem_info.used / (1024**3),
                    memory_free_gb=mem_info.free / (1024**3),
                    temperature=float(temp),
                    power_usage=power,
                    fan_speed=fan_speed
                )

            except ImportError:
                logger.warning("Neither GPUtil nor pynvml available for GPU monitoring")
                return None
            except Exception as e:
                logger.error(f"Error getting GPU metrics via pynvml: {e}")
                return None

        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                logger.warning("No GPUs detected by GPUtil")
                return None

            # Get first GPU
            gpu = gpus[0]

            return GPUMetrics(
                name=gpu.name,
                usage_percent=gpu.load * 100,
                memory_total_gb=gpu.memoryTotal / 1024,
                memory_used_gb=gpu.memoryUsed / 1024,
                memory_free_gb=gpu.memoryFree / 1024,
                temperature=gpu.temperature,
                power_usage=getattr(gpu, 'powerDraw', 0.0),
                fan_speed=getattr(gpu, 'fanSpeed', None)
            )

        except Exception as e:
            logger.error(f"Error getting GPU metrics: {e}")
            return None
    
    async def _get_network_metrics(self) -> NetworkMetrics:
        """Get network metrics"""
        if not PSUTIL_AVAILABLE:
            return NetworkMetrics(
                bytes_sent=1024000,
                bytes_received=2048000,
                packets_sent=1000,
                packets_received=1500,
                connections=25
            )
        
        net_io = psutil.net_io_counters()
        connections = len(psutil.net_connections())
        
        return NetworkMetrics(
            bytes_sent=net_io.bytes_sent,
            bytes_received=net_io.bytes_recv,
            packets_sent=net_io.packets_sent,
            packets_received=net_io.packets_recv,
            connections=connections
        )
    
    async def _get_ai_metrics(self) -> AIComponentMetrics:
        """Get AI component metrics"""
        total_requests = (
            self._ai_metrics['reasoning_requests'] +
            self._ai_metrics['evolution_requests'] +
            self._ai_metrics['search_requests']
        )
        
        avg_response_time = 0.0
        if self._ai_metrics['success_count'] > 0:
            avg_response_time = (
                self._ai_metrics['total_response_time'] / 
                self._ai_metrics['success_count']
            )
        
        success_rate = 0.0
        if total_requests > 0:
            success_rate = self._ai_metrics['success_count'] / total_requests
        
        return AIComponentMetrics(
            reasoning_requests=self._ai_metrics['reasoning_requests'],
            evolution_requests=self._ai_metrics['evolution_requests'],
            search_requests=self._ai_metrics['search_requests'],
            average_response_time=avg_response_time,
            success_rate=success_rate,
            error_count=self._ai_metrics['error_count']
        )
    
    def track_ai_request(self, component: str, response_time: float, success: bool):
        """Track AI component request metrics"""
        if component == 'reasoning':
            self._ai_metrics['reasoning_requests'] += 1
        elif component == 'evolution':
            self._ai_metrics['evolution_requests'] += 1
        elif component == 'search':
            self._ai_metrics['search_requests'] += 1
        
        if success:
            self._ai_metrics['success_count'] += 1
            self._ai_metrics['total_response_time'] += response_time
        else:
            self._ai_metrics['error_count'] += 1
    
    def get_latest_metrics(self) -> Optional[SystemMetrics]:
        """Get the latest collected metrics"""
        return self._metrics_history[-1] if self._metrics_history else None
    
    def get_metrics_history(self, limit: int = 100) -> List[SystemMetrics]:
        """Get recent metrics history"""
        return self._metrics_history[-limit:]


# Global monitor instance
_system_monitor: Optional[SystemMonitor] = None


async def get_system_monitor() -> SystemMonitor:
    """Get or create the global system monitor instance"""
    global _system_monitor
    
    if _system_monitor is None:
        _system_monitor = SystemMonitor()
        await _system_monitor.start()
    
    return _system_monitor


async def get_system_metrics() -> Dict[str, Any]:
    """Get current system metrics as a dictionary"""
    monitor = await get_system_monitor()
    metrics = await monitor.collect_metrics()
    return asdict(metrics)
