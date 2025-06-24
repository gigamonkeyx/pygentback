#!/usr/bin/env python3
"""
Advanced GPU Optimization System for PyGent Factory

Comprehensive GPU acceleration with RTX 3080 optimization, memory management,
model inference acceleration, and performance monitoring.
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

from .gpu_config import gpu_manager

logger = logging.getLogger(__name__)


@dataclass
class GPUOptimizationConfig:
    """GPU optimization configuration"""
    # Memory management
    memory_fraction: float = 0.8  # Use 80% of GPU memory
    memory_growth: bool = True    # Allow dynamic memory growth
    cache_cleanup_threshold: float = 0.9  # Cleanup when 90% full
    
    # Performance optimization
    use_mixed_precision: bool = True
    use_tensor_cores: bool = True
    optimize_for_inference: bool = True
    batch_size_optimization: bool = True
    
    # Model optimization
    use_torch_compile: bool = True  # PyTorch 2.0+ compilation
    use_flash_attention: bool = True
    quantization_enabled: bool = False  # INT8 quantization
    
    # Monitoring
    performance_monitoring: bool = True
    memory_monitoring: bool = True
    temperature_monitoring: bool = True
    
    # RTX 3080 specific optimizations
    rtx_3080_optimizations: bool = True
    tensor_core_optimization: bool = True
    cuda_graphs: bool = True


@dataclass
class GPUPerformanceMetrics:
    """GPU performance metrics tracking"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    gpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    temperature: float = 0.0
    power_usage: float = 0.0
    inference_time_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    memory_allocated_mb: float = 0.0
    memory_cached_mb: float = 0.0


class GPUOptimizer:
    """Advanced GPU optimization and acceleration manager"""
    
    def __init__(self, config: Optional[GPUOptimizationConfig] = None):
        self.config = config or GPUOptimizationConfig()
        self.is_initialized = False
        self.scaler = None
        self.performance_history: List[GPUPerformanceMetrics] = []
        self.optimization_cache = {}
        
        # RTX 3080 specific settings
        self.rtx_3080_detected = False
        self.tensor_core_available = False
        
    async def initialize(self) -> bool:
        """Initialize GPU optimization system"""
        try:
            logger.info("Initializing GPU optimization system...")
            
            if not gpu_manager.is_available():
                logger.warning("No GPU available, optimization disabled")
                return False
            
            # Detect RTX 3080
            self._detect_rtx_3080()
            
            # Configure memory management
            self._configure_memory_management()
            
            # Setup mixed precision
            if self.config.use_mixed_precision and gpu_manager.config.supports_mixed_precision:
                self.scaler = GradScaler()
                logger.info("Mixed precision training enabled with GradScaler")
            
            # Configure PyTorch optimizations
            self._configure_pytorch_optimizations()
            
            # Setup FAISS GPU acceleration
            if FAISS_AVAILABLE:
                self._configure_faiss_gpu()
            
            # Setup CuPy acceleration
            if CUPY_AVAILABLE:
                self._configure_cupy()
            
            # Start performance monitoring
            if self.config.performance_monitoring:
                asyncio.create_task(self._performance_monitor())
            
            self.is_initialized = True
            logger.info("GPU optimization system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize GPU optimization: {e}")
            return False
    
    def _detect_rtx_3080(self):
        """Detect RTX 3080 and configure specific optimizations"""
        if gpu_manager.config.device_name:
            device_name = gpu_manager.config.device_name.lower()
            if "rtx 3080" in device_name or "geforce rtx 3080" in device_name:
                self.rtx_3080_detected = True
                logger.info("RTX 3080 detected - enabling specific optimizations")
                
                # RTX 3080 has Ampere architecture with enhanced Tensor Cores
                if gpu_manager.config.compute_capability and gpu_manager.config.compute_capability >= (8, 6):
                    self.tensor_core_available = True
                    logger.info("Ampere Tensor Cores available")
    
    def _configure_memory_management(self):
        """Configure GPU memory management"""
        if gpu_manager.config.device.startswith("cuda"):
            # Set memory fraction
            if hasattr(torch.cuda, 'set_memory_fraction'):
                torch.cuda.set_memory_fraction(self.config.memory_fraction)
            
            # Enable memory growth if supported
            if self.config.memory_growth:
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            
            # Configure memory pool
            if hasattr(torch.cuda, 'memory'):
                torch.cuda.memory.set_per_process_memory_fraction(self.config.memory_fraction)
            
            logger.info(f"GPU memory configured: {self.config.memory_fraction*100:.1f}% allocation")
    
    def _configure_pytorch_optimizations(self):
        """Configure PyTorch-specific optimizations"""
        if gpu_manager.config.device.startswith("cuda"):
            # Enable cuDNN optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable TensorFloat-32 for RTX 3080
            if self.rtx_3080_detected:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("TensorFloat-32 enabled for RTX 3080")
            
            # Enable Tensor Core optimizations
            if self.tensor_core_available and self.config.tensor_core_optimization:
                torch.backends.cuda.enable_flash_sdp(True)
                logger.info("Flash Attention enabled for Tensor Cores")
    
    def _configure_faiss_gpu(self):
        """Configure FAISS GPU acceleration"""
        try:
            if faiss.get_num_gpus() > 0:
                # Configure FAISS GPU resources
                res = faiss.StandardGpuResources()
                
                # Set memory fraction for FAISS
                faiss_memory = int(gpu_manager.config.memory_total * 0.3 * 1024 * 1024)  # 30% for FAISS
                res.setTempMemory(faiss_memory)
                
                logger.info(f"FAISS GPU acceleration configured with {faiss_memory//1024//1024}MB")
                
        except Exception as e:
            logger.warning(f"Failed to configure FAISS GPU: {e}")
    
    def _configure_cupy(self):
        """Configure CuPy acceleration"""
        try:
            if cp.cuda.is_available():
                # Set memory pool
                mempool = cp.get_default_memory_pool()
                mempool.set_limit(size=int(gpu_manager.config.memory_total * 0.2 * 1024 * 1024))  # 20% for CuPy
                
                logger.info("CuPy GPU acceleration configured")
                
        except Exception as e:
            logger.warning(f"Failed to configure CuPy: {e}")
    
    @contextmanager
    def optimized_inference(self, model_name: str = "default"):
        """Context manager for optimized inference"""
        start_time = time.time()
        
        try:
            # Clear cache if needed
            if self._should_clear_cache():
                self.clear_memory_cache()
            
            # Enable inference optimizations
            with torch.inference_mode():
                if self.config.use_mixed_precision and self.scaler:
                    with autocast():
                        yield
                else:
                    yield
                    
        finally:
            # Record performance metrics
            inference_time = (time.time() - start_time) * 1000  # ms
            self._record_inference_metrics(model_name, inference_time)
    
    def _should_clear_cache(self) -> bool:
        """Check if memory cache should be cleared"""
        if gpu_manager.config.device.startswith("cuda"):
            try:
                allocated = torch.cuda.memory_allocated()
                total = torch.cuda.get_device_properties(0).total_memory
                utilization = allocated / total
                
                return utilization > self.config.cache_cleanup_threshold
                
            except Exception:
                return False
        return False
    
    def clear_memory_cache(self):
        """Clear GPU memory cache"""
        if gpu_manager.config.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        if CUPY_AVAILABLE and cp.cuda.is_available():
            cp.get_default_memory_pool().free_all_blocks()
            
        logger.debug("GPU memory cache cleared")
    
    def optimize_model_for_inference(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimize model for inference"""
        try:
            # Move to GPU
            if gpu_manager.is_available():
                model = model.to(gpu_manager.get_device())
            
            # Set to evaluation mode
            model.eval()
            
            # Disable gradient computation
            for param in model.parameters():
                param.requires_grad = False
            
            # Apply PyTorch 2.0 compilation if available
            if self.config.use_torch_compile and hasattr(torch, 'compile'):
                model = torch.compile(model, mode='max-autotune')
                logger.info("Model compiled with torch.compile")
            
            # Apply quantization if enabled
            if self.config.quantization_enabled:
                model = self._apply_quantization(model)
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to optimize model: {e}")
            return model
    
    def _apply_quantization(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply INT8 quantization to model"""
        try:
            # Dynamic quantization for inference
            quantized_model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            logger.info("INT8 quantization applied")
            return quantized_model
            
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
            return model
    
    async def _performance_monitor(self):
        """Background performance monitoring"""
        while self.is_initialized:
            try:
                metrics = self._collect_performance_metrics()
                self.performance_history.append(metrics)
                
                # Keep only last 1000 metrics
                if len(self.performance_history) > 1000:
                    self.performance_history = self.performance_history[-1000:]
                
                # Log warnings for high utilization
                if metrics.memory_utilization > 0.95:
                    logger.warning(f"High GPU memory utilization: {metrics.memory_utilization:.1%}")
                
                if metrics.temperature > 80:
                    logger.warning(f"High GPU temperature: {metrics.temperature}°C")
                
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(10)
    
    def _collect_performance_metrics(self) -> GPUPerformanceMetrics:
        """Collect current GPU performance metrics"""
        metrics = GPUPerformanceMetrics()
        
        try:
            if GPUTIL_AVAILABLE:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # First GPU
                    metrics.gpu_utilization = gpu.load
                    metrics.memory_utilization = gpu.memoryUtil
                    metrics.temperature = gpu.temperature
            
            if gpu_manager.config.device.startswith("cuda"):
                allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
                cached = torch.cuda.memory_reserved() / (1024**2)  # MB
                metrics.memory_allocated_mb = allocated
                metrics.memory_cached_mb = cached
                
        except Exception as e:
            logger.debug(f"Error collecting metrics: {e}")
        
        return metrics
    
    def _record_inference_metrics(self, model_name: str, inference_time_ms: float):
        """Record inference performance metrics"""
        if self.performance_history:
            latest_metrics = self.performance_history[-1]
            latest_metrics.inference_time_ms = inference_time_ms
            
            # Estimate throughput (rough approximation)
            if inference_time_ms > 0:
                latest_metrics.throughput_tokens_per_sec = 1000 / inference_time_ms
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status"""
        status = {
            "initialized": self.is_initialized,
            "gpu_available": gpu_manager.is_available(),
            "rtx_3080_detected": self.rtx_3080_detected,
            "tensor_cores_available": self.tensor_core_available,
            "mixed_precision_enabled": self.scaler is not None,
            "faiss_gpu_available": FAISS_AVAILABLE and faiss.get_num_gpus() > 0 if FAISS_AVAILABLE else False,
            "cupy_available": CUPY_AVAILABLE and cp.cuda.is_available() if CUPY_AVAILABLE else False,
            "config": {
                "memory_fraction": self.config.memory_fraction,
                "use_mixed_precision": self.config.use_mixed_precision,
                "use_tensor_cores": self.config.use_tensor_cores,
                "quantization_enabled": self.config.quantization_enabled
            }
        }
        
        # Add latest performance metrics
        if self.performance_history:
            latest = self.performance_history[-1]
            status["performance"] = {
                "gpu_utilization": latest.gpu_utilization,
                "memory_utilization": latest.memory_utilization,
                "temperature": latest.temperature,
                "inference_time_ms": latest.inference_time_ms,
                "memory_allocated_mb": latest.memory_allocated_mb
            }
        
        return status
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        if not self.performance_history:
            return {"error": "No performance data available"}
        
        recent_metrics = self.performance_history[-60:]  # Last 5 minutes
        
        gpu_utils = [m.gpu_utilization for m in recent_metrics if m.gpu_utilization > 0]
        mem_utils = [m.memory_utilization for m in recent_metrics if m.memory_utilization > 0]
        temps = [m.temperature for m in recent_metrics if m.temperature > 0]
        inference_times = [m.inference_time_ms for m in recent_metrics if m.inference_time_ms > 0]
        
        return {
            "gpu_utilization": {
                "avg": sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0,
                "max": max(gpu_utils) if gpu_utils else 0,
                "min": min(gpu_utils) if gpu_utils else 0
            },
            "memory_utilization": {
                "avg": sum(mem_utils) / len(mem_utils) if mem_utils else 0,
                "max": max(mem_utils) if mem_utils else 0,
                "min": min(mem_utils) if mem_utils else 0
            },
            "temperature": {
                "avg": sum(temps) / len(temps) if temps else 0,
                "max": max(temps) if temps else 0,
                "min": min(temps) if temps else 0
            },
            "inference_performance": {
                "avg_time_ms": sum(inference_times) / len(inference_times) if inference_times else 0,
                "min_time_ms": min(inference_times) if inference_times else 0,
                "max_time_ms": max(inference_times) if inference_times else 0
            },
            "sample_count": len(recent_metrics),
            "monitoring_duration_minutes": 5
        }


# Global GPU optimizer instance
gpu_optimizer = GPUOptimizer()
