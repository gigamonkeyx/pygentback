#!/usr/bin/env python3
"""
GPU Monitoring and Optimization API

Real-time GPU monitoring, performance analytics, and optimization controls
for PyGent Factory with RTX 3080 specific features.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

try:
    from ..core.gpu_optimization import gpu_optimizer, GPUOptimizationConfig
    from ..core.gpu_config import gpu_manager
    from ..core.ollama_gpu_integration import ollama_gpu_manager
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from core.gpu_optimization import gpu_optimizer, GPUOptimizationConfig
    from core.gpu_config import gpu_manager
    from core.ollama_gpu_integration import ollama_gpu_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/gpu", tags=["GPU Monitoring"])


class GPUStatusResponse(BaseModel):
    """GPU status response model"""
    gpu_available: bool
    device_name: str
    device_type: str
    memory_total_mb: int
    memory_available_mb: int
    memory_utilization: float
    temperature: float
    power_usage: float
    compute_capability: Optional[tuple]
    cuda_version: Optional[str]
    optimization_enabled: bool
    rtx_3080_detected: bool


class GPUPerformanceResponse(BaseModel):
    """GPU performance response model"""
    current_metrics: Dict[str, Any]
    performance_summary: Dict[str, Any]
    optimization_status: Dict[str, Any]
    ollama_performance: Dict[str, Any]


class OptimizationConfigRequest(BaseModel):
    """Optimization configuration request"""
    memory_fraction: Optional[float] = None
    use_mixed_precision: Optional[bool] = None
    use_tensor_cores: Optional[bool] = None
    quantization_enabled: Optional[bool] = None
    performance_monitoring: Optional[bool] = None


@router.get("/status", response_model=GPUStatusResponse)
async def get_gpu_status():
    """Get current GPU status and capabilities"""
    try:
        if not gpu_manager.is_available():
            return GPUStatusResponse(
                gpu_available=False,
                device_name="CPU",
                device_type="cpu",
                memory_total_mb=0,
                memory_available_mb=0,
                memory_utilization=0.0,
                temperature=0.0,
                power_usage=0.0,
                optimization_enabled=False,
                rtx_3080_detected=False
            )
        
        config = gpu_manager.get_config()
        memory_info = gpu_manager.get_memory_info()
        optimization_status = gpu_optimizer.get_optimization_status()
        
        # Get temperature and power usage if available
        temperature = 0.0
        power_usage = 0.0
        
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                temperature = gpu.temperature
                # Power usage not available in GPUtil, would need nvidia-ml-py
        except ImportError:
            pass
        
        return GPUStatusResponse(
            gpu_available=True,
            device_name=config.device_name,
            device_type=config.device,
            memory_total_mb=config.memory_total,
            memory_available_mb=memory_info.get("free_mb", 0),
            memory_utilization=memory_info.get("utilization", 0.0),
            temperature=temperature,
            power_usage=power_usage,
            compute_capability=config.compute_capability,
            cuda_version=config.cuda_version,
            optimization_enabled=optimization_status["initialized"],
            rtx_3080_detected=optimization_status["rtx_3080_detected"]
        )
        
    except Exception as e:
        logger.error(f"Failed to get GPU status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get GPU status: {str(e)}")


@router.get("/performance", response_model=GPUPerformanceResponse)
async def get_gpu_performance():
    """Get comprehensive GPU performance metrics"""
    try:
        # Get current optimization status
        optimization_status = gpu_optimizer.get_optimization_status()
        
        # Get performance summary
        performance_summary = gpu_optimizer.get_performance_summary()
        
        # Get Ollama performance if available
        ollama_performance = {}
        if ollama_gpu_manager.is_initialized:
            ollama_performance = ollama_gpu_manager.get_performance_summary()
        
        # Get current metrics
        current_metrics = {}
        if gpu_manager.is_available():
            current_metrics = gpu_manager.get_memory_info()
            
            # Add system info
            system_info = gpu_manager.get_system_info()
            current_metrics.update(system_info)
        
        return GPUPerformanceResponse(
            current_metrics=current_metrics,
            performance_summary=performance_summary,
            optimization_status=optimization_status,
            ollama_performance=ollama_performance
        )
        
    except Exception as e:
        logger.error(f"Failed to get GPU performance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get GPU performance: {str(e)}")


@router.post("/optimize")
async def optimize_gpu(background_tasks: BackgroundTasks):
    """Trigger GPU optimization and memory cleanup"""
    try:
        if not gpu_manager.is_available():
            raise HTTPException(status_code=400, detail="No GPU available for optimization")
        
        # Clear memory cache
        gpu_optimizer.clear_memory_cache()
        
        # Re-initialize optimization if needed
        if not gpu_optimizer.is_initialized:
            background_tasks.add_task(gpu_optimizer.initialize)
        
        return {
            "status": "success",
            "message": "GPU optimization triggered",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to optimize GPU: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to optimize GPU: {str(e)}")


@router.post("/config")
async def update_optimization_config(config: OptimizationConfigRequest):
    """Update GPU optimization configuration"""
    try:
        if not gpu_optimizer.is_initialized:
            raise HTTPException(status_code=400, detail="GPU optimizer not initialized")
        
        # Update configuration
        current_config = gpu_optimizer.config
        
        if config.memory_fraction is not None:
            current_config.memory_fraction = max(0.1, min(1.0, config.memory_fraction))
        
        if config.use_mixed_precision is not None:
            current_config.use_mixed_precision = config.use_mixed_precision
        
        if config.use_tensor_cores is not None:
            current_config.use_tensor_cores = config.use_tensor_cores
        
        if config.quantization_enabled is not None:
            current_config.quantization_enabled = config.quantization_enabled
        
        if config.performance_monitoring is not None:
            current_config.performance_monitoring = config.performance_monitoring
        
        return {
            "status": "success",
            "message": "GPU optimization configuration updated",
            "config": {
                "memory_fraction": current_config.memory_fraction,
                "use_mixed_precision": current_config.use_mixed_precision,
                "use_tensor_cores": current_config.use_tensor_cores,
                "quantization_enabled": current_config.quantization_enabled,
                "performance_monitoring": current_config.performance_monitoring
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to update optimization config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")


@router.get("/models")
async def get_gpu_optimized_models():
    """Get list of GPU-optimized models and their performance"""
    try:
        if not ollama_gpu_manager.is_initialized:
            return {
                "status": "ollama_not_initialized",
                "models": [],
                "message": "Ollama GPU manager not initialized"
            }
        
        models_info = []
        for model_name, config in ollama_gpu_manager.available_models.items():
            model_info = {
                "name": model_name,
                "size_gb": config.size_gb,
                "context_length": config.context_length,
                "gpu_layers": config.gpu_layers,
                "gpu_memory_required_gb": config.gpu_memory_required_gb,
                "recommended_batch_size": config.recommended_batch_size,
                "optimization_level": config.optimization_level,
                "loaded": model_name in ollama_gpu_manager.loaded_models
            }
            
            # Add performance data if available
            if model_name in ollama_gpu_manager.model_performance:
                perf_data = ollama_gpu_manager.model_performance[model_name]
                if perf_data:
                    model_info["performance"] = {
                        "avg_tokens_per_second": sum(perf_data) / len(perf_data),
                        "max_tokens_per_second": max(perf_data),
                        "sample_count": len(perf_data)
                    }
            
            models_info.append(model_info)
        
        return {
            "status": "success",
            "gpu_optimized": ollama_gpu_manager.gpu_optimized,
            "max_gpu_memory_gb": ollama_gpu_manager.max_gpu_memory_gb,
            "loaded_models": ollama_gpu_manager.loaded_models,
            "current_model": ollama_gpu_manager.current_model,
            "models": models_info
        }
        
    except Exception as e:
        logger.error(f"Failed to get GPU models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get GPU models: {str(e)}")


@router.post("/models/{model_name}/load")
async def load_gpu_model(model_name: str, background_tasks: BackgroundTasks):
    """Load a specific model with GPU optimization"""
    try:
        if not ollama_gpu_manager.is_initialized:
            raise HTTPException(status_code=400, detail="Ollama GPU manager not initialized")
        
        if model_name not in ollama_gpu_manager.available_models:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not available")
        
        # Load model in background
        background_tasks.add_task(ollama_gpu_manager._ensure_model_loaded, model_name)
        
        return {
            "status": "success",
            "message": f"Loading model {model_name} with GPU optimization",
            "model": model_name,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to load GPU model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@router.get("/benchmark")
async def run_gpu_benchmark():
    """Run GPU performance benchmark"""
    try:
        if not gpu_manager.is_available():
            raise HTTPException(status_code=400, detail="No GPU available for benchmarking")
        
        benchmark_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "device": gpu_manager.config.device_name,
            "tests": {}
        }
        
        # Memory bandwidth test
        import torch
        device = gpu_manager.get_device()
        
        # Test 1: Memory allocation/deallocation speed
        start_time = datetime.utcnow()
        test_tensor = torch.randn(1000, 1000, device=device)
        allocation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Test 2: Matrix multiplication performance
        start_time = datetime.utcnow()
        result = torch.matmul(test_tensor, test_tensor)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        matmul_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Test 3: Memory cleanup
        start_time = datetime.utcnow()
        del test_tensor, result
        gpu_optimizer.clear_memory_cache()
        cleanup_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        benchmark_results["tests"] = {
            "memory_allocation_ms": allocation_time,
            "matrix_multiplication_ms": matmul_time,
            "memory_cleanup_ms": cleanup_time,
            "total_time_ms": allocation_time + matmul_time + cleanup_time
        }
        
        # Add memory info
        memory_info = gpu_manager.get_memory_info()
        benchmark_results["memory_info"] = memory_info
        
        return benchmark_results
        
    except Exception as e:
        logger.error(f"GPU benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")


@router.get("/health")
async def gpu_health_check():
    """Comprehensive GPU health check"""
    try:
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy",
            "checks": {}
        }
        
        # Check GPU availability
        health_status["checks"]["gpu_available"] = {
            "status": "pass" if gpu_manager.is_available() else "fail",
            "message": f"GPU: {gpu_manager.config.device_name}" if gpu_manager.is_available() else "No GPU detected"
        }
        
        # Check GPU optimization
        health_status["checks"]["optimization_enabled"] = {
            "status": "pass" if gpu_optimizer.is_initialized else "fail",
            "message": "GPU optimization active" if gpu_optimizer.is_initialized else "GPU optimization not initialized"
        }
        
        # Check memory usage
        if gpu_manager.is_available():
            memory_info = gpu_manager.get_memory_info()
            memory_util = memory_info.get("utilization", 0)
            
            health_status["checks"]["memory_usage"] = {
                "status": "pass" if memory_util < 0.9 else "warning" if memory_util < 0.95 else "critical",
                "message": f"Memory utilization: {memory_util:.1%}",
                "utilization": memory_util
            }
        
        # Check Ollama GPU integration
        health_status["checks"]["ollama_gpu"] = {
            "status": "pass" if ollama_gpu_manager.is_initialized else "warning",
            "message": "Ollama GPU integration active" if ollama_gpu_manager.is_initialized else "Ollama GPU not initialized"
        }
        
        # Determine overall status
        check_statuses = [check["status"] for check in health_status["checks"].values()]
        if "critical" in check_statuses:
            health_status["overall_status"] = "critical"
        elif "fail" in check_statuses:
            health_status["overall_status"] = "unhealthy"
        elif "warning" in check_statuses:
            health_status["overall_status"] = "warning"
        
        return health_status
        
    except Exception as e:
        logger.error(f"GPU health check failed: {e}")
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "error",
            "error": str(e)
        }
